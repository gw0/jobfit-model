#!/usr/bin/env python3
"""`headtrain` / `finetune` stages (specs §4): linear-probe-then-fine-tune of the
quantile head with an HF Trainer and a confidence-weighted pinball loss.

- headtrain: backbone frozen (asserted via its weight hash), only the head trains.
- finetune: starts from the headtrain checkpoint, LoRA on the attention/MLP
  projections plus the trainable head; adapters are merged before saving, so every
  later stage sees a plain dense model.

With a `val` split the best epoch by val MAE is kept, otherwise the last one.

Usage:
    ./pipeline/train.py --stage headtrain --dataset-dir datasets_smoke --runs-dir runs_smoke
    ./pipeline/train.py --stage finetune --dataset-dir datasets_smoke --runs-dir runs_smoke

Reads <runs-dir>/<slug>/cache/{train,val}.pt, writes <runs-dir>/<slug>/checkpoints/<stage>/.
"""

import argparse
import shutil

import transformers

import common
import metrics

DEFAULTS = {
    "headtrain": {"epochs": 30, "lr": 1e-3},
    "finetune": {"epochs": 5, "lr": 1e-4},
}
LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LABEL_NAMES = ["targets", "confidences"]


class QuantileTrainer(transformers.Trainer):
    def __init__(self, *args, confidence_floor, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_floor = confidence_floor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        loss = common.pinball_loss(
            common.reshape_quantile_logits(outputs.logits),
            inputs["targets"], inputs["confidences"], self.confidence_floor,
        )
        return (loss, outputs) if return_outputs else loss


def make_dataset(cache, targets, confidences):
    import torch

    targets = torch.tensor(targets, dtype=torch.float32)
    confidences = torch.tensor(confidences, dtype=torch.float32)
    return [
        {"input_ids": cache["input_ids"][i], "attention_mask": cache["attention_mask"][i],
         "targets": targets[i], "confidences": confidences[i]}
        for i in range(len(cache["pairs"]))
    ]


def make_compute_metrics(confidence_floor):
    def compute_metrics(eval_pred):
        preds_q = common.reshape_quantile_logits(eval_pred.predictions)
        targets, confidences = eval_pred.label_ids
        return {"mae": metrics.mean_mae(preds_q, targets, confidences, confidence_floor)}
    return compute_metrics


def _build_model(args, slug):
    """(model, tokenizer) for the requested stage."""
    if args.stage == "headtrain":
        tokenizer = common.load_tokenizer(args.model)
        model = common.load_classification_model(args.model, tokenizer.pad_token_id)
        common.freeze_backbone(model)
        return model, tokenizer

    from peft import LoraConfig, get_peft_model

    headtrain_dir = args.runs_dir / slug / "checkpoints" / "headtrain"
    if not headtrain_dir.is_dir():
        raise SystemExit(f"no headtrain checkpoint under {headtrain_dir} -- run --stage headtrain first")
    tokenizer = common.load_tokenizer(headtrain_dir)
    model = common.load_classification_model(headtrain_dir, tokenizer.pad_token_id)
    # LoRA adapters sit in every layer, so gradients flow through the whole backbone;
    # recomputing activations in the backward pass keeps that tractable on CPU.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                      target_modules=args.lora_target_modules.split(","), modules_to_save=["score"])
    return get_peft_model(model, lora), tokenizer


def _run(args):
    slug = common.model_slug(args.model)
    train_cache = common.load_cache(args.runs_dir, slug, "train")
    if train_cache is None:
        raise SystemExit(f"no train cache under {args.runs_dir / slug / 'cache'} -- run prepare.py first")
    val_cache = common.load_cache(args.runs_dir, slug, "val")
    labels = common.load_labels(args.dataset_dir)

    transformers.set_seed(args.seed)
    model, tokenizer = _build_model(args, slug)
    hash_before = common.backbone_state_dict_hash(model) if args.stage == "headtrain" else None

    out_dir = args.runs_dir / slug / "checkpoints" / args.stage
    trainer_dir = out_dir.with_name(f"{args.stage}-trainer")
    strategy = "epoch" if val_cache is not None else "no"
    training_args = transformers.TrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        seed=args.seed,
        eval_strategy=strategy,
        save_strategy=strategy,
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=val_cache is not None,
        metric_for_best_model="mae",
        greater_is_better=False,
        logging_strategy="epoch",
        label_names=LABEL_NAMES,
        remove_unused_columns=False,
        report_to="none",
        run_name=f"{args.stage}-{slug}",
    )
    trainer = QuantileTrainer(
        model=model,
        args=training_args,
        train_dataset=make_dataset(train_cache, *common.build_targets(train_cache["pairs"], labels)),
        eval_dataset=(make_dataset(val_cache, *common.build_targets(val_cache["pairs"], labels))
                      if val_cache is not None else None),
        compute_metrics=make_compute_metrics(args.confidence_floor),
        confidence_floor=args.confidence_floor,
    )

    if val_cache is None:
        print("no val split -- keeping the final epoch's weights")

    trainer.train()
    model = trainer.model
    if args.stage == "headtrain":
        if common.backbone_state_dict_hash(model) != hash_before:
            raise SystemExit("BACKBONE HASH CHANGED during headtrain -- freezing is broken")
        print("backbone hash unchanged: freezing verified")
    else:
        model = model.merge_and_unload()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    shutil.rmtree(trainer_dir, ignore_errors=True)
    print(f"saved {args.stage} checkpoint -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument("--stage", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--epochs", type=int, default=None, help="default: 30 headtrain, 5 finetune")
    parser.add_argument("--lr", type=float, default=None, help="default: 1e-3 headtrain, 1e-4 finetune")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target-modules", default=LORA_TARGET_MODULES)
    args = parser.parse_args()
    for key, value in DEFAULTS[args.stage].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    _run(args)


if __name__ == "__main__":
    main()
