#!/usr/bin/env python3
"""`finetune` stage (specs/20260923-jev-model.md §5): LoRA on the attention/MLP
projections of the base causal LM, trained through the Jev readout -- each question's
answer is its lm_head distribution over its candidate tokens, fit to the two-hot
target of the judge's label with a confidence-weighted soft cross-entropy. Adapters are
merged before saving, so every later stage sees a plain dense causal LM. There is no
trained head: the untrained base model is the `zeroshot` evaluation.

With a `val` split the best epoch by val loss is kept, otherwise the last one.
The Trainer logs to the active MLflow run.

Usage:
    ./pipeline/train.py --dataset-dir datasets_smoke --runs-dir runs_smoke

Reads <runs-dir>/<candidate>/cache/{train,val}.pt, writes <runs-dir>/<candidate>/checkpoints/finetune/.
"""

import argparse
import shutil

import torch
import transformers

import common
import jev_model

LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LABEL_NAMES = ["targets", "weights"]


class JevTrainer(transformers.Trainer):
    def __init__(self, *args, candidate_ids, candidate_counts, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_ids = candidate_ids
        self.candidate_mask = jev_model.candidate_mask(candidate_counts)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        logits = model(inputs["input_ids"], inputs["segment_ids"], inputs["answer_positions"],
                       self.candidate_ids.to(inputs["input_ids"].device))
        loss = jev_model.answer_loss(logits, inputs["targets"], inputs["weights"],
                                     self.candidate_mask.to(logits.device))
        return (loss, {"answer_logits": logits}) if return_outputs else loss


def make_dataset(cache, targets, confidences, confidence_floor):
    """One row per pair: the model inputs, (Q, C) target distributions and (Q,) loss weights."""
    dists = torch.tensor(common.answer_targets(targets), dtype=torch.float32)
    weights = torch.tensor(common.label_weights(targets, confidences, confidence_floor), dtype=torch.float32)
    return [
        {**{name: cache[name][i] for name in common.MODEL_INPUTS},
         "targets": dists[i], "weights": weights[i]}
        for i in range(len(cache["pairs"]))
    ]


def _build_model(args):
    from peft import LoraConfig, get_peft_model

    tokenizer = common.load_tokenizer(args.model)
    model = common.load_jev_model(args.model)
    # LoRA adapters sit in every layer, so gradients flow through the whole backbone;
    # recomputing activations in the backward pass keeps that tractable on CPU.
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                      target_modules=args.lora_target_modules.split(","))
    peft_lm = get_peft_model(model.lm, lora)  # injects the adapters into model.lm in place
    return model, peft_lm, tokenizer


def _run(args):
    import mlflow

    train_cache = common.load_cache(args.run_dir, "train")
    if train_cache is None:
        raise SystemExit(f"no train cache under {args.run_dir / 'cache'} -- run prepare.py first")
    val_cache = common.load_cache(args.run_dir, "val")
    labels = common.load_labels(args.dataset_dir)

    transformers.set_seed(args.seed)
    model, peft_lm, tokenizer = _build_model(args)

    def dataset(cache):
        return make_dataset(cache, *common.build_targets(cache["pairs"], labels), args.confidence_floor)

    out_dir = args.run_dir / "checkpoints" / "finetune"
    trainer_dir = out_dir.with_name("finetune-trainer")
    strategy = "epoch" if val_cache is not None else "no"
    training_args = transformers.TrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        seed=args.seed,
        bf16=torch.cuda.is_available(),  # autocast only: the weights stay fp32 (common.load_jev_model)
        eval_strategy=strategy,
        save_strategy=strategy,
        save_total_limit=1,
        save_only_model=True,
        save_safetensors=False,  # a tied lm_head shares its tensor with the embeddings
        load_best_model_at_end=val_cache is not None,
        metric_for_best_model="loss",
        prediction_loss_only=True,
        logging_strategy="epoch",
        label_names=LABEL_NAMES,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name=f"finetune-{args.candidate}",
    )
    trainer = JevTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset(train_cache),
        eval_dataset=dataset(val_cache) if val_cache is not None else None,
        candidate_ids=train_cache["candidate_ids"],
        candidate_counts=train_cache["candidate_counts"],
    )

    with common.mlflow_run("finetune", args):
        mlflow.log_params({"confidence_floor": args.confidence_floor,
                           "train_pairs": len(train_cache["pairs"]),
                           "val_pairs": 0 if val_cache is None else len(val_cache["pairs"]),
                           "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                           "lora_target_modules": args.lora_target_modules})
        common.log_dataset_input(train_cache, args.run_dir, "train", context="training")
        if val_cache is not None:
            common.log_dataset_input(val_cache, args.run_dir, "val", context="training")
        else:
            print("no val split -- keeping the final epoch's weights")

        trainer.train()
        model.lm = peft_lm.merge_and_unload()
        model.lm.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        shutil.rmtree(trainer_dir, ignore_errors=True)
        print(f"saved finetune checkpoint -> {out_dir}")
        common.log_model_signature(model.eval(), train_cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target-modules", default=LORA_TARGET_MODULES)
    args = common.parse_args(parser, "finetune")
    with common.stage_span("finetune"):
        _run(args)


if __name__ == "__main__":
    main()
