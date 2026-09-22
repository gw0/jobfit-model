#!/usr/bin/env python3
"""`calibrate` stage (specs §4): split-conformal calibration of the finetuned model's
quantiles on `calib` (90% target coverage, per aspect), plus each aspect's
"insufficient data" interval-width threshold. Inference only; the backbone hash is
asserted unchanged.

Usage:
    ./pipeline/calibrate.py --dataset-dir datasets_smoke --runs-dir runs_smoke

Writes <runs-dir>/<slug>/calibration/params.json:
{aspect_id: {"delta", "insufficient_data_threshold"}}, all null when `calib` is empty.
"""

import argparse
import math

import common
import metrics


def _run(args):
    import mlflow

    slug = common.model_slug(args.model)
    checkpoint_dir = args.runs_dir / slug / "checkpoints" / "finetune"
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"no finetune checkpoint under {checkpoint_dir} -- run train.py --stage finetune first")
    out_path = args.runs_dir / slug / "calibration" / "params.json"

    calib_cache = common.load_cache(args.runs_dir, slug, "calib")
    if calib_cache is None:
        print("0 calib pairs -- skipped, writing empty calibration")
        common.write_json(out_path, {aid: {"delta": math.nan, "insufficient_data_threshold": math.nan}
                                     for aid in common.ASPECT_IDS})
        return

    tokenizer = common.load_tokenizer(checkpoint_dir)
    model = common.load_classification_model(checkpoint_dir, tokenizer.pad_token_id).eval()
    hash_before = common.backbone_state_dict_hash(model)
    preds_q = common.predict_quantiles_batched(
        model, calib_cache["input_ids"], calib_cache["attention_mask"], args.batch_size
    )
    if common.backbone_state_dict_hash(model) != hash_before:
        raise SystemExit("BACKBONE HASH CHANGED during calibrate -- inference must not touch weights")

    targets, confidences = common.build_targets(calib_cache["pairs"], common.load_labels(args.dataset_dir))
    params = metrics.fit_calibration(preds_q, targets, confidences, args.confidence_floor)
    common.write_json(out_path, params)
    print(f"fit conformal calibration on {len(calib_cache['pairs'])} calib pair(s) -> {out_path}")

    with common.mlflow_run("calibrate", args.model, args.run_group):
        mlflow.log_param("calib_pairs", len(calib_cache["pairs"]))
        common.log_dataset_input(calib_cache, args.runs_dir, slug, "calib", context="calibration")
        mlflow.log_metrics({f"delta_{aid}": p["delta"] for aid, p in params.items() if not math.isnan(p["delta"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    args = parser.parse_args()
    with common.stage_span("calibrate"):
        _run(args)


if __name__ == "__main__":
    main()
