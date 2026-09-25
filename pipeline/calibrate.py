#!/usr/bin/env python3
"""`calibrate` stage (specs/20260923-jev-model.md §4): fits, on `calib`, the finetuned
model's one temperature (pooled over every question, so it applies to any question)
and JobFit's one "insufficient data" confidence threshold. Inference only; the weight
hash is asserted unchanged.

Usage:
    ./pipeline/calibrate.py --dataset-dir datasets_smoke --runs-dir runs_smoke

Writes <runs-dir>/<candidate>/calibration/params.json: {"temperature", "confidence_threshold"},
both null when `calib` is empty.
"""

import argparse
import math

import common
import metrics


def _run(args):
    import mlflow

    checkpoint_dir = args.run_dir / "checkpoints" / "finetune"
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"no finetune checkpoint under {checkpoint_dir} -- run train.py first")
    out_path = args.run_dir / "calibration" / "params.json"

    calib_cache = common.load_cache(args.run_dir, "calib")
    if calib_cache is None:
        print("0 calib pairs -- skipped, writing empty calibration")
        common.write_json(out_path, {"temperature": math.nan, "confidence_threshold": math.nan})
        return

    model = common.load_jev_model(checkpoint_dir).eval()
    hash_before = common.state_dict_hash(model)
    answer_logits = common.predict_answer_logits(model, calib_cache, args.batch_size)
    if common.state_dict_hash(model) != hash_before:
        raise SystemExit("WEIGHT HASH CHANGED during calibrate -- inference must not touch weights")

    targets, confidences = common.build_targets(calib_cache["pairs"], common.load_labels(args.dataset_dir))
    params = metrics.fit_calibration(answer_logits, targets, confidences, args.confidence_floor)
    common.write_json(out_path, params)
    print(f"fit calibration on {len(calib_cache['pairs'])} calib pair(s) -> {out_path}: {params}")

    with common.mlflow_run("calibrate", args):
        mlflow.log_param("calib_pairs", len(calib_cache["pairs"]))
        common.log_dataset_input(calib_cache, args.run_dir, "calib", context="calibration")
        mlflow.log_metrics({k: v for k, v in params.items() if not math.isnan(v)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    args = common.parse_args(parser, "calibrate")
    with common.stage_span("calibrate"):
        _run(args)


if __name__ == "__main__":
    main()
