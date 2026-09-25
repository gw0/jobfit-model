#!/usr/bin/env python3
"""`evaluate` stage (specs §4), run after every stage that changes the model.

Per-question MAE and Spearman rho of the [0,1] score (score / (K-1)) against the
train-mean and keyword-overlap baselines, bootstrap MAE CIs over CVs, the shuffled-pair
control and, for calibrated stages, confidence-vs-error and the insufficient-data rate.
The model and calibration are chosen by --stage:

    zeroshot     the base --model, no training (temperature 1)
    finetuned    checkpoints/finetune (temperature 1)
    calibrated   checkpoints/finetune + calibration/params.json
    quantized    export/quantized (ONNX) + calibration/params.json (pre-quantization fit)

Usage:
    ./pipeline/evaluate.py --dataset-dir datasets_smoke --runs-dir runs_smoke --stage finetuned

Writes <runs-dir>/<candidate>/eval/<stage>.json; an empty split is recorded as skipped.
"""

import argparse

import numpy as np

import common
import metrics
from common import corpus


def model_path(args):
    return {
        "zeroshot": args.model,
        "finetuned": args.run_dir / "checkpoints" / "finetune",
        "calibrated": args.run_dir / "checkpoints" / "finetune",
        "quantized": args.run_dir / "export" / "quantized" / "model_quantized.onnx",
    }[args.stage]


def load_calibration(args):
    """The fixed calibration this stage is evaluated under; None before `calibrate`."""
    if args.stage not in ("calibrated", "quantized"):
        return None
    return common.read_json(args.run_dir / "calibration" / "params.json")


def _load_model(stage, path):
    if stage == "quantized":
        import onnxruntime

        return onnxruntime.InferenceSession(str(path))
    return common.load_jev_model(path).eval()


def _overlaps(dataset_dir, pairs):
    cvs = {p["cv"]: (dataset_dir / p["cv"]).read_text(encoding="utf-8") for p in pairs}
    jobs = {p["job"]: corpus.read_job_body(dataset_dir / p["job"]) for p in pairs}
    return [metrics.keyword_overlap(cvs[p["cv"]], jobs[p["job"]]) for p in pairs]


def shuffle_control(scores, shuffled_scores, targets, confidences, confidence_floor):
    """Per question: how far scores move when each CV is paired with an unrelated JD
    (mean |true - shuffled| over all pairs), and the shuffled scores' MAE against the
    true pairs' labels. A model that ignores the JD shows a shift near 0."""
    shift = np.abs(scores - shuffled_scores).mean(axis=0)
    shuffled_mae = metrics.per_question_mae(shuffled_scores, targets, confidences, confidence_floor)
    return {qid: {"mean_abs_shift": float(shift[j]), "mae": shuffled_mae[qid]}
            for j, qid in enumerate(common.QUESTION_IDS)}


def baselines(dataset_dir, train_pairs, eval_pairs, labels, targets, confidences, confidence_floor):
    train_targets, train_conf = common.build_targets(train_pairs, labels)
    preds = {
        "train_mean": metrics.train_mean_preds(train_targets, train_conf, confidence_floor, len(eval_pairs)),
        "keyword_overlap": metrics.keyword_overlap_preds(
            _overlaps(dataset_dir, train_pairs), train_targets, train_conf, confidence_floor,
            _overlaps(dataset_dir, eval_pairs)),
    }
    return {
        name: {qid: {"mae": m} for qid, m in metrics.per_question_mae(p, targets, confidences, confidence_floor).items()}
        for name, p in preds.items()
    }


def _run_jevbench(model_path):
    """Informational side-eval (specs §4); never blocks the stage."""
    try:
        import jevbench
    except ImportError:
        print("  JevBench: package not available -- skipped (informational only)")
        return None
    try:
        return jevbench.run(model_path)
    except Exception as exc:  # noqa: BLE001 -- API surface unverified; a side-eval must not fail the stage
        print(f"  JevBench: run failed ({exc}) -- skipped (informational only)")
        return None


def _run(args):
    import mlflow

    out_path = args.run_dir / "eval" / f"{args.stage}.json"
    cache = common.load_cache(args.run_dir, args.split)
    if cache is None:
        print(f"0 pairs in split {args.split!r} -- skipped")
        common.write_json(out_path, {"split": args.split, "stage": args.stage, "model": args.model,
                                     "n": 0, "skipped": True})
        return

    path = model_path(args)
    model = _load_model(args.stage, path)
    calibration = load_calibration(args)
    temperature = common.temperature(calibration)
    scores, answer_confidences = common.read_scores(
        common.predict_answer_logits(model, cache, args.batch_size), temperature)

    labels = common.load_labels(args.dataset_dir)
    targets, confidences = common.build_targets(cache["pairs"], labels)
    floor = args.confidence_floor
    report_metrics = metrics.compute_metrics(scores, targets, confidences, floor)
    if calibration is not None:
        calibrated = metrics.confidence_error(scores, answer_confidences, targets, confidences, floor,
                                              calibration["confidence_threshold"])
        for qid in common.QUESTION_IDS:
            report_metrics[qid].update(calibrated[qid])

    train_cache = common.load_cache(args.run_dir, "train")
    shuffled_cache = common.load_cache(args.run_dir, f"{args.split}_shuffled")
    shuffled = None
    if shuffled_cache is not None:
        shuffled_scores, _ = common.read_scores(
            common.predict_answer_logits(model, shuffled_cache, args.batch_size), temperature)
        shuffled = shuffle_control(scores, shuffled_scores, targets, confidences, floor)

    report = {
        "split": args.split, "stage": args.stage, "model": args.model,
        "n": len(cache["pairs"]), "confidence_floor": floor,
        "metrics": report_metrics,
        "baselines": (baselines(args.dataset_dir, train_cache["pairs"], cache["pairs"], labels,
                                targets, confidences, floor) if train_cache is not None else {}),
        "calibration": calibration,
        "bootstrap_mae_ci": metrics.bootstrap_mae_ci(scores, targets, confidences, floor,
                                                     [p["cv"] for p in cache["pairs"]]),
        "shuffle_control": shuffled,
        "jevbench": _run_jevbench(path) if args.run_jevbench else None,
    }
    common.write_json(out_path, report)
    print(f"evaluated {report['n']} pair(s) on split {args.split!r} -> {out_path}")

    with common.mlflow_run("evaluate", args, tags={"eval_stage": args.stage}):
        mlflow.log_params({"split": args.split, "n_pairs": report["n"]})
        summary = {
            "mean_mae": metrics.nanmean([m["mae"] for m in report_metrics.values()]),
            "mean_abs_shift": metrics.nanmean([s["mean_abs_shift"] for s in shuffled.values()]) if shuffled else np.nan,
        }
        mlflow.log_metrics({k: v for k, v in summary.items() if not np.isnan(v)})
        common.log_dataset_input(cache, args.run_dir, args.split, context="evaluation")
        if train_cache is not None:
            common.log_dataset_input(train_cache, args.run_dir, "train", context="baseline_fit")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument("--stage", required=True, choices=common.STAGES)
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-jevbench", action="store_true")
    args = common.parse_args(parser)
    with common.stage_span("evaluate"):
        _run(args)


if __name__ == "__main__":
    main()
