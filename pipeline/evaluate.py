#!/usr/bin/env python3
"""`evaluate` stage (specs §4), run after every stage that changes the model.

Per-aspect MAE and Spearman rho against the train-mean and keyword-overlap baselines,
bootstrap MAE CIs over CVs, the shuffled-pair control and, for calibrated stages,
coverage and mean interval width. The model and calibration are chosen by --stage:

    headtrained  checkpoints/headtrain
    finetuned    checkpoints/finetune
    calibrated   checkpoints/finetune + calibration/params.json

Usage:
    ./pipeline/evaluate.py --dataset-dir datasets_smoke --runs-dir runs_smoke --stage finetuned

Writes <runs-dir>/<slug>/eval/<stage>.json; an empty split is recorded as skipped.
"""

import argparse

import numpy as np

import common
import metrics
from common import corpus


def model_dir(runs_dir, slug, stage):
    base = runs_dir / slug
    return {
        "headtrained": base / "checkpoints" / "headtrain",
        "finetuned": base / "checkpoints" / "finetune",
        "calibrated": base / "checkpoints" / "finetune",
    }[stage]


def calibration_path(runs_dir, slug, stage):
    return runs_dir / slug / "calibration" / "params.json" if stage == "calibrated" else None


def _load_model(path):
    tokenizer = common.load_tokenizer(path)
    return common.load_classification_model(path, tokenizer.pad_token_id).eval()


def _overlaps(dataset_dir, pairs):
    cvs = {p["cv"]: (dataset_dir / p["cv"]).read_text(encoding="utf-8") for p in pairs}
    jobs = {p["job"]: corpus.read_job_body(dataset_dir / p["job"]) for p in pairs}
    return [metrics.keyword_overlap(cvs[p["cv"]], jobs[p["job"]]) for p in pairs]


def shuffle_control(preds_q, shuffled_preds_q, targets, confidences, confidence_floor):
    """Per aspect: how far predictions move when each CV is paired with an unrelated JD
    (mean |mid_true - mid_shuffled| over all pairs), and the shuffled predictions' MAE
    against the true pairs' labels. A model that ignores the JD shows a shift near 0."""
    shift = np.abs(preds_q[..., common.MID] - shuffled_preds_q[..., common.MID]).mean(axis=0)
    shuffled_mae = metrics.per_aspect_mae(shuffled_preds_q, targets, confidences, confidence_floor)
    return {aid: {"mean_abs_shift": float(shift[j]), "mae": shuffled_mae[aid]}
            for j, aid in enumerate(common.ASPECT_IDS)}


def baselines(dataset_dir, train_pairs, eval_pairs, labels, targets, confidences, confidence_floor):
    train_targets, train_conf = common.build_targets(train_pairs, labels)
    preds = {
        "train_mean": metrics.train_mean_preds(train_targets, train_conf, confidence_floor, len(eval_pairs)),
        "keyword_overlap": metrics.keyword_overlap_preds(
            _overlaps(dataset_dir, train_pairs), train_targets, train_conf, confidence_floor,
            _overlaps(dataset_dir, eval_pairs)),
    }
    return {
        name: {aid: {"mae": m} for aid, m in metrics.per_aspect_mae(p, targets, confidences, confidence_floor).items()}
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

    slug = common.model_slug(args.model)
    out_path = args.runs_dir / slug / "eval" / f"{args.stage}.json"
    cache = common.load_cache(args.runs_dir, slug, args.split)
    if cache is None:
        print(f"0 pairs in split {args.split!r} -- skipped")
        common.write_json(out_path, {"split": args.split, "stage": args.stage, "model": args.model,
                                     "n": 0, "skipped": True})
        return

    path = model_dir(args.runs_dir, slug, args.stage)
    model = _load_model(path)
    preds_q = common.predict_quantiles_batched(model, cache["input_ids"], cache["attention_mask"], args.batch_size)

    labels = common.load_labels(args.dataset_dir)
    targets, confidences = common.build_targets(cache["pairs"], labels)
    floor = args.confidence_floor
    report_metrics = metrics.compute_metrics(preds_q, targets, confidences, floor)

    calib_path = calibration_path(args.runs_dir, slug, args.stage)
    if calib_path is not None:
        coverage = metrics.compute_coverage(preds_q, targets, confidences, common.read_json(calib_path), floor)
        for aid in common.ASPECT_IDS:
            report_metrics[aid].update(coverage[aid])

    train_cache = common.load_cache(args.runs_dir, slug, "train")
    shuffled_cache = common.load_cache(args.runs_dir, slug, f"{args.split}_shuffled")
    shuffled = None
    if shuffled_cache is not None:
        shuffled_preds_q = common.predict_quantiles_batched(
            model, shuffled_cache["input_ids"], shuffled_cache["attention_mask"], args.batch_size)
        shuffled = shuffle_control(preds_q, shuffled_preds_q, targets, confidences, floor)

    report = {
        "split": args.split, "stage": args.stage, "model": args.model,
        "n": len(cache["pairs"]), "confidence_floor": floor,
        "metrics": report_metrics,
        "baselines": (baselines(args.dataset_dir, train_cache["pairs"], cache["pairs"], labels,
                                targets, confidences, floor) if train_cache is not None else {}),
        "bootstrap_mae_ci": metrics.bootstrap_mae_ci(preds_q, targets, confidences, floor,
                                                     [p["cv"] for p in cache["pairs"]]),
        "shuffle_control": shuffled,
        "jevbench": _run_jevbench(path) if args.run_jevbench else None,
    }
    common.write_json(out_path, report)
    print(f"evaluated {report['n']} pair(s) on split {args.split!r} -> {out_path}")

    with common.mlflow_run("evaluate", args.model, args.run_group, tags={"eval_stage": args.stage}):
        mlflow.log_params({"split": args.split, "n_pairs": report["n"]})
        summary = {
            "mean_mae": metrics.nanmean([m["mae"] for m in report_metrics.values()]),
            "mean_abs_shift": metrics.nanmean([s["mean_abs_shift"] for s in shuffled.values()]) if shuffled else np.nan,
        }
        mlflow.log_metrics({k: v for k, v in summary.items() if not np.isnan(v)})
        common.log_dataset_input(cache, args.runs_dir, slug, args.split, context="evaluation")
        if train_cache is not None:
            common.log_dataset_input(train_cache, args.runs_dir, slug, "train", context="baseline_fit")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument("--stage", required=True, choices=common.STAGES)
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-jevbench", action="store_true")
    args = parser.parse_args()
    with common.stage_span("evaluate"):
        _run(args)


if __name__ == "__main__":
    main()
