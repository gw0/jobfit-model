"""Benchmark report (specs §10): picking the winning candidate, assembling the
self-contained report dict from per-stage eval results, and rendering it as Markdown.
Pure functions, no I/O."""

import math

import metrics
from common import NUM_ASPECTS, STAGES

STAGE_LABELS = {
    "headtrained": "head-trained (backbone frozen)",
    "finetuned": "fine-tuned (LoRA, end-to-end)",
    "calibrated": "calibrated (conformal, pre-quantization)",
    "quantized": "quantized (shipped, int8)",
}

LIMITATIONS = [
    "The test split covers few distinct CV profiles, so candidate-axis generalisation is "
    "the weakest claim here; read test metrics with the bootstrap CIs in the per-stage "
    "eval JSONs, not as point estimates.",
]


def _usable(result):
    return bool(result) and not result.get("skipped")


def mean_mae(result):
    if not _usable(result):
        return math.nan
    return metrics.nanmean([m["mae"] for m in result["metrics"].values()])


def beats_floor_count(result):
    """Aspects whose MAE beats the train-mean baseline (acceptance criterion 2)."""
    if not _usable(result):
        return None
    floor = result.get("baselines", {}).get("train_mean", {})
    return sum(
        1 for aid, m in result["metrics"].items()
        if m["mae"] < floor.get(aid, {}).get("mae", math.nan)  # NaN compares False
    )


def candidate_score(by_stage):
    """Mean MAE of a candidate's most advanced evaluated stage, or NaN."""
    for stage in reversed(STAGES):
        if _usable(by_stage.get(stage)):
            return mean_mae(by_stage[stage])
    return math.nan


def pick_winner(candidates):
    """candidates: {slug: {stage: eval_result}}. The lowest-scoring candidate, None if
    no candidate has a usable result."""
    scored = {slug: candidate_score(by_stage) for slug, by_stage in candidates.items()}
    scored = {slug: s for slug, s in scored.items() if not math.isnan(s)}
    return min(scored, key=scored.get) if scored else None


def assemble_report(candidate, model, git_sha, eval_by_stage, calibration_params, onnx_sizes, candidates=None):
    """Quality vs. the floor baseline, calibration coverage/width and the shuffled-pair
    shift per stage, deployability and JevBench drift. `candidates` ({slug: {stage:
    eval_result}}) adds the cross-candidate comparison and winner."""
    quality, calibration = {}, {}
    for stage in STAGES:
        result = eval_by_stage.get(stage)
        shuffle = (result or {}).get("shuffle_control") or {}
        quality[stage] = {
            "mean_mae": mean_mae(result),
            "beats_floor_count": beats_floor_count(result),
            "n_aspects": NUM_ASPECTS,
            "shuffle_mean_abs_shift": metrics.nanmean([s["mean_abs_shift"] for s in shuffle.values()]),
        }
        if _usable(result):
            calibration[stage] = {
                aid: {"coverage": m["coverage"], "mean_width": m["mean_width"]}
                for aid, m in result["metrics"].items() if "coverage" in m
            }
    candidates = candidates or {candidate: eval_by_stage}
    return {
        "git_sha": git_sha,
        "candidate": candidate,
        "model": model,
        "winner": pick_winner(candidates),
        "candidates": {slug: candidate_score(by_stage) for slug, by_stage in sorted(candidates.items())},
        "quality": quality,
        "calibration": {stage: c for stage, c in calibration.items() if c},
        "calibration_thresholds": calibration_params,
        "deployability": {
            "onnx_fp32_bytes": onnx_sizes.get("fp32"),
            "onnx_quantized_bytes": onnx_sizes.get("quantized"),
            # measured in a real browser, not by the pipeline
            "cold_load_time_s": None,
            "webgpu_latency_s": None,
            "wasm_latency_s": None,
            "loads_in_browser": None,
        },
        "drift_jevbench": {stage: (eval_by_stage.get(stage) or {}).get("jevbench")
                           for stage in ("headtrained", "finetuned")},
        "limitations": LIMITATIONS,
    }


def stage_rows(report):
    """One summary row per stage, shared by the Markdown and W&B renderings."""
    rows = []
    for stage in STAGES:
        q = report["quality"].get(stage, {})
        cal = (report.get("calibration") or {}).get(stage) or {}
        rows.append({
            "stage": stage,
            "label": STAGE_LABELS[stage],
            "mean_mae": q.get("mean_mae", math.nan),
            "beats_floor_count": q.get("beats_floor_count"),
            "n_aspects": q.get("n_aspects"),
            "shuffle_mean_abs_shift": q.get("shuffle_mean_abs_shift", math.nan),
            "mean_coverage": metrics.nanmean([m["coverage"] for m in cal.values()]),
            "mean_width": metrics.nanmean([m["mean_width"] for m in cal.values()]),
        })
    return rows


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def _with_unit(x, scale, unit, digits):
    return "n/a" if fmt(x) == "n/a" else f"{x / scale:.{digits}f} {unit}"


def render_markdown(report):
    rows = stage_rows(report)
    deploy = report.get("deployability", {})
    drift = report.get("drift_jevbench", {})
    lines = [
        f"# JobFit benchmark report -- {report['candidate']}",
        "",
        f"- Model: `{report['model']}`",
        f"- Git SHA: `{report['git_sha']}`",
        f"- Winner across candidates: `{report.get('winner')}`",
        "",
        "## Candidates (mean MAE, most advanced stage)",
        "",
        "| Candidate | Mean MAE |",
        "|---|---|",
        *[f"| {slug} | {fmt(score)} |" for slug, score in report.get("candidates", {}).items()],
        "",
        "## Quality (`test`)",
        "",
        "| Stage | Mean MAE | Beats train-mean floor | Shuffled-pair shift |",
        "|---|---|---|---|",
    ]
    for r in rows:
        beats = "n/a" if r["beats_floor_count"] is None else f"{r['beats_floor_count']}/{r['n_aspects']}"
        lines.append(f"| {r['label']} | {fmt(r['mean_mae'])} | {beats} | {fmt(r['shuffle_mean_abs_shift'])} |")
    lines += [
        "",
        "Shuffled-pair shift: mean |prediction change| when each test CV is paired with an "
        "unrelated JD; near 0 means the model ignores the JD. Per-aspect numbers and the "
        "keyword-overlap baseline are in the per-stage `eval/*.json`.",
        "",
        "## Calibration (target coverage 0.90)",
        "",
        "| Stage | Mean coverage | Mean interval width |",
        "|---|---|---|",
        *[f"| {r['label']} | {fmt(r['mean_coverage'])} | {fmt(r['mean_width'])} |"
          for r in rows if r["stage"] in ("calibrated", "quantized")],
        "",
        "## Deployability",
        "",
        f"- Quantized ONNX: {_with_unit(deploy.get('onnx_quantized_bytes'), 1e6, 'MB', 1)} "
        f"(fp32: {_with_unit(deploy.get('onnx_fp32_bytes'), 1e6, 'MB', 1)})",
        "",
        "Measured in a browser, not by the pipeline:",
        "",
        f"- Cold load: {_with_unit(deploy.get('cold_load_time_s'), 1, 's', 1)}",
        f"- Per-inference latency: WebGPU {_with_unit(deploy.get('webgpu_latency_s'), 1, 's', 2)} "
        f"(budget < 5 s), WASM {_with_unit(deploy.get('wasm_latency_s'), 1, 's', 2)} (budget < 30 s)",
        f"- Loads in a browser: {fmt(deploy.get('loads_in_browser'))}",
        "",
        "## Drift (JevBench, informational)",
        "",
        f"- Base weights: {fmt(drift.get('headtrained'))}",
        f"- Post-finetune: {fmt(drift.get('finetuned'))}",
        "",
        "## Limitations",
        "",
        *[f"- {note}" for note in report.get("limitations", [])],
        "",
    ]
    return "\n".join(lines)
