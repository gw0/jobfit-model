"""Benchmark report (specs §10): picking the winning candidate, assembling the
self-contained report dict from per-stage eval results, and rendering it as Markdown.
Pure functions, no I/O."""

import math

import metrics
from common import NUM_QUESTIONS, STAGES

STAGE_LABELS = {
    "zeroshot": "zero-shot (base model, prompted)",
    "finetuned": "fine-tuned (LoRA, end-to-end)",
    "calibrated": "calibrated (temperature, pre-quantization)",
    "quantized": "quantized (shipped, int8)",
}

LIMITATIONS = [
    "The test split covers few distinct CV profiles, so candidate-axis generalisation is "
    "the weakest claim here; read the per-question table with its bootstrap CIs, not "
    "the mean as a point estimate.",
]


def _usable(result):
    return bool(result) and not result.get("skipped")


def final_stage(by_stage):
    """The most advanced stage with a usable eval result, or None."""
    return next((stage for stage in reversed(STAGES) if _usable(by_stage.get(stage))), None)


def _shuffle_shift(result):
    shuffle = result.get("shuffle_control") if _usable(result) else None
    return metrics.mean_of(shuffle, "mean_abs_shift") if shuffle else math.nan


def mean_mae(result):
    return metrics.mean_of(result["metrics"], "mae") if _usable(result) else math.nan


def beats_floor_count(result):
    """Questions whose MAE beats the train-mean baseline."""
    if not _usable(result):
        return None
    floor = result.get("baselines", {}).get("train_mean", {})
    return sum(
        1 for qid, m in result["metrics"].items()
        if m["mae"] < floor.get(qid, {}).get("mae", math.nan)  # NaN compares False
    )


def candidate_score(by_stage):
    """Mean MAE of a candidate's most advanced evaluated stage, or NaN."""
    stage = final_stage(by_stage)
    return math.nan if stage is None else mean_mae(by_stage[stage])


def pick_winner(candidates):
    """candidates: {candidate: {stage: eval_result}}. The lowest-scoring candidate, None
    if no candidate has a usable result."""
    scored = {name: candidate_score(by_stage) for name, by_stage in candidates.items()}
    scored = {name: s for name, s in scored.items() if not math.isnan(s)}
    return min(scored, key=scored.get) if scored else None


def _question_rows(result):
    """Per question of one eval result: MAE with its bootstrap CI, the train-mean MAE,
    and the rank correlation and calibration numbers."""
    floor = result.get("baselines", {}).get("train_mean", {})
    ci = result.get("bootstrap_mae_ci", {})
    return {
        qid: {
            "mae": m["mae"],
            "mae_ci": [ci.get(qid, {}).get("mae_ci_lo", math.nan), ci.get(qid, {}).get("mae_ci_hi", math.nan)],
            "floor_mae": floor.get(qid, {}).get("mae", math.nan),
            **{k: m.get(k, math.nan) for k in ("spearman_rho", "confidence_error_spearman", "insufficient_rate")},
        }
        for qid, m in result["metrics"].items()
    }


def assemble_report(candidate, model, git_sha, candidates, config, calibration_params, onnx_sizes):
    """Quality per stage against the baselines, the shipped stage question by question,
    the shipped calibration and deployability. `candidates` is {candidate: {stage:
    eval_result}}, this one included, for the cross-candidate comparison and winner;
    `config` is this candidate's config.json ({} if absent), the settings it was run with."""
    eval_by_stage = candidates[candidate]
    shipped = final_stage(eval_by_stage)
    quality, calibration = {}, {}
    for stage in STAGES:
        result = eval_by_stage.get(stage)
        quality[stage] = {
            "mean_mae": mean_mae(result),
            "beats_floor_count": beats_floor_count(result),
            "n_questions": NUM_QUESTIONS,
            "shuffle_mean_abs_shift": _shuffle_shift(result),
        }
        if _usable(result):
            calibration[stage] = {
                qid: {k: m[k] for k in ("confidence_error_spearman", "insufficient_rate")}
                for qid, m in result["metrics"].items() if "insufficient_rate" in m
            }
    baselines = {name: metrics.mean_of(per_question, "mae")
                 for name, per_question in eval_by_stage[shipped].get("baselines", {}).items()} if shipped else {}
    return {
        "git_sha": git_sha,
        "candidate": candidate,
        "model": model,
        "settings": {k: v for k, v in config.get("finetune", {}).items() if k not in ("model", "candidate")},
        "winner": pick_winner(candidates),
        "candidates": {name: {stage: mean_mae(by_stage.get(stage)) for stage in STAGES}
                       for name, by_stage in sorted(candidates.items())},
        "n_pairs": eval_by_stage[shipped]["n"] if shipped else None,
        "quality": quality,
        "baselines": baselines,
        "shipped_stage": shipped,
        "questions": _question_rows(eval_by_stage[shipped]) if shipped else {},
        "calibration": {stage: c for stage, c in calibration.items() if c},
        "calibration_params": calibration_params,
        "deployability": {
            "onnx_fp32_bytes": onnx_sizes.get("fp32"),
            "onnx_quantized_bytes": onnx_sizes.get("quantized"),
        },
        "limitations": LIMITATIONS,
    }


def stage_rows(report):
    """One summary row per stage, shared by the Markdown and W&B renderings."""
    rows = []
    for stage in STAGES:
        q = report["quality"][stage]
        cal = report["calibration"].get(stage, {})
        rows.append({
            "stage": stage,
            "label": STAGE_LABELS[stage],
            "mean_mae": q["mean_mae"],
            "beats_floor_count": q["beats_floor_count"],
            "n_questions": q["n_questions"],
            "shuffle_mean_abs_shift": q["shuffle_mean_abs_shift"],
            "mean_confidence_error_spearman": metrics.mean_of(cal, "confidence_error_spearman"),
            "mean_insufficient_rate": metrics.mean_of(cal, "insufficient_rate"),
        })
    return rows


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def _megabytes(x):
    return "n/a" if x is None else f"{x / 1e6:.1f} MB"


def render_markdown(report):
    rows = stage_rows(report)
    params = report["calibration_params"] or {}
    deploy = report["deployability"]
    stage_labels = [STAGE_LABELS[stage].split(" (")[0] for stage in STAGES]
    lines = [
        f"# JobFit benchmark report -- {report['candidate']}",
        "",
        "[![GitHub](https://img.shields.io/badge/GitHub-gw0%2Fjobfit--model-181717?logo=github)](https://github.com/gw0/jobfit-model)",
        "[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HF-dataset-orange)](https://huggingface.co/datasets/gw0/jobfit-jevbench)",
        "[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-model-yellow)](https://huggingface.co/gw0/jobfit-model)",
        "[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-space-blue)](https://huggingface.co/spaces/gw0/jobfit-app)",
        "[![Sponsor](https://img.shields.io/badge/sponsor-%E2%9D%A4-red?logo=github-sponsors)](https://github.com/sponsors/gw0)",
        "",
        f"- Model: `{report['model']}`",
        f"- Git SHA: `{report['git_sha']}`",
    ]
    if report["settings"]:
        lines += [f"- Settings: {', '.join(f'{k}={v}' for k, v in report['settings'].items())}"]
    lines += [
        "",
        "## Candidates (mean MAE on `test`)",
        "",
        f"| Candidate | {' | '.join(stage_labels)} |",
        f"|---|{'---|' * len(STAGES)}",
        *[f"| {'**' + name + '** (winner)' if name == report['winner'] else name} | "
          f"{' | '.join(fmt(by_stage[stage]) for stage in STAGES)} |"
          for name, by_stage in report["candidates"].items()],
        "",
        "The winner has the lowest mean MAE at its most advanced stage.",
        "",
        f"## Quality (`test`, {fmt(report['n_pairs'])} pairs)",
        "",
        "| | Mean MAE | Beats train-mean baseline | Shuffled-pair shift |",
        "|---|---|---|---|",
        f"| train-mean baseline | {fmt(report['baselines'].get('train_mean'))} | | |",
        f"| keyword-overlap baseline | {fmt(report['baselines'].get('keyword_overlap'))} | | |",
    ]
    for r in rows:
        beats = "n/a" if r["beats_floor_count"] is None else f"{r['beats_floor_count']}/{r['n_questions']}"
        lines.append(f"| {r['label']} | {fmt(r['mean_mae'])} | {beats} | {fmt(r['shuffle_mean_abs_shift'])} |")
    lines += [
        "",
        "Shuffled-pair shift: mean |prediction change| when each test CV is paired with an "
        "unrelated JD; near 0 means the model ignores the JD.",
        "",
    ]
    if report["questions"]:
        lines += [
            f"## Per question ({report['shipped_stage']})",
            "",
            "| Question | MAE [90% CI] | Train-mean MAE | Spearman | Confidence vs error | Insufficient-data rate |",
            "|---|---|---|---|---|---|",
            *[f"| {qid} | {fmt(q['mae'])} [{fmt(q['mae_ci'][0])}, {fmt(q['mae_ci'][1])}] | {fmt(q['floor_mae'])} | "
              f"{fmt(q['spearman_rho'])} | {fmt(q['confidence_error_spearman'])} | {fmt(q['insufficient_rate'])} |"
              for qid, q in report["questions"].items()],
            "",
        ]
    lines += [
        "## Calibration",
        "",
        f"- Shipped (fit on the quantized model): temperature {fmt(params.get('temperature'))}, "
        f"confidence threshold {fmt(params.get('confidence_threshold'))}",
        "",
        "| Stage | Spearman(confidence, abs. error) | Insufficient-data rate |",
        "|---|---|---|",
        *[f"| {r['label']} | {fmt(r['mean_confidence_error_spearman'])} | {fmt(r['mean_insufficient_rate'])} |"
          for r in rows if r["stage"] in ("calibrated", "quantized")],
        "",
        "Means over questions. The Spearman correlation should be clearly negative: the "
        "more confident an answer, the smaller its error.",
        "",
        "## Deployability",
        "",
        f"- Quantized ONNX: {_megabytes(deploy['onnx_quantized_bytes'])} (fp32: {_megabytes(deploy['onnx_fp32_bytes'])})",
        "",
        "## Limitations",
        "",
        *[f"- {note}" for note in report["limitations"]],
        "",
    ]
    return "\n".join(lines)
