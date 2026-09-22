#!/usr/bin/env python3
"""`publish` stage (specs §6/§10): compare every candidate under --runs-dir, write this
candidate's benchmark report, register it in MLflow (moving the `champion` alias to
the winner), and optionally push the winner to the HF Hub and the report to W&B.

Usage:
    ./pipeline/publish.py --runs-dir runs_smoke --model Qwen/Qwen3-0.6B
    HF_TOKEN=... ./pipeline/publish.py --runs-dir runs_smoke --push-hf --hf-org <org>
    WANDB_API_KEY=... ./pipeline/publish.py --runs-dir runs_smoke --push-wandb

Reads <runs-dir>/*/eval/<stage>.json and <runs-dir>/<slug>/export/web/, writes
<runs-dir>/<slug>/reports/{report.json,report.md} -- the one pipeline output that is
committed, so candidates trained on different hosts stay comparable.
"""

import argparse
import os
from pathlib import Path

import common
import report as report_lib

REGISTERED_MODEL = "jobfit"

# Base-model licences for the specs §4 candidates; None means "verify on the model card".
BASE_MODEL_LICENSES = {
    "qwen3-0.6b": "apache-2.0",
    "qwen3.5-4b": "apache-2.0",
    "llama-3.2-1b": "llama3.2",
    "minicpm5-2b": None,
}


def load_candidates(runs_dir):
    """{slug: {stage: eval_result}} for every <runs-dir>/<slug>/eval/."""
    candidates = {}
    for eval_dir in sorted(Path(runs_dir).glob("*/eval")):
        results = {stage: common.read_json(eval_dir / f"{stage}.json")
                   for stage in common.STAGES if (eval_dir / f"{stage}.json").is_file()}
        if results:
            candidates[eval_dir.parent.name] = results
    return candidates


def _model_of(by_stage):
    return next(r["model"] for r in by_stage.values() if "model" in r)


def build_report(runs_dir, slug, candidates, git_sha):
    export_dir = Path(runs_dir) / slug / "export"
    calib_path = export_dir / "web" / "calibration.json"
    sizes = {"fp32": common.onnx_bytes(export_dir) or None,
             "quantized": common.onnx_bytes(export_dir / "quantized") or None}
    return report_lib.assemble_report(
        slug, _model_of(candidates[slug]), git_sha, candidates[slug],
        common.read_json(calib_path) if calib_path.is_file() else None, sizes, candidates,
    )


def write_report(report, reports_dir):
    reports_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(reports_dir / "report.json", report)
    (reports_dir / "report.md").write_text(report_lib.render_markdown(report), encoding="utf-8")
    return reports_dir / "report.json", reports_dir / "report.md"


# --- MLflow registry -------------------------------------------------------------------

def register(run_id, slug, winner):
    """Registers this run's bundle as a new `jobfit` version and points `champion` at
    the winner's latest version (which may come from an earlier publish run)."""
    import mlflow

    client = mlflow.MlflowClient()
    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL, tags={"candidate": slug}).version
    print(f"registered {REGISTERED_MODEL} v{version} ({slug})")
    winner_versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL}' and tags.candidate='{winner}'",
        order_by=["version_number DESC"], max_results=1,
    )
    if winner_versions:
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", winner_versions[0].version)
        print(f"alias 'champion' -> v{winner_versions[0].version} ({winner})")
    else:
        print(f"winner {winner} has no registered version yet -- 'champion' left unchanged")


# --- HF Hub ----------------------------------------------------------------------------

def hf_repo_id(slug, hf_org, override=None):
    return override or f"{hf_org}/jobfit-{slug}"


def build_model_card(report, repo_id):
    licence = BASE_MODEL_LICENSES.get(report["candidate"])
    quantized = report["quality"]["quantized"]
    lines = [
        "---",
        f"base_model: {report['model']}",
        f"license: {licence or 'other'}",
        "tags:", "- jobfit", "- cv-job-fit-scoring", "- conformal-prediction",
        "---",
        "",
        f"# {repo_id}",
        "",
        "CV/job-description fit-scoring model (conformalized quantile regression head) from "
        "the [JobFit](https://github.com/gw0/jobfit) project; see its `runs/<slug>/reports/` "
        "for the full benchmark report.",
        "",
        f"- base model: `{report['model']}`" + ("" if licence else " -- licence per model card, verify before use"),
        f"- git SHA: `{report['git_sha']}`",
        f"- mean MAE (quantized, `test`): {report_lib.fmt(quantized['mean_mae'])}",
        f"- beats train-mean floor on {quantized['beats_floor_count']}/{quantized['n_aspects']} aspects",
        "",
        "Ships as dynamic-int8 ONNX for [transformers.js](https://github.com/huggingface/transformers.js), "
        "loaded through the generic `PreTrainedModel` class (this architecture isn't in its "
        "`AutoModelForSequenceClassification` mapping). `calibration.json` holds the per-aspect "
        "conformal deltas and insufficient-data thresholds.",
    ]
    return "\n".join(lines) + "\n"


def push_hf(report, web_dir, repo_id, private):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("--push-hf needs an HF Hub token with write access in HF_TOKEN")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(web_dir),
                      ignore_patterns=["parity.json"])
    api.upload_file(repo_id=repo_id, repo_type="model", path_in_repo="README.md",
                    path_or_fileobj=build_model_card(report, repo_id).encode())
    print(f"pushed model -> https://huggingface.co/{repo_id}")


# --- W&B -------------------------------------------------------------------------------

def push_wandb(report, report_paths, project):
    """Draft public benchmark export: per-stage table, summary metrics, report files."""
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("--push-wandb needs WANDB_API_KEY")
    import wandb

    run = wandb.init(project=project, job_type="benchmark", name=f"{report['candidate']}-{report['git_sha'][:7]}",
                     config={"model": report["model"], "candidate": report["candidate"], "git_sha": report["git_sha"]})
    rows = report_lib.stage_rows(report)
    columns = ["stage", "mean_mae", "beats_floor_count", "shuffle_mean_abs_shift", "mean_coverage", "mean_width"]
    run.log({"stages": wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in rows])})
    run.summary.update({
        "winner": report["winner"],
        **{f"{r['stage']}/{c}": r[c] for r in rows for c in columns[1:] if r[c] is not None},
    })
    artifact = wandb.Artifact(f"report-{report['candidate']}", type="report")
    for path in report_paths:
        artifact.add_file(str(path))
    run.log_artifact(artifact)
    run.finish()


def _run(args):
    import mlflow

    slug = common.model_slug(args.model)
    candidates = load_candidates(args.runs_dir)
    if slug not in candidates:
        raise SystemExit(f"no eval results under {args.runs_dir / slug / 'eval'} -- run evaluate.py first")
    report = build_report(args.runs_dir, slug, candidates, common.git_sha())
    winner = report["winner"]
    if winner is None:
        raise SystemExit("no candidate has a non-skipped eval result")
    report_paths = write_report(report, args.runs_dir / slug / "reports")
    print(f"winner: {winner} (of {len(candidates)}) -> {report_paths[0]}")

    web_dir = args.runs_dir / slug / "export" / "web"
    with common.mlflow_run("publish", args.model, args.run_group) as run:
        mlflow.log_params({"winner": winner, "candidates": len(candidates)})
        for path in report_paths:
            mlflow.log_artifact(str(path))
        if web_dir.is_dir():
            mlflow.log_artifacts(str(web_dir), artifact_path="model")
            register(run.info.run_id, slug, winner)

    if args.push_hf:
        winner_report = report if winner == slug else build_report(args.runs_dir, winner, candidates, report["git_sha"])
        push_hf(winner_report, args.runs_dir / winner / "export" / "web",
                hf_repo_id(winner, args.hf_org, args.hf_repo), args.hf_private)
    if args.push_wandb:
        push_wandb(report, report_paths, args.wandb_project)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument("--push-hf", action="store_true", help="push the winner's bundle; needs HF_TOKEN")
    parser.add_argument("--hf-org", default="jobfit")
    parser.add_argument("--hf-repo", default=None, help="full repo id, overrides <hf-org>/jobfit-<slug>")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--push-wandb", action="store_true", help="needs WANDB_API_KEY")
    parser.add_argument("--wandb-project", default="jobfit")
    args = parser.parse_args()
    with common.stage_span("publish"):
        _run(args)


if __name__ == "__main__":
    main()
