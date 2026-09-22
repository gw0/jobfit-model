"""Shared pipeline plumbing: aspect constants, JSON/cache I/O, label -> target joining,
the pinball loss, model loading and batched inference, CLI args, MLflow runs and OTel
spans. torch/transformers/mlflow/opentelemetry are imported lazily, so the pure
helpers are testable without them.
"""

import hashlib
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data"))
import corpus  # noqa: E402

with open(REPO_ROOT / "aspects.json", encoding="utf-8") as f:
    ASPECTS = json.load(f)
ASPECT_IDS = [a["id"] for a in ASPECTS]
NUM_ASPECTS = len(ASPECT_IDS)

QUANTILES = (0.05, 0.5, 0.95)
LOW, MID, HIGH = 0, 1, 2  # positions in QUANTILES
NUM_LABELS = NUM_ASPECTS * len(QUANTILES)

# Label field names for the cv/job-scope aspects, as written by data/label_docs.py.
CV_JOB_FIELD = {
    "cv_clarity_structure_quality": "clarity",
    "cv_likely_llm_generated": "llm_generated",
    "job_post_clarity_structure_quality": "clarity",
    "job_post_likely_llm_generated": "llm_generated",
}

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
STAGES = ("headtrained", "finetuned", "calibrated")
MLFLOW_EXPERIMENT = "jobfit-pipeline"


def add_common_args(parser):
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets",
                        help="committed corpus (cvs/jobs/labels/splits), read-only")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs",
                        help="pipeline output, one <model-slug>/ subdirectory per candidate")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF base model id")
    parser.add_argument("--confidence-floor", type=float, default=0.3,
                        help="labels below this judge confidence are masked out")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-group", default=None,
                        help="nest this stage's MLflow run under the parent run of this group")


def model_slug(model_name):
    return model_name.split("/")[-1].lower()


# --- JSON with NaN <-> null ------------------------------------------------------------

def _nan_to_none(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    return obj


def _none_to_nan(obj):
    if obj is None:
        return float("nan")
    if isinstance(obj, dict):
        return {k: _none_to_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_none_to_nan(v) for v in obj]
    return obj


def write_json(path, obj):
    """NaN (the in-memory "missing") is written as null, which any JSON reader accepts."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_nan_to_none(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return _none_to_nan(json.loads(Path(path).read_text(encoding="utf-8")))


# --- labels -> targets -----------------------------------------------------------------

def load_labels(dataset_dir):
    """labels/{cvs,jobs,pairs}.jsonl keyed by path-id ((cv, job) for pairs)."""
    labels_dir = Path(dataset_dir) / "labels"
    cv_labels = {r["cv"]: r for r in corpus.load_jsonl(labels_dir / "cvs.jsonl")}
    job_labels = {r["job"]: r for r in corpus.load_jsonl(labels_dir / "jobs.jsonl")}
    pair_labels = {(r["cv"], r["job"]): r for r in corpus.load_jsonl(labels_dir / "pairs.jsonl")}
    return cv_labels, job_labels, pair_labels


def build_targets(pairs, labels):
    """(targets, confidences), both float arrays of shape (len(pairs), NUM_ASPECTS),
    NaN wherever a label is missing."""
    cv_labels, job_labels, pair_labels = labels
    targets, confidences = [], []
    for pair in pairs:
        records = {
            "pairwise": pair_labels.get((pair["cv"], pair["job"]), {}),
            "cv": cv_labels.get(pair["cv"], {}),
            "job": job_labels.get(pair["job"], {}),
        }
        row_t, row_c = [], []
        for aspect in ASPECTS:
            field = aspect["id"] if aspect["scope"] == "pairwise" else CV_JOB_FIELD[aspect["id"]]
            record = records[aspect["scope"]]
            row_t.append(record.get(f"{field}_score"))
            row_c.append(record.get(f"{field}_confidence"))
        targets.append(row_t)
        confidences.append(row_c)
    shape = (len(pairs), NUM_ASPECTS)
    return np.array(targets, dtype=float).reshape(shape), np.array(confidences, dtype=float).reshape(shape)


# --- loss ------------------------------------------------------------------------------

def pinball_loss(preds, targets, confidences, confidence_floor):
    """Confidence-weighted pinball loss. preds: (B, NUM_ASPECTS, len(QUANTILES));
    targets/confidences: (B, NUM_ASPECTS). NaN or below-floor targets are masked out;
    a batch with nothing left yields a zero loss."""
    import torch

    mask = ~targets.isnan() & (confidences >= confidence_floor)
    weights = torch.where(mask, confidences, torch.zeros_like(confidences))
    quantiles = torch.tensor(QUANTILES, dtype=preds.dtype, device=preds.device)
    diff = torch.nan_to_num(targets).unsqueeze(-1) - preds
    loss = torch.maximum(quantiles * diff, (quantiles - 1) * diff).sum(-1)
    return (loss * weights).sum() / (weights.sum() * len(QUANTILES)).clamp_min(1e-12)


# --- model -----------------------------------------------------------------------------

def freeze_backbone(model):
    """Freezes every parameter under the HF `base_model_prefix`, leaving the head trainable."""
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith(model.base_model_prefix)


def backbone_state_dict_hash(model):
    """sha256 over the backbone's raw parameter bytes (specs §4's freeze assertion)."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if name.startswith(model.base_model_prefix):
            h.update(name.encode())
            h.update(param.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def load_tokenizer(model_name_or_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_classification_model(model_name_or_path, pad_token_id):
    """The quantile-head classifier. sdpa attention avoids materialising the full
    2048x2048 attention matrix. fp32, not bf16: CPUs without native bf16 emulate it far
    too slowly to train on. `pad_token_id` must be the tokenizer's real pad id -- HF
    pools the last position where `input_ids != pad_token_id`, not by attention mask."""
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=NUM_LABELS, problem_type="regression",
        attn_implementation="sdpa", dtype=torch.float32, low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = pad_token_id
    model.config.use_cache = False
    return model


def reshape_quantile_logits(logits):
    """(B, NUM_LABELS) -> (B, NUM_ASPECTS, len(QUANTILES)), aspect-major."""
    return logits.reshape(logits.shape[0], NUM_ASPECTS, len(QUANTILES))


def predict_quantiles_batched(model, input_ids, attention_mask, batch_size=4):
    """Quantile predictions as a numpy (N, NUM_ASPECTS, 3) array, sorted per aspect so
    low <= mid <= high even where the raw head outputs cross. Works for a torch model
    and an optimum ORTModel alike."""
    import torch

    chunks = []
    for start in range(0, input_ids.shape[0], batch_size):
        with torch.no_grad():
            outputs = model(input_ids=input_ids[start:start + batch_size],
                            attention_mask=attention_mask[start:start + batch_size])
        chunks.append(reshape_quantile_logits(torch.as_tensor(outputs.logits)).float().numpy())
    return np.sort(np.concatenate(chunks), axis=-1)


# --- cache (prepare.py's tokenized splits) ---------------------------------------------

def cache_path(runs_dir, slug, name):
    return Path(runs_dir) / slug / "cache" / f"{name}.pt"


def load_cache(runs_dir, slug, name):
    """{"pairs", "input_ids", "attention_mask"} for one cached split, or None if that
    split was empty -- at smoke scale some are."""
    import torch

    path = cache_path(runs_dir, slug, name)
    return torch.load(path, weights_only=False) if path.is_file() else None


# --- provenance / tracking --------------------------------------------------------------

def git_sha():
    """GIT_SHA from the environment (baked into the pipeline image, which has no .git),
    else the checkout's HEAD."""
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _group_parent_run_id(mlflow, run_group):
    """The parent run tagged `run_group=<run_group>`, created on first use. Stages run
    sequentially, so find-or-create never races."""
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.run_group = '{run_group}'",
        max_results=1, output_format="list",
    )
    if runs:
        return runs[0].info.run_id
    run = mlflow.MlflowClient().create_run(
        experiment.experiment_id, tags={"run_group": run_group, "mlflow.runName": run_group}
    )
    return run.info.run_id


@contextmanager
def mlflow_run(stage, model, run_group=None, tags=None):
    """An MLflow run tagged with stage/model/git SHA. With `run_group` (Argo passes the
    workflow name) it is nested under that group's parent run. The tracking server is
    taken from MLFLOW_TRACKING_URI, else a local ./mlruns store."""
    import mlflow

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    tags = {"stage": stage, "model": model, "git_sha": git_sha(), **(tags or {})}
    run_name = f"{stage}-{model_slug(model)}"
    if not run_group:
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            yield run
        return
    with mlflow.start_run(run_id=_group_parent_run_id(mlflow, run_group)):
        with mlflow.start_run(run_name=run_name, nested=True, tags=tags) as run:
            yield run


def log_dataset_input(cache, runs_dir, slug, name, context):
    """mlflow.log_input() lineage for a cached split, pointing at the real .pt file."""
    import mlflow
    import mlflow.data

    dataset = mlflow.data.from_numpy(
        cache["input_ids"].numpy(), source=str(cache_path(runs_dir, slug, name)), name=f"{slug}-{name}"
    )
    mlflow.log_input(dataset, context=context)


def log_model_signature(model, input_ids, attention_mask):
    """Logs an inferred model signature from one sample forward pass."""
    import mlflow
    import torch

    sample = {"input_ids": input_ids[:1], "attention_mask": attention_mask[:1]}
    with torch.no_grad():
        logits = torch.as_tensor(model(**sample).logits).numpy()
    signature = mlflow.models.infer_signature({k: v.numpy() for k, v in sample.items()}, logits)
    mlflow.log_dict(signature.to_dict(), "signature.json")


# --- OpenTelemetry ----------------------------------------------------------------------

_current_stage = None


def _observe_gpu_utilization(_options):
    """Sampled by the periodic metric reader for as long as the process runs; 0 on
    CPU-only hosts (specs §7 accepts that)."""
    from opentelemetry.metrics import Observation

    value = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            value = float(torch.cuda.utilization())
    except Exception:  # noqa: BLE001 -- telemetry must never fail a stage
        pass
    yield Observation(value, {"stage": _current_stage or "unknown"})


def _otel_tracer():
    """A tracer exporting to OTEL_EXPORTER_OTLP_ENDPOINT, with the GPU gauge registered
    once per process; None when no collector is configured or the SDK is missing."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return None

    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        resource = Resource.create({"service.name": "jobfit-pipeline"})
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(tracer_provider)
        reader = PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=10_000)
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))
        metrics.get_meter("jobfit.pipeline").create_observable_gauge(
            "gpu_utilization_percent", callbacks=[_observe_gpu_utilization],
            description="GPU utilization during a pipeline stage, 0 on CPU-only hosts",
        )
    return trace.get_tracer("jobfit.pipeline")


@contextmanager
def stage_span(name):
    """Wraps a stage in an OTel span and attributes GPU-utilization samples to it."""
    global _current_stage
    _current_stage = name
    tracer = _otel_tracer()
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(name):
        yield
