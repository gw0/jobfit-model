"""Shared pipeline plumbing: JobFit's questions and state, JSON/cache I/O, label ->
target joining, model loading and batched inference, CLI args, MLflow runs and OTel
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

import jev

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data"))
import corpus  # noqa: E402

QUESTIONS = corpus.load_questions()
QUESTION_IDS = list(QUESTIONS)
NUM_QUESTIONS = len(QUESTION_IDS)

# JobFit's state is {CV, Job description}; frontend/src/jobfit.mjs mirrors these.
STATE_BUDGET = 3072
QUESTIONS_BUDGET = 1024

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
STAGES = ("zeroshot", "finetuned", "calibrated", "quantized")
MLFLOW_EXPERIMENT = "jobfit-pipeline"


def jobfit_state(cv_text, jd_text):
    return {"CV": cv_text, "Job description": jd_text}


def num_levels(qid):
    return len(QUESTIONS[qid]["criteria"])


def add_common_args(parser):
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets",
                        help="committed corpus (cvs/jobs/labels/splits), read-only")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs",
                        help="pipeline output, one <candidate>/ subdirectory per candidate")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF base model id")
    parser.add_argument("--candidate", default=None,
                        help="this run's name: the model slug for default settings, else the slug "
                             "plus what changed, e.g. smollm2-135m-instruct-r16 (default: the model slug)")
    parser.add_argument("--confidence-floor", type=float, default=0.3,
                        help="labels below this judge confidence are masked out")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-group", default=None,
                        help="nest this stage's MLflow run under the parent run of this group")


def model_slug(model_name):
    return model_name.split("/")[-1].lower()


def parse_args(parser, stage=None, argv=None):
    """Parses a stage's CLI and sets `args.candidate` (default: the model slug) and
    `args.run_dir` (<runs-dir>/<candidate>). A stage that builds the candidate passes
    its `stage` name to record its settings under that key in <run-dir>/config.json --
    the candidate's name is only a label, config.json is what it was run with."""
    args = parser.parse_args(argv)
    args.candidate = args.candidate or model_slug(args.model)
    args.run_dir = args.runs_dir / args.candidate
    if stage is not None:
        path = args.run_dir / "config.json"
        config = read_json(path) if path.is_file() else {}
        config[stage] = {k: v for k, v in vars(args).items() if not isinstance(v, Path) and k != "run_group"}
        write_json(path, config)
    return args


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
    """(targets, confidences), both float arrays of shape (len(pairs), NUM_QUESTIONS),
    targets on the labels' [0,1] scale, NaN wherever a label is missing."""
    cv_labels, job_labels, pair_labels = labels
    targets, confidences = [], []
    for pair in pairs:
        records = {
            "pairwise": pair_labels.get((pair["cv"], pair["job"]), {}),
            "cv": cv_labels.get(pair["cv"], {}),
            "job": job_labels.get(pair["job"], {}),
        }
        row_t, row_c = [], []
        for qid, question in QUESTIONS.items():
            record = records[question["scope"]]
            row_t.append(record.get(f"{qid}_score"))
            row_c.append(record.get(f"{qid}_confidence"))
        targets.append(row_t)
        confidences.append(row_c)
    shape = (len(pairs), NUM_QUESTIONS)
    return np.array(targets, dtype=float).reshape(shape), np.array(confidences, dtype=float).reshape(shape)


def usable(targets, confidences, confidence_floor):
    """(N, Q) bool: the label is present and its judge confidence is at/above the floor --
    the one mask shared by the loss weights and every metric."""
    targets, confidences = np.asarray(targets, dtype=float), np.asarray(confidences, dtype=float)
    return ~np.isnan(targets) & (np.nan_to_num(confidences) >= confidence_floor)


def label_weights(targets, confidences, confidence_floor):
    """(N, Q) loss weights: the judge confidence, zero where the label is not usable."""
    return np.where(usable(targets, confidences, confidence_floor), np.nan_to_num(confidences), 0.0)


def answer_targets(targets):
    """(N, Q, MAX_CANDIDATES) target distributions for [0,1] labels (zeros where missing)."""
    dists = np.zeros((*targets.shape, jev.MAX_CANDIDATES))
    for (i, j), y in np.ndenumerate(targets):
        if not math.isnan(y):
            qid = QUESTION_IDS[j]
            dists[i, j] = jev.target(QUESTIONS[qid], y * (num_levels(qid) - 1))
    return dists


def read_scores(answer_logits, temperature=1.0):
    """(scores, confidences), both (N, Q): each question's expected level on the labels'
    [0,1] scale (score / (K-1)) and its ordinal confidence."""
    answer_logits = np.asarray(answer_logits, dtype=float)
    scores, confidences = np.empty(answer_logits.shape[:2]), np.empty(answer_logits.shape[:2])
    for j, qid in enumerate(QUESTION_IDS):
        k = num_levels(qid)
        score, confidences[:, j] = jev.score_stats(jev.probabilities(answer_logits[:, j], k, temperature))
        scores[:, j] = score / (k - 1)
    return scores, confidences


def temperature(calibration):
    """A calibration's temperature, 1 (no scaling) before or without a fit."""
    t = (calibration or {}).get("temperature", math.nan)
    return 1.0 if math.isnan(t) else t


# --- model -----------------------------------------------------------------------------

def state_dict_hash(model):
    """sha256 over every parameter's raw bytes (the calibrate stage's no-weight-change assertion)."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(param.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def load_tokenizer(model_name_or_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def bf16_supported():
    """CUDA, or a CPU with native bf16 (AVX512-BF16/AMX); others emulate it far too slowly."""
    import torch

    return torch.cuda.is_available() or torch.ops.mkldnn._is_mkldnn_bf16_supported()


def load_jev_model(model_name_or_path, attn_implementation="sdpa", device=None):
    """The causal LM (native lm_head intact) wrapped as a JevModel, on `device` (default:
    CUDA when available, else CPU). fp32 weights, not bf16: train.py gets bf16 speed
    from autocast where `bf16_supported()`, so every checkpoint is the same fp32 model
    whichever host trained it."""
    import torch
    from transformers import AutoModelForCausalLM

    import jev_model

    lm = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, attn_implementation=attn_implementation, dtype=torch.float32, low_cpu_mem_usage=True,
    )
    lm.config.use_cache = False
    return jev_model.JevModel(lm).to(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def onnx_bytes(directory):
    """Size of the ONNX graphs in `directory`, including external-data weight files."""
    return sum(f.stat().st_size for pattern in ("*.onnx", "*.onnx_data") for f in Path(directory).glob(pattern))


MODEL_INPUTS = ("input_ids", "segment_ids", "answer_positions")


def predict_answer_logits(model, cache, batch_size=4):
    """(N, Q, MAX_CANDIDATES) answer logits as numpy, for a JevModel or an ONNX Runtime
    InferenceSession of its export alike."""
    import torch
    from tqdm import tqdm

    chunks = []
    for start in tqdm(range(0, len(cache["pairs"]), batch_size), desc="predicting", unit="batch", dynamic_ncols=True):
        batch = [cache[name][start:start + batch_size] for name in MODEL_INPUTS]
        if isinstance(model, torch.nn.Module):
            device = next((p.device for p in model.parameters()), torch.device("cpu"))
            with torch.no_grad():
                logits = model(*[t.to(device) for t in batch], cache["candidate_ids"].to(device))
            chunks.append(logits.float().cpu().numpy())
        else:
            feeds = {name: t.numpy() for name, t in zip(MODEL_INPUTS, batch)}
            chunks.append(model.run(["answer_logits"], {**feeds, "candidate_ids": cache["candidate_ids"].numpy()})[0])
    return np.concatenate(chunks)


# --- cache (prepare.py's tokenized splits) ---------------------------------------------

def cache_path(run_dir, name):
    return Path(run_dir) / "cache" / f"{name}.pt"


def load_cache(run_dir, name):
    """{"pairs", "input_ids", "segment_ids", "answer_positions", "candidate_ids",
    "candidate_counts"} for one cached split, or None if that split was empty -- at
    smoke scale some are."""
    import torch

    path = cache_path(run_dir, name)
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
def mlflow_run(stage, args, tags=None):
    """An MLflow run tagged with stage/model/candidate/git SHA. With `args.run_group`
    (Argo passes the workflow name) it is nested under that group's parent run. The
    tracking server is taken from MLFLOW_TRACKING_URI, else a local ./mlruns store."""
    import mlflow

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    tags = {"stage": stage, "model": args.model, "candidate": args.candidate, "git_sha": git_sha(), **(tags or {})}
    run_name, run_group = f"{stage}-{args.candidate}", args.run_group
    if not run_group:
        with mlflow.start_run(run_name=run_name, tags=tags, log_system_metrics=True) as run:
            yield run
        return
    with mlflow.start_run(run_id=_group_parent_run_id(mlflow, run_group)):
        with mlflow.start_run(run_name=run_name, nested=True, tags=tags, log_system_metrics=True) as run:
            yield run


def log_dataset_input(cache, run_dir, name, context):
    """mlflow.log_input() lineage for a cached split, pointing at the real .pt file."""
    import mlflow
    import mlflow.data

    dataset = mlflow.data.from_numpy(
        cache["input_ids"].numpy(), source=str(cache_path(run_dir, name)), name=f"{Path(run_dir).name}-{name}"
    )
    mlflow.log_input(dataset, context=context)


def log_model_signature(model, cache):
    """Logs an inferred model signature from one sample forward pass."""
    import mlflow

    one = {**{name: cache[name][:1] for name in MODEL_INPUTS}, "candidate_ids": cache["candidate_ids"],
           "pairs": cache["pairs"][:1]}
    inputs = {name: one[name].numpy() for name in (*MODEL_INPUTS, "candidate_ids")}
    signature = mlflow.models.infer_signature(inputs, predict_answer_logits(model, one))
    mlflow.log_dict(signature.to_dict(), "signature.json")


# --- OpenTelemetry ----------------------------------------------------------------------

_current_stage = None


def _cpu_utilization():
    import psutil

    return psutil.cpu_percent()


def _cpu_memory():
    import psutil

    return psutil.virtual_memory().percent


def _gpu_utilization():
    import torch

    return torch.cuda.utilization() if torch.cuda.is_available() else 0.0


def _gpu_memory():
    import torch

    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / total * 100


# Gauge name -> (description, reader). The GPU readers give 0 on CPU-only hosts (specs §7
# accepts that).
_GAUGES = {
    "cpu_utilization_percent": ("CPU utilization during a pipeline stage", _cpu_utilization),
    "cpu_memory_percent": ("System memory in use during a pipeline stage", _cpu_memory),
    "gpu_utilization_percent": ("GPU utilization during a pipeline stage, 0 on CPU-only hosts", _gpu_utilization),
    "gpu_memory_percent": ("GPU memory in use during a pipeline stage, 0 on CPU-only hosts", _gpu_memory),
}


def _observer(read):
    """A gauge callback, sampled by the periodic metric reader for as long as the process
    runs, attributing the reading to the current stage."""

    def observe(_options):
        from opentelemetry.metrics import Observation

        try:
            value = float(read())
        except Exception:  # noqa: BLE001 -- telemetry must never fail a stage
            value = 0.0
        yield Observation(value, {"stage": _current_stage or "unknown"})

    return observe


def _otel_tracer():
    """A tracer exporting to OTEL_EXPORTER_OTLP_ENDPOINT, with the utilization gauges
    registered once per process; None when no collector is configured or the SDK is
    missing."""
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
        _cpu_utilization()  # psutil measures since its previous call: the first one reads 0
        meter = metrics.get_meter("jobfit.pipeline")
        for name, (description, read) in _GAUGES.items():
            meter.create_observable_gauge(name, callbacks=[_observer(read)], description=description)
    return trace.get_tracer("jobfit.pipeline")


@contextmanager
def stage_span(name):
    """Wraps a stage in an OTel span and attributes utilization samples to it."""
    global _current_stage
    _current_stage = name
    tracer = _otel_tracer()
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(name):
        yield
