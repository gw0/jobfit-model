"""Shared pipeline plumbing: aspect constants, JSON/cache I/O, label -> target joining,
the pinball loss, model loading and batched inference, and CLI args. torch/transformers
are imported lazily, so the pure helpers are testable without them.
"""

import hashlib
import json
import math
import sys
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
STAGES = ("headtrained", "finetuned")


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
