"""Shared pipeline plumbing: aspect constants, JSON/cache I/O, label -> target joining,
tokenizer loading and CLI args. transformers is imported lazily, so the pure helpers are
testable without it.
"""

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

# Label field names for the cv/job-scope aspects, as written by data/label_docs.py.
CV_JOB_FIELD = {
    "cv_clarity_structure_quality": "clarity",
    "cv_likely_llm_generated": "llm_generated",
    "job_post_clarity_structure_quality": "clarity",
    "job_post_likely_llm_generated": "llm_generated",
}

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


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


# --- model -----------------------------------------------------------------------------

def load_tokenizer(model_name_or_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# --- cache (prepare.py's tokenized splits) ---------------------------------------------

def cache_path(runs_dir, slug, name):
    return Path(runs_dir) / slug / "cache" / f"{name}.pt"


def load_cache(runs_dir, slug, name):
    """{"pairs", "input_ids", "attention_mask"} for one cached split, or None if that
    split was empty -- at smoke scale some are."""
    import torch

    path = cache_path(runs_dir, slug, name)
    return torch.load(path, weights_only=False) if path.is_file() else None
