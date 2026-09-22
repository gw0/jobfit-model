"""Reading and writing the dataset corpus (`<dataset-dir>/{cvs,jobs,labels,splits}`).

Stdlib only: imported by the data/ scripts, data/fetch_jobs/ and pipeline/.
Documents are addressed by path-id relative to the dataset dir, e.g.
`cvs/jane-doe.md` or `jobs/<company>/<file>.md`.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets"
SPLIT_NAMES = ("train", "val", "calib", "test")


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=False) + "\n")


def load_splits(dataset_dir):
    """{split_name: [{"cv": ..., "job": ...}, ...]}, empty list for a missing split."""
    return {name: load_jsonl(Path(dataset_dir) / "splits" / f"{name}.jsonl") for name in SPLIT_NAMES}


def slugify(text):
    """Lowercase a-z0-9 slug used for corpus file and directory names; "" for empty input."""
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def format_frontmatter(meta):
    """The `---`-fenced block parse_frontmatter reads; values are flattened to one line."""
    lines = [f"{key}: {str(value or '').replace(chr(10), ' ').strip()}" for key, value in meta.items()]
    return "\n".join(["---", *lines, "---"])


def parse_frontmatter(text):
    """Splits a `---`-fenced `key: value` block from the body. Only job posts carry
    one; text without it comes back as ({}, text)."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            meta = {}
            for line in text[4:end].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip()
            return meta, text[end + 4:].lstrip("\n")
    return {}, text


def read_job_body(path):
    """The job-post text a user would paste: the file without its frontmatter. Not for
    CVs -- a CV may legitimately open with a `---` rule."""
    return parse_frontmatter(Path(path).read_text(encoding="utf-8"))[1]


def company_of(job_id):
    return job_id.split("/")[1]


def list_cvs(dataset_dir):
    cvs_dir = Path(dataset_dir) / "cvs"
    return sorted(f"cvs/{p.name}" for p in cvs_dir.glob("*.md")) if cvs_dir.is_dir() else []


def list_jobs(dataset_dir):
    """{company: [job path-ids]}, companies with no posts omitted."""
    jobs_root = Path(dataset_dir) / "jobs"
    jobs_by_company = {}
    if jobs_root.is_dir():
        for company_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
            job_ids = sorted(f"jobs/{company_dir.name}/{p.name}" for p in company_dir.glob("*.md"))
            if job_ids:
                jobs_by_company[company_dir.name] = job_ids
    return jobs_by_company
