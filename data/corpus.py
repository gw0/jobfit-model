"""Reading and writing the dataset corpus (`<datasets-dir>/{cvs,jobs,labels,splits}`).

Stdlib only: imported by the data/ scripts, data/fetch_jobs/ and pipeline/.
Documents are addressed by path-id relative to the dataset dir, e.g.
`cvs/jane-doe.md` or `jobs/<company>/<file>.md`.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "val", "calib", "test")
DEFAULT_SEED = 42


def load_questions():
    """questions.json: {question_id: {"type", "name", "scope", "question", "criteria"}},
    the one rubric shared by the LLM judge and the model."""
    with open(REPO_ROOT / "questions.json", encoding="utf-8") as f:
        return json.load(f)


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


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


def load_splits(datasets_dir):
    """{split_name: [{"cv": ..., "job": ...}, ...]}, empty list for a missing split."""
    return {name: load_jsonl(Path(datasets_dir) / "splits" / f"{name}.jsonl") for name in SPLIT_NAMES}


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


def list_cvs(datasets_dir):
    cvs_dir = Path(datasets_dir) / "cvs"
    return sorted(f"cvs/{p.name}" for p in cvs_dir.glob("*.md")) if cvs_dir.is_dir() else []


def list_jobs(datasets_dir):
    """{company: [job path-ids]}, companies with no posts omitted."""
    jobs_root = Path(datasets_dir) / "jobs"
    jobs_by_company = {}
    if jobs_root.is_dir():
        for company_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
            job_ids = sorted(f"jobs/{company_dir.name}/{p.name}" for p in company_dir.glob("*.md"))
            if job_ids:
                jobs_by_company[company_dir.name] = job_ids
    return jobs_by_company


def load_texts(datasets_dir):
    """{path-id: text} for every CV and job post (job posts without their frontmatter)."""
    root = Path(datasets_dir)
    texts = {cv_id: (root / cv_id).read_text(encoding="utf-8") for cv_id in list_cvs(root)}
    texts.update({job_id: read_job_body(root / job_id)
                  for job_ids in list_jobs(root).values() for job_id in job_ids})
    return texts
