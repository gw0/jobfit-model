"""Pairwise labels (specs §5): LLM-judge scores for one CV x job-post pair."""

import json

from claude_json import ask_json
from corpus import REPO_ROOT

# Frontmatter `company` values that carry no real company (HN "Who is hiring" posts all
# share one placeholder); aspect #17 is left unlabeled for these.
NO_COMPANY_PLACEHOLDERS = {"", "hn who is hiring"}

PAIR_PROMPT_TEMPLATE = """\
You are an expert technical recruiter acting as an LLM judge, scoring how well \
this CV fits this job description.

For each of the following aspects, give a continuous score in [0,1] and your \
confidence in that score in [0,1]:
{aspect_list}

CV:
---
{cv_text}
---

Job description:
---
{job_text}
---

{company_block}

Respond with ONLY a single raw JSON object, no markdown fences, no commentary, \
with exactly one key per aspect id listed above, each mapping to an object \
{{"score": <float 0-1>, "confidence": <float 0-1>}}.
"""


def load_pairwise_aspects():
    with open(REPO_ROOT / "aspects.json", encoding="utf-8") as f:
        return [a for a in json.load(f) if a["scope"] == "pairwise"]


def _aspect_list_text(aspects):
    return "\n".join(f"- {a['id']}: {a['name']} -- {a['description']}" for a in aspects)


def has_company_info(job_meta):
    return (job_meta.get("company") or "").strip().lower() not in NO_COMPANY_PLACEHOLDERS


def label_pair(cv_id, cv_body, job_id, job_body, job_meta, pairwise_aspects, judge_model=None):
    """Missing scores are written as JSON null."""
    company_present = has_company_info(job_meta)
    company_block = (
        f"Company: {job_meta.get('company')}" if company_present
        else "No company info supplied for this job."
    )
    prompt = PAIR_PROMPT_TEMPLATE.format(
        aspect_list=_aspect_list_text(pairwise_aspects),
        cv_text=cv_body, job_text=job_body, company_block=company_block,
    )
    data = ask_json(prompt, model=judge_model)

    record = {"cv": cv_id, "job": job_id}
    for aspect in pairwise_aspects:
        aid = aspect["id"]
        entry = data.get(aid) or {}
        score, confidence = entry.get("score"), entry.get("confidence")
        if aid == "culture_company_alignment_score" and not company_present:
            # Deterministic rather than relying on the judge to report low confidence.
            score = confidence = None
        record[f"{aid}_score"] = None if score is None else float(score)
        record[f"{aid}_confidence"] = None if confidence is None else float(confidence)
    record["labeling_method"] = f"llm-judge:{judge_model or 'default'}"
    return record
