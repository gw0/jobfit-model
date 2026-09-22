"""Per-document labels (specs §5): LLM-judge clarity plus the AI-text detector's
score, for one CV or job post."""

import ai_text_detector
from claude_json import ask_json

DOC_LABELS = {"cv": "CV", "job": "job post"}

CLARITY_PROMPT_TEMPLATE = """\
You are scoring how clear and well-structured this {doc_label} text is, on a \
continuous 0-1 scale (0 = very poorly structured/hard to parse, 1 = excellent \
structure/very clear). Also give your confidence in that score, 0-1.

{doc_label} text:
---
{text}
---

Respond with ONLY a single raw JSON object, no markdown fences, no commentary:
{{"clarity_score": <float 0-1>, "clarity_confidence": <float 0-1>}}
"""


def detector_confidence(score):
    """Distance from the 0.5 decision boundary, scaled to [0,1]."""
    return abs(score - 0.5) * 2.0


def label_doc(kind, doc_id, body, judge_model=None):
    """`kind` is "cv" or "job". With `judge_model` (the double-label QC pass) only
    clarity is re-judged: the detector is deterministic, re-running it adds nothing."""
    prompt = CLARITY_PROMPT_TEMPLATE.format(doc_label=DOC_LABELS[kind], text=body)
    data = ask_json(prompt, model=judge_model)
    record = {
        kind: doc_id,
        "clarity_score": float(data["clarity_score"]),
        "clarity_confidence": float(data["clarity_confidence"]),
    }
    if judge_model:
        record["labeling_method"] = f"llm-judge:{judge_model}"
    else:
        llm_score = ai_text_detector.score_llm_generated(body)
        record["llm_generated_score"] = llm_score
        record["llm_generated_confidence"] = detector_confidence(llm_score)
    return record
