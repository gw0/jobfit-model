"""Per-document labels (specs §5): the LLM judge's answer to the clarity question plus
the AI-text detector's score, for one CV or job post."""

import ai_text_detector
from claude_json import ask_json
from rubric import level_to_score, load_questions, rubric_text

DOC_LABELS = {"cv": "CV", "job": "job description"}
CLARITY_QUESTION = {"cv": "cv_clarity_structure_quality", "job": "job_post_clarity_structure_quality"}
LLM_GENERATED_QUESTION = {"cv": "cv_likely_llm_generated", "job": "job_post_likely_llm_generated"}

CLARITY_PROMPT_TEMPLATE = """\
You are an LLM judge answering a question about this {doc_label} text.

Answer the question below with a level from its scale (a fractional level such as \
2.5 is fine when the answer falls between two) and your confidence in that answer in \
[0,1]:
{rubric}

{doc_label} text:
---
{text}
---

Respond with ONLY a single raw JSON object, no markdown fences, no commentary, no tool use:
{{"level": <number>, "confidence": <float 0-1>}}
"""


def detector_confidence(score):
    """Distance from the 0.5 decision boundary, scaled to [0,1]."""
    return abs(score - 0.5) * 2.0


def label_doc(kind, doc_id, body, model=None, double_label=False):
    """`kind` is "cv" or "job". `model` picks the judge model for the clarity question
    (None defers to the `claude` CLI's own default). `double_label` marks the inter-rater
    QC pass: only clarity is re-judged there -- the detector is deterministic, re-running
    it adds nothing -- and the record is tagged with which model judged it."""
    qid = CLARITY_QUESTION[kind]
    question = load_questions(kind)[qid]
    prompt = CLARITY_PROMPT_TEMPLATE.format(doc_label=DOC_LABELS[kind], rubric=rubric_text({qid: question}), text=body)
    data = ask_json(prompt, model=model)
    record = {
        kind: doc_id,
        f"{qid}_score": level_to_score(data["level"], question),
        f"{qid}_confidence": float(data["confidence"]),
    }
    if double_label:
        record["labeling_method"] = f"llm-judge:{model}"
    else:
        llm_qid = LLM_GENERATED_QUESTION[kind]
        llm_score = ai_text_detector.score_llm_generated(body)
        record[f"{llm_qid}_score"] = llm_score
        record[f"{llm_qid}_confidence"] = detector_confidence(llm_score)
    return record
