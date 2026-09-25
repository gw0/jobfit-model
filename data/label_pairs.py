"""Pairwise labels (specs §5): LLM-judge answers to the pairwise questions for one CV x
job-post pair. The judge sees exactly what the model sees: the CV, the job description
and the questions."""

from claude_json import ask_json
from rubric import level_to_score, load_questions, rubric_text

BATCH_PAIR_PROMPT_TEMPLATE = """\
You are an expert technical recruiter acting as an LLM judge, answering questions \
about how well this CV fits each of the following job descriptions. Judge every job \
on its own merits -- do not compare or rank jobs against each other.

Answer each question below with a level from its scale (a fractional level such as \
2.5 is fine when the answer falls between two) and your confidence in that answer in \
[0,1]:
{rubric}

CV:
---
{cv_text}
---

{jobs_block}

Respond with ONLY a single raw JSON object, no markdown fences, no commentary, no \
tool use, with exactly one key per job label above ({job_labels}), each mapping to an \
object with one key per question id listed above, each mapping to an object \
{{"level": <number>, "confidence": <float 0-1>}}.
"""


def load_pairwise_questions():
    return load_questions("pairwise")


def label_pairs_for_cv(cv_id, cv_body, jobs, questions, judge_model=None):
    """`jobs`: [(job_id, job_body), ...], 1..N of them, all for one cv_id, judged in a
    single call. Returns one record per job. A job or question missing from the reply
    is written as JSON null, not a crash."""
    job_labels = [f"job_{i}" for i in range(1, len(jobs) + 1)]
    jobs_block = "\n".join(f"{label}:\n---\n{body}\n---" for label, (_, body) in zip(job_labels, jobs))
    prompt = BATCH_PAIR_PROMPT_TEMPLATE.format(
        rubric=rubric_text(questions), cv_text=cv_body, jobs_block=jobs_block,
        job_labels=", ".join(job_labels))
    data = ask_json(prompt, model=judge_model)

    records = []
    for label, (job_id, _) in zip(job_labels, jobs):
        answers = data.get(label) or {}
        record = {"cv": cv_id, "job": job_id}
        for qid, question in questions.items():
            entry = answers.get(qid) or {}
            confidence = entry.get("confidence")
            record[f"{qid}_score"] = level_to_score(entry.get("level"), question)
            record[f"{qid}_confidence"] = None if confidence is None else float(confidence)
        record["labeling_method"] = f"llm-judge:{judge_model or 'default'}"
        records.append(record)
    return records
