"""The LLM judge's rubric: questions.json, the same questions and criteria the model is
asked (specs/20260923-jev-model.md §3.2). The judge answers each score question with a
level on its criteria scale, stored as a label on [0,1]."""

import corpus

ANSWER_FORMAT = ("a level from its scale (a fractional level such as 2.5 is fine when the answer "
                 "falls between two) and your confidence in that answer in [0,1]")


def load_questions(scope):
    """{id: question} for one scope ("pairwise", "cv" or "job")."""
    return {qid: q for qid, q in corpus.load_questions().items() if q["scope"] == scope}


def rubric_text(questions):
    """Each question with its numbered criteria levels."""
    blocks = []
    for qid, q in questions.items():
        levels = "\n".join(f"  {k} = {c}" for k, c in enumerate(q["criteria"]))
        blocks.append(f"- {qid}: {q['question']}\n{levels}")
    return "\n".join(blocks)


def level_to_score(level, question):
    """A judge's (possibly fractional) level -> the [0,1] label; None stays None."""
    if level is None:
        return None
    top = len(question["criteria"]) - 1
    return min(max(float(level), 0.0), float(top)) / top


def answer_fields(qid, question, entry):
    """`{qid}_score` and `{qid}_confidence` from one judge answer {"level", "confidence"};
    a missing answer or part is None."""
    entry = entry or {}
    confidence = entry.get("confidence")
    return {f"{qid}_score": level_to_score(entry.get("level"), question),
            f"{qid}_confidence": None if confidence is None else float(confidence)}
