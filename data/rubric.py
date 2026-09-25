"""The LLM judge's rubric: questions.json, the same questions and criteria the model is
asked (specs/20260923-jev-model.md §3.2). The judge answers each score question with a
level on its criteria scale, stored as a label on [0,1]."""

import corpus


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
