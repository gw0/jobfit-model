"""Schema check for questions.json (specs/20260923-jev-model.md §3.2). CI-safe: numpy only."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
import jev  # noqa: E402

VALID_SCOPES = {"pairwise", "cv", "job"}


def _load():
    with open(REPO_ROOT / "questions.json", encoding="utf-8") as f:
        return json.load(f)


def test_valid_jev_questions():
    jev.validate(_load())


def test_seventeen_score_questions_with_names_and_scopes():
    questions = _load()
    assert len(questions) == 17
    for q in questions.values():
        assert q["type"] == "score"
        assert isinstance(q["name"], str) and q["name"]
        assert q["scope"] in VALID_SCOPES


def test_scope_counts_match_spec():
    counts = {"pairwise": 0, "cv": 0, "job": 0}
    for q in _load().values():
        counts[q["scope"]] += 1
    assert counts == {"pairwise": 13, "cv": 2, "job": 2}
