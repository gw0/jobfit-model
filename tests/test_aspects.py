"""Schema check for aspects.json (specs §3/§8). Stdlib-only, CI-safe."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_KEYS = {"id", "name", "scope", "description"}
VALID_SCOPES = {"pairwise", "cv", "job"}


def _load():
    with open(REPO_ROOT / "aspects.json", encoding="utf-8") as f:
        return json.load(f)


def test_seventeen_aspects():
    assert len(_load()) == 17


def test_required_keys_present_and_typed():
    for aspect in _load():
        assert REQUIRED_KEYS <= aspect.keys()
        for key in REQUIRED_KEYS:
            assert isinstance(aspect[key], str) and aspect[key]


def test_ids_unique():
    ids = [a["id"] for a in _load()]
    assert len(ids) == len(set(ids))


def test_scopes_valid():
    for aspect in _load():
        assert aspect["scope"] in VALID_SCOPES


def test_scope_counts_match_spec():
    aspects = _load()
    counts = {"pairwise": 0, "cv": 0, "job": 0}
    for aspect in aspects:
        counts[aspect["scope"]] += 1
    assert counts == {"pairwise": 13, "cv": 2, "job": 2}
