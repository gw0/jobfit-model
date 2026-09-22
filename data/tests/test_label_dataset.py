"""Unit tests for the labeling helpers. No claude/transformers calls."""

import json

import pytest

import label_pairs
from label_dataset import _label_all
from label_docs import detector_confidence
from label_pairs import _aspect_list_text, has_company_info, load_pairwise_aspects


def test_has_company_info():
    assert has_company_info({"company": "Grafana Labs"}) is True
    assert has_company_info({}) is False
    assert has_company_info({"company": "HN Who is Hiring"}) is False
    assert has_company_info({"company": "hn who is hiring"}) is False


def test_detector_confidence():
    assert detector_confidence(0.5) == 0.0
    assert detector_confidence(0.0) == 1.0
    assert detector_confidence(1.0) == 1.0


def test_pairwise_aspects_listed_in_prompt():
    aspects = load_pairwise_aspects()
    assert len(aspects) == 13
    text = _aspect_list_text(aspects)
    assert all(a["id"] in text for a in aspects)


def test_label_pair_writes_null_not_nan(monkeypatch):
    monkeypatch.setattr(label_pairs, "ask_json", lambda prompt, model=None: {})
    record = label_pairs.label_pair("cvs/a.md", "cv", "jobs/x/1.md", "jd", {"company": "X"},
                                    load_pairwise_aspects())
    assert record["skills_match_score"] is None
    json.dumps(record, allow_nan=False)


def test_label_all_skips_individual_failures():
    def fn(item):
        if item == 2:
            raise ValueError("bad reply")
        return item
    assert _label_all("thing", [1, 2, 3], fn) == [1, 3]


def test_label_all_raises_when_every_item_fails():
    def fn(item):
        raise RuntimeError("not logged in")
    with pytest.raises(SystemExit, match="every thing labeling call failed"):
        _label_all("thing", [1, 2], fn)


def test_label_all_empty_input_is_fine():
    assert _label_all("thing", [], lambda item: item) == []
