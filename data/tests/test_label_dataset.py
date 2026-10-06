"""Unit tests for the labeling helpers. No claude/transformers calls."""

import json
import subprocess

import pytest

import claude_json
import label_pairs
from claude_json import UsageLimitError, extract_json_object, run_parallel
from corpus import load_jsonl
from label_dataset import _chunk_by_cv, _label_file
from label_docs import detector_confidence
from rubric import level_to_score, load_questions, rubric_text


def _claude_prints(monkeypatch, returncode, event):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, returncode, stdout="banner\n" + json.dumps(event), stderr="")
    monkeypatch.setattr(claude_json.subprocess, "run", run)
    return calls


def test_usage_limit_is_raised_at_once_not_retried(monkeypatch):
    calls = _claude_prints(monkeypatch, 1, {"type": "result", "is_error": True, "api_error_status": 429,
                                            "result": "You've hit your session limit"})
    with pytest.raises(UsageLimitError, match="session limit"):
        claude_json.ask_json("prompt")
    assert len(calls) == 1


def test_other_claude_errors_are_retried_then_raised(monkeypatch):
    calls = _claude_prints(monkeypatch, 1, {"type": "result", "is_error": True, "result": "overloaded"})
    with pytest.raises(RuntimeError, match="claude exited 1"):
        claude_json.ask_json("prompt")
    assert len(calls) == 2
    _claude_prints(monkeypatch, 0, {"type": "result", "result": '{"a": 1}'})
    assert claude_json.ask_json("prompt") == {"a": 1}


def test_extract_json_object_unwraps_fences_and_prose():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('Here:\n```json\n{"a": "x\ny"}\n```') == {"a": "x\ny"}
    with pytest.raises(ValueError, match="could not find"):
        extract_json_object("no object here")


def test_run_parallel_stops_on_usage_limit_keeping_finished_results():
    kept = []

    def fn(i):
        if i >= 2:
            raise UsageLimitError("session limit")
        return i
    with pytest.raises(SystemExit, match="session limit"):
        run_parallel("thing", range(5), fn, 1, lambda item, result: kept.append(result))
    assert kept == [0, 1]


def test_run_parallel_skips_individual_failures():
    def fn(i):
        if i == 2:
            raise ValueError("bad reply")
        return i
    kept = []
    assert run_parallel("thing", [1, 2, 3], fn, 1, lambda item, result: kept.append(result)) == 2
    assert kept == [1, 3]


def test_detector_confidence():
    assert detector_confidence(0.5) == 0.0
    assert detector_confidence(0.0) == 1.0
    assert detector_confidence(1.0) == 1.0


def test_pairwise_questions_and_criteria_in_rubric():
    questions = load_questions("pairwise")
    text = rubric_text(questions)
    assert all(qid in text and q["question"] in text and f"4 = {q['criteria'][4]}" in text
               for qid, q in questions.items())


def test_level_to_score():
    question = {"criteria": ["a", "b", "c", "d", "e"]}
    assert level_to_score(2, question) == 0.5
    assert level_to_score(2.5, question) == 0.625
    assert level_to_score(9, question) == 1.0 and level_to_score(-1, question) == 0.0
    assert level_to_score(None, question) is None


def test_label_pairs_for_cv_maps_levels_and_writes_null_not_nan(monkeypatch):
    monkeypatch.setattr(label_pairs, "ask_json",
                        lambda prompt, model=None: {"job_1": {"overall_fit": {"level": 3, "confidence": 0.8}}})
    records = label_pairs.label_pairs_for_cv("cvs/a.md", "cv", [("jobs/x/1.md", "jd")], load_questions("pairwise"))
    record = records[0]
    assert record["job"] == "jobs/x/1.md"
    assert record["overall_fit_score"] == 0.75 and record["overall_fit_confidence"] == 0.8
    assert record["skills_match_score"] is None
    json.dumps(record, allow_nan=False)


def test_label_pairs_for_cv_batched_reply_maps_each_job_independently(monkeypatch):
    reply = {
        "job_1": {"overall_fit": {"level": 4, "confidence": 0.9}},
        # job_2 is entirely missing from the reply
        "job_3": {"overall_fit": {"level": 0, "confidence": 0.1}},
    }
    monkeypatch.setattr(label_pairs, "ask_json", lambda prompt, model=None: reply)
    jobs = [("jobs/a.md", "a"), ("jobs/b.md", "b"), ("jobs/c.md", "c")]
    records = label_pairs.label_pairs_for_cv("cvs/x.md", "cv", jobs, load_questions("pairwise"))

    assert [r["job"] for r in records] == [job_id for job_id, _ in jobs]
    assert records[0]["overall_fit_score"] == 1.0
    assert records[1]["overall_fit_score"] is None and records[1]["skills_match_score"] is None
    assert records[2]["overall_fit_score"] == 0.0


def test_chunk_by_cv_never_mixes_cvs_and_respects_max_size():
    pairs = [{"cv": "a", "job": "j1"}, {"cv": "a", "job": "j2"}, {"cv": "a", "job": "j3"}, {"cv": "b", "job": "j4"}]
    chunks = _chunk_by_cv(pairs, max_size=2)
    assert [[p["job"] for p in chunk] for chunk in chunks] == [["j1", "j2"], ["j3"], ["j4"]]
    assert all(len({p["cv"] for p in chunk}) == 1 for chunk in chunks)  # no chunk spans two CVs


def test_label_file_raises_when_every_item_fails(tmp_path):
    def fn(chunk):
        raise RuntimeError("not logged in")
    with pytest.raises(SystemExit, match="every CV labeling call failed"):
        _label_file(tmp_path / "cvs.jsonl", "CV", [{"cv": "a"}, {"cv": "b"}], fn, force=False, workers=1)


def test_label_file_appends_as_it_goes_and_resumes(tmp_path):
    path = tmp_path / "labels" / "cvs.jsonl"

    def crash_on_c(chunk):
        if chunk[0]["cv"] == "c":
            raise KeyboardInterrupt  # an interrupted run: not a per-item failure
        return [{"cv": chunk[0]["cv"], "n": 1}]

    with pytest.raises(KeyboardInterrupt):
        _label_file(path, "CV", [{"cv": "a"}, {"cv": "b"}, {"cv": "c"}], crash_on_c, force=False, workers=1)
    assert [r["cv"] for r in load_jsonl(path)] == ["a", "b"]  # finished records survived

    calls = []
    items = [{"cv": "c"}, {"cv": "b"}, {"cv": "a"}]
    _label_file(path, "CV", items, lambda chunk: calls.append(chunk[0]["cv"]) or [{"cv": chunk[0]["cv"], "n": 2}],
                force=False, workers=2)
    assert calls == ["c"]
    assert [(r["cv"], r["n"]) for r in load_jsonl(path)] == [("c", 2), ("b", 1), ("a", 1)]  # items order

    _label_file(path, "CV", items, lambda chunk: [{"cv": chunk[0]["cv"], "n": 3}], force=True, workers=1)
    assert [(r["cv"], r["n"]) for r in load_jsonl(path)] == [("c", 3), ("b", 3), ("a", 3)]  # --force starts over
