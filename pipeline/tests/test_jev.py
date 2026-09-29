"""Unit tests for jev.py, with a whitespace stand-in tokenizer. frontend/tests/jev.test.ts
mirrors the encode and readout cases."""

import math

import numpy as np
import pytest

import jev

SCORE = {"type": "score", "question": "How good?", "criteria": ["bad", "ok", "good"]}
CHOICE = {"type": "choice", "question": "Which?", "criteria": {"x": "first", "y": "second"}}
NOUL = {"type": "noul", "question": "Is it?", "criteria": {"true": "yes", "false": "no"}}


class FakeTokenizer:
    """One token per whitespace-separated word; ids assigned on first sight."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]


def test_validate():
    jev.validate({"s": SCORE, "c": CHOICE, "n": NOUL})
    for bad in ({}, {"s": {**SCORE, "type": "rank"}}, {"s": {**SCORE, "criteria": ["only"]}},
                {"s": {**SCORE, "question": " "}}, {"c": {**CHOICE, "criteria": ["x", "y"]}},
                {"n": {**NOUL, "criteria": {"yes": "", "no": ""}}}):
        with pytest.raises(ValueError):
            jev.validate(bad)


def test_keys_and_candidates_per_type():
    assert (jev.answer_keys(SCORE), jev.candidates(SCORE)) == (["0", "1", "2"], ["0", "1", "2"])
    assert (jev.answer_keys(CHOICE), jev.candidates(CHOICE)) == (["x", "y"], ["A", "B"])
    assert (jev.answer_keys(NOUL), jev.candidates(NOUL)) == (["true", "false"], ["true", "false"])


def test_render_question_is_markdown_ending_in_the_answer_prompt():
    text = jev.render_question(CHOICE)
    assert text == "\n\n---\n\n# Question\nWhich?\n\n# Criteria\nA = first\nB = second\n\n# Answer\n"


def test_candidate_ids_single_token_check():
    tok = FakeTokenizer()
    ids, counts = jev.candidate_ids(tok, {"s": SCORE, "n": NOUL})
    assert counts == [3, 2] and len(ids[0]) == jev.MAX_CANDIDATES
    assert ids[1][:2] == [tok.vocab["true"], tok.vocab["false"]] and ids[1][2:] == [0] * 8
    with pytest.raises(ValueError, match="single token"):
        jev.candidate_ids(_TwoTokenDigits(), {"s": SCORE})


class _TwoTokenDigits(FakeTokenizer):
    """Splits a trailing digit into two tokens."""

    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text)
        return ids + [999] if text[-1:].isdigit() else ids


def test_encode_layout_segments_and_answer_positions():
    tok = FakeTokenizer()
    out = jev.encode(tok, {"CV": "a b", "Job": "c"}, {"s": SCORE, "n": NOUL}, state_budget=10,
                     questions_budget=40, pad_token_id=0)
    # "# CV a b" = 4 tokens, "--- # Job c" = 4 tokens, then the two branches.
    seg = out["segment_ids"]
    assert seg[:8] == [0] * 8 and out["truncated"] is False
    assert len(out["input_ids"]) == len(seg) == 10 + 40
    for k, pos in enumerate(out["answer_positions"], start=1):
        assert seg[pos] == k and (pos + 1 == len(seg) or seg[pos + 1] != k)
    assert seg[out["answer_positions"][-1] + 1:] == [-1] * (len(seg) - out["answer_positions"][-1] - 1)
    assert out["input_ids"][out["answer_positions"][0]] == tok.vocab["Answer"]


def test_encode_truncates_the_state_as_a_whole():
    out = jev.encode(FakeTokenizer(), {"CV": "a b", "Job": "w " * 10}, {"s": SCORE}, state_budget=6,
                     questions_budget=40)
    assert out["truncated"] is True
    assert out["segment_ids"].count(0) == 6  # "# CV a b" kept whole, the Job part cut inside its header


def test_encode_refuses_to_truncate_questions():
    with pytest.raises(ValueError, match="budget"):
        jev.encode(FakeTokenizer(), {"CV": "a"}, {"s": SCORE}, state_budget=5, questions_budget=5)


def test_score_target_is_two_hot_with_the_label_as_its_mean():
    dist = jev.target({**SCORE, "criteria": ["a"] * 5}, 1.25)
    assert np.allclose(dist[:3], [0, 0.75, 0.25]) and dist.sum() == 1.0
    assert np.allclose((dist * np.arange(10)).sum(), 1.25)
    assert jev.target(SCORE, 2.0)[2] == 1.0 and jev.target(SCORE, -1.0)[0] == 1.0
    assert jev.target(CHOICE, "y")[:2].tolist() == [0.0, 1.0]
    assert jev.target(NOUL, 0.8)[:2].tolist() == [0.8, pytest.approx(0.2)]


@pytest.mark.parametrize("probs,confidence", [
    ([0, 0, 1, 0, 0], 1.0), ([0, 0.5, 0.5, 0, 0], 0.75), ([0.2] * 5, 1 - math.sqrt(2) / 2), ([0.5, 0, 0, 0, 0.5], 0.0),
])
def test_score_confidence_worked_examples(probs, confidence):
    _, c = jev.score_stats(np.array(probs))
    assert math.isclose(c, confidence, abs_tol=1e-12)


def test_answers_per_type():
    logits = [np.log([0.1, 0.2, 0.7] + [1e-9] * 7), np.log([0.25, 0.75] + [1e-9] * 8), np.log([0.9, 0.1] + [1e-9] * 8)]
    out = jev.answers({"s": SCORE, "c": CHOICE, "n": NOUL}, logits)
    assert out["s"]["type"] == "score" and math.isclose(out["s"]["score"], 1.6)
    assert set(out["s"]["probabilities"]) == {"0", "1", "2"}
    assert out["c"]["choice"] == "y" and math.isclose(out["c"]["confidence"], 0.5)
    assert out["n"] == {"type": "noul", "noul": pytest.approx(0.9)}


def test_fit_temperature():
    # Overconfident logits against 50/50 targets: the fit softens them (T > 1).
    logits = [np.tile([4.0, 0.0], (20, 1))]
    t = jev.fit_temperature(logits, [np.full((20, 2), 0.5)], [np.ones(20)])
    assert t > 1.0
    # Correct, soft logits against one-hot targets: sharpen (T < 1).
    assert jev.fit_temperature([np.tile([1.0, 0.0], (5, 1))], [np.tile([1.0, 0.0], (5, 1))], [np.ones(5)]) < 1.0
    assert math.isnan(jev.fit_temperature(logits, [np.full((20, 2), 0.5)], [np.zeros(20)]))
