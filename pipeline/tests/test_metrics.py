"""Unit tests for metrics.py."""

import math

import numpy as np
import pytest

import metrics
from common import NUM_QUESTIONS, QUESTION_IDS

Q = NUM_QUESTIONS


def _full(n, value):
    return np.full((n, Q), value, dtype=float)


def test_masked_columns_applies_nan_and_floor():
    targets = np.array([[0.1] * Q, [np.nan] * Q, [0.3] * Q])
    confidences = np.array([[0.9] * Q, [0.9] * Q, [0.1] * Q])
    preds = np.arange(3 * Q, dtype=float).reshape(3, Q)
    columns = list(metrics.masked_columns(preds, targets, confidences, 0.3))
    assert [aid for aid, *_ in columns] == QUESTION_IDS
    aid, p, t, rows = columns[2]
    assert rows.tolist() == [True, False, False]
    assert p.tolist() == [2.0] and t.tolist() == [0.1]


def test_compute_metrics_perfect_and_masked():
    result = metrics.compute_metrics(_full(4, 0.5), _full(4, 0.5), _full(4, 1.0), 0.3)
    assert all(m["n"] == 4 and m["mae"] == 0.0 for m in result.values())
    masked = metrics.compute_metrics(_full(1, 0.1), _full(1, 0.9), _full(1, 0.05), 0.3)
    assert all(m["n"] == 0 and math.isnan(m["mae"]) for m in masked.values())


def test_spearman():
    assert math.isclose(metrics.spearman(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])), 1.0)
    assert math.isclose(metrics.spearman(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])), -1.0)
    assert math.isnan(metrics.spearman(np.array([1.0]), np.array([1.0])))
    assert math.isnan(metrics.spearman(np.array([1.0, 1.0]), np.array([0.0, 1.0])))


def test_mean_mae_skips_unlabeled_questions():
    targets = _full(2, 0.2)
    targets[:, 0] = np.nan
    assert math.isclose(metrics.mean_mae(_full(2, 0.7), targets, _full(2, 1.0), 0.3), 0.5)
    assert math.isnan(metrics.mean_mae(_full(2, 0.7), _full(2, np.nan), _full(2, 1.0), 0.3))


def test_confidence_threshold_nearest_rank():
    assert metrics.confidence_threshold(np.arange(1, 21, dtype=float)) == 2.0
    assert metrics.confidence_threshold(np.array([0.5, 0.1, 0.9])) == 0.1
    assert math.isnan(metrics.confidence_threshold(np.array([])))


def _logits(n, top):
    """(n, Q, 10) answer logits putting weight `top` on the highest level."""
    logits = np.zeros((n, Q, 10))
    logits[:, :, 4] = top
    return logits


def test_fit_calibration_pools_one_temperature_and_threshold():
    # Labels at the top level: a sharper softmax fits better, so T < 1.
    params = metrics.fit_calibration(_logits(6, 2.0), _full(6, 1.0), _full(6, 1.0), 0.3)
    assert set(params) == {"temperature", "confidence_threshold"}
    assert params["temperature"] < 1.0 and 0.0 <= params["confidence_threshold"] <= 1.0
    empty = metrics.fit_calibration(_logits(3, 2.0), _full(3, 1.0), _full(3, 0.0), 0.3)
    assert all(math.isnan(v) for v in empty.values())


def test_confidence_error():
    scores, targets = _full(4, 0.5), _full(4, 0.5)
    scores[:2] = 0.9  # the two least confident answers are the wrong ones
    answer_conf = np.tile(np.array([[0.1], [0.15], [0.85], [0.9]]), (1, Q))
    result = metrics.confidence_error(scores, answer_conf, targets, _full(4, 1.0), 0.3, threshold=0.15)
    r = result[QUESTION_IDS[0]]
    assert r["confidence_error_spearman"] < 0 and r["insufficient_rate"] == 0.25
    assert [b["n"] for b in r["confidence_error_bins"]] == [2, 0, 0, 0, 2]
    assert math.isclose(r["confidence_error_bins"][0]["mean_abs_error"], 0.4)
    uncalibrated = metrics.confidence_error(scores, answer_conf, targets, _full(4, 1.0), 0.3, threshold=math.nan)
    assert all(v["insufficient_rate"] == 1.0 for v in uncalibrated.values())


def test_bootstrap_ci():
    assert all(math.isnan(v) for v in metrics.bootstrap_ci([[0.1, 0.2]]))
    lo, hi = metrics.bootstrap_ci([[0.5 + 0.01 * i] for i in range(10)], n_resamples=200)
    assert lo <= hi


def test_bootstrap_mae_ci_respects_confidence_floor():
    # Each CV's one high-confidence pair is predicted perfectly; its below-floor pair is
    # far off. Only the unmasked errors (all 0) may enter the CI.
    preds = _full(6, 0.5)
    targets = _full(6, 0.5)
    targets[3:] = 0.0
    confidences = _full(6, 1.0)
    confidences[3:] = 0.1
    ci = metrics.bootstrap_mae_ci(preds, targets, confidences, 0.3, ["a", "b", "c", "a", "b", "c"],
                                  n_resamples=100)
    assert all(v["mae_ci_lo"] == 0.0 and v["mae_ci_hi"] == 0.0 for v in ci.values())


def test_train_mean_baseline():
    train = np.vstack([_full(1, 0.2), _full(1, 0.6), _full(1, 0.9)])
    conf = np.vstack([_full(2, 1.0), _full(1, 0.0)])  # the 0.9 row is below the floor
    preds = metrics.train_mean_preds(train, conf, 0.3, n=2)
    assert preds.shape == (2, Q) and np.allclose(preds, 0.4)
    assert all(math.isclose(m, 0.2) for m in metrics.per_question_mae(preds, _full(2, 0.6), _full(2, 1.0), 0.3).values())


def test_keyword_overlap_baseline():
    assert metrics.keyword_overlap("python django", "python django") == 1.0
    assert metrics.keyword_overlap("python django", "java spring") == 0.0
    train_targets = np.vstack([_full(1, 0.0), _full(1, 0.5), _full(1, 1.0)])
    preds = metrics.keyword_overlap_preds([0.0, 0.5, 1.0], train_targets, _full(3, 1.0), 0.3, [0.25])
    assert preds.shape == (1, Q) and np.allclose(preds, 0.25)
    flat = metrics.keyword_overlap_preds([0.3, 0.3], _full(2, 0.4), _full(2, 1.0), 0.3, [0.9])
    assert np.allclose(flat, 0.4)


@pytest.mark.parametrize("values,expected", [([1.0, np.nan, 3.0], 2.0), ([np.nan], math.nan), ([], math.nan)])
def test_nanmean(values, expected):
    result = metrics.nanmean(values)
    assert (math.isnan(result) and math.isnan(expected)) or result == expected
