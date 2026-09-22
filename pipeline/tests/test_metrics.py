"""Unit tests for metrics.py."""

import math

import numpy as np
import pytest

import metrics
from common import ASPECT_IDS, NUM_ASPECTS

A = NUM_ASPECTS


def _full(n, value):
    return np.full((n, A), value, dtype=float)


def _quantiles(n, low, mid, high):
    return np.tile(np.array([low, mid, high], dtype=float), (n, A, 1))


def test_masked_columns_applies_nan_and_floor():
    targets = np.array([[0.1] * A, [np.nan] * A, [0.3] * A])
    confidences = np.array([[0.9] * A, [0.9] * A, [0.1] * A])
    preds = np.arange(3 * A, dtype=float).reshape(3, A)
    columns = list(metrics.masked_columns(preds, targets, confidences, 0.3))
    assert [aid for aid, *_ in columns] == ASPECT_IDS
    aid, p, t, rows = columns[2]
    assert rows.tolist() == [True, False, False]
    assert p.tolist() == [2.0] and t.tolist() == [0.1]


def test_compute_metrics_perfect_and_masked():
    result = metrics.compute_metrics(_quantiles(4, 0.4, 0.5, 0.6), _full(4, 0.5), _full(4, 1.0), 0.3)
    assert all(m["n"] == 4 and m["mae"] == 0.0 for m in result.values())
    masked = metrics.compute_metrics(_quantiles(1, 0, 0.1, 0.2), _full(1, 0.9), _full(1, 0.05), 0.3)
    assert all(m["n"] == 0 and math.isnan(m["mae"]) for m in masked.values())


def test_spearman():
    assert math.isclose(metrics.spearman(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])), 1.0)
    assert math.isclose(metrics.spearman(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])), -1.0)
    assert math.isnan(metrics.spearman(np.array([1.0]), np.array([1.0])))
    assert math.isnan(metrics.spearman(np.array([1.0, 1.0]), np.array([0.0, 1.0])))


def test_mean_mae_skips_unlabeled_aspects():
    targets = _full(2, 0.2)
    targets[:, 0] = np.nan
    assert math.isclose(metrics.mean_mae(_quantiles(2, 0, 0.7, 1), targets, _full(2, 1.0), 0.3), 0.5)
    assert math.isnan(metrics.mean_mae(_quantiles(2, 0, 0.7, 1), _full(2, np.nan), _full(2, 1.0), 0.3))


def test_bootstrap_ci():
    assert all(math.isnan(v) for v in metrics.bootstrap_ci([[0.1, 0.2]]))
    lo, hi = metrics.bootstrap_ci([[0.5 + 0.01 * i] for i in range(10)], n_resamples=200)
    assert lo <= hi


def test_bootstrap_mae_ci_respects_confidence_floor():
    # Each CV's one high-confidence pair is predicted perfectly; its below-floor pair is
    # far off. Only the unmasked errors (all 0) may enter the CI.
    preds = _quantiles(6, 0, 0.5, 1)
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
    assert preds.shape == (2, A) and np.allclose(preds, 0.4)
    assert all(math.isclose(m, 0.2) for m in metrics.per_aspect_mae(preds, _full(2, 0.6), _full(2, 1.0), 0.3).values())


def test_keyword_overlap_baseline():
    assert metrics.keyword_overlap("python django", "python django") == 1.0
    assert metrics.keyword_overlap("python django", "java spring") == 0.0
    train_targets = np.vstack([_full(1, 0.0), _full(1, 0.5), _full(1, 1.0)])
    preds = metrics.keyword_overlap_preds([0.0, 0.5, 1.0], train_targets, _full(3, 1.0), 0.3, [0.25])
    assert preds.shape == (1, A) and np.allclose(preds, 0.25)
    flat = metrics.keyword_overlap_preds([0.3, 0.3], _full(2, 0.4), _full(2, 1.0), 0.3, [0.9])
    assert np.allclose(flat, 0.4)


@pytest.mark.parametrize("values,expected", [([1.0, np.nan, 3.0], 2.0), ([np.nan], math.nan), ([], math.nan)])
def test_nanmean(values, expected):
    result = metrics.nanmean(values)
    assert (math.isnan(result) and math.isnan(expected)) or result == expected
