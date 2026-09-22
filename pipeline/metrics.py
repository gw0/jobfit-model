"""Evaluation statistics over numpy arrays: masking, per-aspect MAE/Spearman,
bootstrap CIs, and the two non-learned baselines. No model or file I/O.

Shapes: preds_q (N, NUM_ASPECTS, len(QUANTILES)); targets/confidences (N, NUM_ASPECTS).
"""

import math
import warnings

import numpy as np
from scipy import stats

from common import ASPECT_IDS, MID


def masked_columns(preds, targets, confidences, confidence_floor):
    """Per aspect: (aspect_id, preds[rows, j], targets[rows, j], rows), where `rows`
    selects the examples whose target is present and at/above the confidence floor --
    the same masking the training loss applies. `preds` may be (N, A) or (N, A, Q)."""
    preds, targets = np.asarray(preds, dtype=float), np.asarray(targets, dtype=float)
    mask = ~np.isnan(targets) & (np.asarray(confidences, dtype=float) >= confidence_floor)
    for j, aid in enumerate(ASPECT_IDS):
        rows = mask[:, j]
        yield aid, preds[rows, j], targets[rows, j], rows


def mae(preds, targets):
    return float(np.mean(np.abs(preds - targets))) if len(targets) else math.nan


def spearman(preds, targets):
    if len(targets) < 2:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant input -> NaN is the right answer here
        return float(stats.spearmanr(preds, targets).statistic)


def nanmean(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values[~np.isnan(values)])) if (~np.isnan(values)).any() else math.nan


def per_aspect_mae(preds, targets, confidences, confidence_floor):
    """{aspect_id: MAE}; for (N, A, Q) preds the mid quantile is scored."""
    preds = np.asarray(preds, dtype=float)
    if preds.ndim == 3:
        preds = preds[..., MID]
    return {aid: mae(p, t) for aid, p, t, _ in masked_columns(preds, targets, confidences, confidence_floor)}


def mean_mae(preds_q, targets, confidences, confidence_floor):
    """Mean over aspects of the per-aspect mid-quantile MAE (aspects with no labels skipped)."""
    return nanmean(list(per_aspect_mae(preds_q, targets, confidences, confidence_floor).values()))


def compute_metrics(preds_q, targets, confidences, confidence_floor):
    """{aspect_id: {"n", "mae", "spearman_rho"}} on the mid quantile."""
    preds_mid = np.asarray(preds_q, dtype=float)[..., MID]
    return {
        aid: {"n": len(t), "mae": mae(p, t), "spearman_rho": spearman(p, t)}
        for aid, p, t, _ in masked_columns(preds_mid, targets, confidences, confidence_floor)
    }


# --- bootstrap ----------------------------------------------------------------------------

def bootstrap_ci(values_by_group, n_resamples=1000, seed=42, level=0.90):
    """Percentile CI of the pooled mean, resampling whole groups (e.g. every pair of one
    CV) with replacement. (NaN, NaN) with fewer than 2 non-empty groups."""
    groups = [np.asarray(v, dtype=float) for v in values_by_group if len(v)]
    if len(groups) < 2:
        return math.nan, math.nan
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    idx = np.random.default_rng(seed).integers(0, len(groups), size=(n_resamples, len(groups)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def bootstrap_mae_ci(preds_q, targets, confidences, confidence_floor, group_ids, **kwargs):
    """{aspect_id: {"mae_ci_lo", "mae_ci_hi"}}: per-aspect mid-quantile MAE CIs over
    groups (CV identity), on the same masked examples as every other metric."""
    preds_mid = np.asarray(preds_q, dtype=float)[..., MID]
    group_ids = np.asarray(group_ids)
    result = {}
    for aid, p, t, rows in masked_columns(preds_mid, targets, confidences, confidence_floor):
        errors, groups = np.abs(p - t), group_ids[rows]
        lo, hi = bootstrap_ci([errors[groups == g] for g in np.unique(groups)], **kwargs)
        result[aid] = {"mae_ci_lo": lo, "mae_ci_hi": hi}
    return result


# --- non-learned baselines (specs §4) ----------------------------------------------------

def train_mean_preds(train_targets, train_confidences, confidence_floor, n):
    """(n, A) predictions of the per-aspect mean train target -- the trivial floor."""
    means = [np.mean(t) if len(t) else math.nan
             for _, _, t, _ in masked_columns(train_targets, train_targets, train_confidences, confidence_floor)]
    return np.tile(np.array(means, dtype=float), (n, 1))


def keyword_overlap(cv_text, jd_text):
    """Jaccard similarity of lowercased word sets."""
    cv_words, jd_words = set(cv_text.lower().split()), set(jd_text.lower().split())
    if not cv_words or not jd_words:
        return 0.0
    return len(cv_words & jd_words) / len(cv_words | jd_words)


def keyword_overlap_preds(train_overlaps, train_targets, train_confidences, confidence_floor, overlaps):
    """(n, A) predictions of a per-aspect linear fit overlap -> score, fit on train."""
    train_overlaps, overlaps = np.asarray(train_overlaps, dtype=float), np.asarray(overlaps, dtype=float)
    columns = []
    for _, _, t, rows in masked_columns(train_targets, train_targets, train_confidences, confidence_floor):
        x = train_overlaps[rows]
        if len(t) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, t, 1)
        else:
            slope, intercept = 0.0, (np.mean(t) if len(t) else math.nan)
        columns.append(slope * overlaps + intercept)
    return np.stack(columns, axis=1) if columns else np.empty((len(overlaps), 0))
