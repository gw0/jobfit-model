"""Evaluation and calibration statistics over numpy arrays: masking, per-question
MAE/Spearman, calibration (temperature + confidence threshold) and confidence-vs-error,
bootstrap CIs, and the two non-learned baselines. No model or file I/O.

Shapes: scores/targets/confidences (N, NUM_QUESTIONS), scores and targets on the labels'
[0,1] scale; answer_logits (N, NUM_QUESTIONS, MAX_CANDIDATES).
"""

import math
import warnings

import numpy as np
from scipy import stats

import common
import jev
from common import QUESTION_IDS, usable

INSUFFICIENT_PERCENTILE = 10  # the least-confident 10% of calib answers read "insufficient data"
EMPTY_CALIBRATION = {"temperature": math.nan, "confidence_threshold": math.nan}
CONFIDENCE_BINS = 5


def masked_columns(preds, targets, confidences, confidence_floor):
    """Per question: (question_id, preds[rows, j], targets[rows, j], rows), where `rows`
    selects the examples whose target is present and at/above the confidence floor --
    the same masking the training loss applies."""
    preds, targets = np.asarray(preds, dtype=float), np.asarray(targets, dtype=float)
    mask = usable(targets, confidences, confidence_floor)
    for j, qid in enumerate(QUESTION_IDS):
        rows = mask[:, j]
        yield qid, preds[rows, j], targets[rows, j], rows


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


def per_question_mae(scores, targets, confidences, confidence_floor):
    """{question_id: MAE}."""
    return {qid: mae(p, t) for qid, p, t, _ in masked_columns(scores, targets, confidences, confidence_floor)}


def mean_of(per_question, key):
    """Mean over questions of one entry of {question_id: {key: value}} (NaN skipped)."""
    return nanmean([m[key] for m in per_question.values()])


def compute_metrics(scores, targets, confidences, confidence_floor):
    """{question_id: {"n", "mae", "spearman_rho"}}."""
    return {
        qid: {"n": len(t), "mae": mae(p, t), "spearman_rho": spearman(p, t)}
        for qid, p, t, _ in masked_columns(scores, targets, confidences, confidence_floor)
    }


# --- calibration ----------------------------------------------------------------------------

def confidence_threshold(confidences):
    """Nearest-rank percentile of answer confidences: less confident answers read
    "insufficient data" in the UI."""
    confidences = np.sort(np.asarray(confidences, dtype=float))
    if len(confidences) == 0:
        return math.nan
    rank = math.ceil(INSUFFICIENT_PERCENTILE / 100 * len(confidences))
    return float(confidences[min(len(confidences) - 1, max(0, rank - 1))])


def fit_calibration(answer_logits, targets, confidences, confidence_floor):
    """{"temperature", "confidence_threshold"}, each pooled over every question: one
    model-level temperature (jev.fit_temperature against the two-hot targets), then the
    threshold over the temperature-scaled confidences of the labeled answers. NaN
    where there is nothing to fit on."""
    answer_logits = np.asarray(answer_logits, dtype=float)
    dists = common.answer_targets(targets)
    weights = common.label_weights(targets, confidences, confidence_floor)
    counts = [common.num_levels(qid) for qid in QUESTION_IDS]
    temperature = jev.fit_temperature([answer_logits[:, j, :k] for j, k in enumerate(counts)],
                                      [dists[:, j, :k] for j, k in enumerate(counts)],
                                      [weights[:, j] for j in range(len(counts))])
    if math.isnan(temperature):
        return EMPTY_CALIBRATION
    _, answer_confidences = common.read_scores(answer_logits, temperature)
    return {"temperature": temperature, "confidence_threshold": confidence_threshold(answer_confidences[weights > 0])}


def confidence_error(scores, answer_confidences, targets, confidences, confidence_floor, threshold):
    """{question_id: {"confidence_error_spearman", "confidence_error_bins",
    "insufficient_rate"}}: Spearman between answer confidence and |score - target|
    (should be clearly negative), the mean |error| per equal-width confidence bin, and
    the share of answers below `threshold` -- all under a fixed calibration, never
    re-fit here."""
    edges = np.linspace(0.0, 1.0, CONFIDENCE_BINS + 1)
    result = {}
    for qid, p, t, rows in masked_columns(scores, targets, confidences, confidence_floor):
        conf, err = np.asarray(answer_confidences, dtype=float)[rows, QUESTION_IDS.index(qid)], np.abs(p - t)
        bins = np.clip(np.digitize(conf, edges[1:-1]), 0, CONFIDENCE_BINS - 1)
        result[qid] = {
            "confidence_error_spearman": spearman(conf, err),
            "confidence_error_bins": [
                {"confidence_lo": float(edges[b]), "confidence_hi": float(edges[b + 1]),
                 "n": int((bins == b).sum()), "mean_abs_error": mae(p[bins == b], t[bins == b])}
                for b in range(CONFIDENCE_BINS)
            ],
            "insufficient_rate": (float(np.mean(~(conf >= threshold))) if len(conf) else math.nan),
        }
    return result


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


def bootstrap_mae_ci(scores, targets, confidences, confidence_floor, group_ids, **kwargs):
    """{question_id: {"mae_ci_lo", "mae_ci_hi"}}: per-question MAE CIs over groups (CV
    identity), on the same masked examples as every other metric."""
    group_ids = np.asarray(group_ids)
    result = {}
    for qid, p, t, rows in masked_columns(scores, targets, confidences, confidence_floor):
        errors, groups = np.abs(p - t), group_ids[rows]
        lo, hi = bootstrap_ci([errors[groups == g] for g in np.unique(groups)], **kwargs)
        result[qid] = {"mae_ci_lo": lo, "mae_ci_hi": hi}
    return result


# --- non-learned baselines (specs §4) ----------------------------------------------------

def train_mean_preds(train_targets, train_confidences, confidence_floor, n):
    """(n, Q) predictions of the per-question mean train target -- the trivial floor."""
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
    """(n, Q) predictions of a per-question linear fit overlap -> score, fit on train."""
    train_overlaps, overlaps = np.asarray(train_overlaps, dtype=float), np.asarray(overlaps, dtype=float)
    columns = []
    for _, _, t, rows in masked_columns(train_targets, train_targets, train_confidences, confidence_floor):
        x = train_overlaps[rows]
        if len(t) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, t, 1)
        else:
            slope, intercept = 0.0, (np.mean(t) if len(t) else math.nan)
        columns.append(slope * overlaps + intercept)
    return np.stack(columns, axis=1)
