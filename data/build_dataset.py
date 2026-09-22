#!/usr/bin/env python3
"""Builds train/val/calib/test splits over the corpus (specs §5).

Splits are grouped by CV identity, and `test` shares no CV and no company with the
other splits; pairs that would break that are dropped.

Usage:
    ./data/build_dataset.py --out-dir datasets_smoke

Reads <out-dir>/cvs/*.md and <out-dir>/jobs/<company>/*.md, writes
<out-dir>/splits/{train,val,calib,test}.jsonl of {"cv": ..., "job": ...} path-ids.
"""

import argparse
import random
from pathlib import Path

import corpus

DEFAULT_CV_RATIOS = (0.7, 0.1, 0.1, 0.1)   # train, val, calib, test
DEFAULT_TEST_COMPANY_RATIO = 0.2
DEFAULT_SEED = 42


def _partition_by_ratio(items, ratios, rng):
    """Shuffles `items` and splits them into len(ratios) groups sized by `ratios`."""
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    counts = [round(r * n) for r in ratios]
    counts[0] += n - sum(counts)  # rounding drift goes to train
    groups, i = [], 0
    for c in counts:
        groups.append(items[i:i + c])
        i += c
    return groups


def make_splits(cv_ids, jobs_by_company, seed=DEFAULT_SEED,
                cv_ratios=DEFAULT_CV_RATIOS, test_company_ratio=DEFAULT_TEST_COMPANY_RATIO):
    """Assigns every CV to one split and every company to either the test-only pool
    or the train/val/calib pool, then pairs each CV with every job of its pool."""
    rng = random.Random(seed)

    cv_split = {}
    for split_name, group in zip(corpus.SPLIT_NAMES, _partition_by_ratio(cv_ids, cv_ratios, rng)):
        for cv_id in group:
            cv_split[cv_id] = split_name

    companies = list(jobs_by_company)
    rng.shuffle(companies)
    n_test_companies = max(1, round(test_company_ratio * len(companies))) if len(companies) > 1 else 0
    test_companies = set(companies[:n_test_companies])

    splits = {name: [] for name in corpus.SPLIT_NAMES}
    for cv_id, split_name in cv_split.items():
        for company, job_ids in jobs_by_company.items():
            if (split_name == "test") != (company in test_companies):
                continue  # would leak a CV or a company between train and test
            splits[split_name].extend({"cv": cv_id, "job": job_id} for job_id in job_ids)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=corpus.DEFAULT_DATASET_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    cv_ids, jobs_by_company = corpus.list_cvs(args.out_dir), corpus.list_jobs(args.out_dir)
    if not cv_ids:
        raise SystemExit(f"no CVs found under {args.out_dir / 'cvs'}")
    if not jobs_by_company:
        raise SystemExit(f"no jobs found under {args.out_dir / 'jobs'}")

    for name, pairs in make_splits(cv_ids, jobs_by_company, seed=args.seed).items():
        path = args.out_dir / "splits" / f"{name}.jsonl"
        corpus.write_jsonl(path, pairs)
        print(f"{name}: {len(pairs)} pairs -> {path}")


if __name__ == "__main__":
    main()
