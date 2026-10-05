#!/usr/bin/env python3
"""Builds train/val/calib/test splits over the corpus (specs §5).

Splits are grouped by CV identity, and `test` shares no CV and no company with the
other splits; pairs that would break that are dropped. By default each CV is paired
with every job of its pool; --jobs-per-cv N samples N of them instead: half the
nearest by TF-IDF cosine (likely fits), half uniformly at random, so labels span the
whole fit range without labeling every combination.

Usage:
    ./data/build_dataset.py --out-dir datasets_smoke
    ./data/build_dataset.py --out-dir datasets_full --jobs-per-cv 18

Reads <out-dir>/cvs/*.md and <out-dir>/jobs/<company>/*.md, writes
<out-dir>/splits/{train,val,calib,test}.jsonl of {"cv": ..., "job": ...} path-ids.
"""

import argparse
import math
import random
import re
from collections import Counter
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


def tfidf_vectors(texts):
    """{doc_id: {term: tf-idf weight}}; terms common to every document weigh 0."""
    bags = {doc_id: Counter(re.findall(r"[a-z][a-z0-9+#]+", text.lower())) for doc_id, text in texts.items()}
    df = Counter(term for bag in bags.values() for term in bag)
    return {doc_id: {term: tf * math.log(len(bags) / df[term]) for term, tf in bag.items()}
            for doc_id, bag in bags.items()}


def cosine(a, b):
    dot = sum(w * b.get(term, 0.0) for term, w in a.items())
    norm = math.sqrt(sum(w * w for w in a.values()) * sum(w * w for w in b.values()))
    return dot / norm if norm else 0.0


def sample_jobs(cv_vector, job_ids, job_vectors, k, rng):
    """k of `job_ids`: the k//2 nearest to the CV, the rest drawn uniformly from the others."""
    if k >= len(job_ids):
        return list(job_ids)
    ranked = sorted(job_ids, key=lambda job_id: -cosine(cv_vector, job_vectors[job_id]))
    nearest = ranked[:k // 2]
    return sorted(nearest + rng.sample(ranked[k // 2:], k - len(nearest)))


def make_splits(cv_ids, jobs_by_company, seed=DEFAULT_SEED, cv_ratios=DEFAULT_CV_RATIOS,
                test_company_ratio=DEFAULT_TEST_COMPANY_RATIO, jobs_per_cv=None, texts=None):
    """Assigns every CV to one split and every company to either the test-only pool
    or the train/val/calib pool, then pairs each CV with the jobs of its pool: all of
    them, or `jobs_per_cv` sampled by sample_jobs() over `texts` ({doc_id: text})."""
    rng = random.Random(seed)

    cv_split = {}
    for split_name, group in zip(corpus.SPLIT_NAMES, _partition_by_ratio(cv_ids, cv_ratios, rng)):
        for cv_id in group:
            cv_split[cv_id] = split_name

    companies = list(jobs_by_company)
    rng.shuffle(companies)
    n_test_companies = max(1, round(test_company_ratio * len(companies))) if len(companies) > 1 else 0
    test_companies = set(companies[:n_test_companies])
    pools = {is_test: [job_id for company, job_ids in jobs_by_company.items()
                       if (company in test_companies) == is_test for job_id in job_ids]
             for is_test in (False, True)}  # a CV only ever pairs within its pool: no leakage
    vectors = tfidf_vectors(texts) if jobs_per_cv is not None else None

    splits = {name: [] for name in corpus.SPLIT_NAMES}
    for cv_id, split_name in cv_split.items():
        pool = pools[split_name == "test"]
        if jobs_per_cv is not None:
            pool = sample_jobs(vectors[cv_id], pool, vectors, jobs_per_cv, rng)
        splits[split_name].extend({"cv": cv_id, "job": job_id} for job_id in pool)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--jobs-per-cv", type=int, default=None, help="sample this many jobs per CV (default: all)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    cv_ids, jobs_by_company = corpus.list_cvs(args.out_dir), corpus.list_jobs(args.out_dir)
    if not cv_ids:
        raise SystemExit(f"no CVs found under {args.out_dir / 'cvs'}")
    if not jobs_by_company:
        raise SystemExit(f"no jobs found under {args.out_dir / 'jobs'}")
    texts = None
    if args.jobs_per_cv is not None:
        texts = {cv_id: (args.out_dir / cv_id).read_text(encoding="utf-8") for cv_id in cv_ids}
        texts.update({job_id: corpus.read_job_body(args.out_dir / job_id)
                      for job_ids in jobs_by_company.values() for job_id in job_ids})

    splits = make_splits(cv_ids, jobs_by_company, seed=args.seed, jobs_per_cv=args.jobs_per_cv, texts=texts)
    for name, pairs in splits.items():
        path = args.out_dir / "splits" / f"{name}.jsonl"
        corpus.write_jsonl(path, pairs)
        print(f"{name}: {len(pairs)} pairs -> {path}")


if __name__ == "__main__":
    main()
