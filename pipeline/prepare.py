#!/usr/bin/env python3
"""`prepare` stage (specs §6): verify and materialise the committed splits, never re-split.

Asserts train/test share no CV and no company, fails on any Presidio PII finding, then
tokenizes each CV/JD pair (each side truncated to 1024 tokens, JD last, padded to
2048) and caches the tensors. `test` additionally gets a `test_shuffled` cache for the
shuffled-pair control: each test CV paired with a job it is not paired with.

Usage:
    ./pipeline/prepare.py --dataset-dir datasets_smoke --runs-dir runs_smoke

Writes <runs-dir>/<model-slug>/cache/{train,val,calib,test,test_shuffled}.pt.
"""

import argparse
import random

import common
from common import corpus

CV_BUDGET = 1024
JD_BUDGET = 1024
MAX_LENGTH = 2048


class LeakageError(Exception):
    pass


def assert_leakage_free(splits):
    """Raises LeakageError unless `train` and `test` share no CV and no company."""
    def axes(pairs):
        return {p["cv"] for p in pairs}, {corpus.company_of(p["job"]) for p in pairs}

    train_cvs, train_companies = axes(splits.get("train", []))
    test_cvs, test_companies = axes(splits.get("test", []))
    cv_leak, company_leak = train_cvs & test_cvs, train_companies & test_companies
    if cv_leak or company_leak:
        raise LeakageError(
            f"train/test split leakage: shared CVs={sorted(cv_leak)}, "
            f"shared companies={sorted(company_leak)}"
        )


def encode_pair(tokenizer, cv_text, jd_text, cv_budget=CV_BUDGET, jd_budget=JD_BUDGET,
                max_length=MAX_LENGTH, pad_token_id=0):
    """Returns (input_ids, attention_mask, cv_truncated, jd_truncated), padded to
    `max_length`. Must stay identical to frontend/src/tokenize.mjs."""
    cv_ids = tokenizer.encode(cv_text, add_special_tokens=False)
    jd_ids = tokenizer.encode(jd_text, add_special_tokens=False)
    input_ids = (cv_ids[:cv_budget] + jd_ids[:jd_budget])[:max_length]
    pad_len = max_length - len(input_ids)
    return (
        input_ids + [pad_token_id] * pad_len,
        [1] * len(input_ids) + [0] * pad_len,
        len(cv_ids) > cv_budget,
        len(jd_ids) > jd_budget,
    )


def derange_pairs(pairs, all_job_ids, seed=42):
    """Pairs every CV in `pairs` with a job it is not paired with there. The pool is
    the whole corpus: a split can be a full CV x job cross product, where no in-split
    permutation mismatches anything."""
    rng = random.Random(seed)
    partners = {}
    for p in pairs:
        partners.setdefault(p["cv"], set()).add(p["job"])
    shuffled = []
    for p in pairs:
        candidates = [job for job in all_job_ids if job not in partners[p["cv"]]]
        if not candidates:
            raise ValueError(f"{p['cv']} is paired with every job in the corpus -- nothing to shuffle to")
        shuffled.append({"cv": p["cv"], "job": rng.choice(candidates)})
    return shuffled


def _cache_split(tokenizer, dataset_dir, runs_dir, slug, name, pairs):
    import torch

    rows = [
        encode_pair(tokenizer, (dataset_dir / p["cv"]).read_text(encoding="utf-8"),
                    corpus.read_job_body(dataset_dir / p["job"]), pad_token_id=tokenizer.pad_token_id)
        for p in pairs
    ]
    path = common.cache_path(runs_dir, slug, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "pairs": pairs,
        "input_ids": torch.tensor([r[0] for r in rows], dtype=torch.long),
        "attention_mask": torch.tensor([r[1] for r in rows], dtype=torch.long),
    }, path)
    n = len(pairs)
    print(f"{name}: {n} pair(s) -> {path} (cv truncated {sum(r[2] for r in rows)}/{n}, "
          f"jd truncated {sum(r[3] for r in rows)}/{n})")


def _pii_scan(dataset_dir):
    import pii_scrub

    return pii_scrub.scan_paths(sorted(dataset_dir.rglob("*.md")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    args = parser.parse_args()

    with common.stage_span("prepare"):
        splits = corpus.load_splits(args.dataset_dir)
        if not any(splits.values()):
            raise SystemExit(f"no splits found under {args.dataset_dir / 'splits'}")
        try:
            assert_leakage_free(splits)
        except LeakageError as exc:
            raise SystemExit(f"LEAKAGE CHECK FAILED: {exc}")
        print("leakage check passed: train/test disjoint on both CV and company axes")

        if not _pii_scan(args.dataset_dir):
            raise SystemExit("PII CHECK FAILED: see findings above")
        print("PII scan passed: no findings")

        tokenizer = common.load_tokenizer(args.model)
        slug = common.model_slug(args.model)
        for name, pairs in splits.items():
            if pairs:
                _cache_split(tokenizer, args.dataset_dir, args.runs_dir, slug, name, pairs)
        if splits["test"]:
            all_jobs = [job for jobs in corpus.list_jobs(args.dataset_dir).values() for job in jobs]
            shuffled = derange_pairs(splits["test"], all_jobs, seed=args.seed)
            _cache_split(tokenizer, args.dataset_dir, args.runs_dir, slug, "test_shuffled", shuffled)


if __name__ == "__main__":
    main()
