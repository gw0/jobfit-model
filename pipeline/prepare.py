#!/usr/bin/env python3
"""`prepare` stage (specs §6): verify and materialise the committed splits, never re-split.

Asserts train/test share no CV and no company, fails on any Presidio PII finding, then
encodes each CV/JD pair with every question (jev.encode: CV then JD truncated to
STATE_BUDGET tokens, then the question branches in a QUESTIONS_BUDGET block, padded to
their sum) and caches the tensors. `test` additionally gets a `test_shuffled` cache for the
shuffled-pair control: each test CV paired with a job it is not paired with.

Usage:
    ./pipeline/prepare.py --datasets-dir datasets_smoke --runs-dir runs_smoke

Writes <runs-dir>/<candidate>/cache/{train,val,calib,test,test_shuffled}.pt.
"""

import argparse
import random

import common
import corpus
import jev


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


def derange_pairs(pairs, all_job_ids, seed=corpus.DEFAULT_SEED):
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


def _cache_split(tokenizer, datasets_dir, run_dir, name, pairs):
    import torch

    rows = [
        jev.encode(tokenizer, common.jobfit_state((datasets_dir / p["cv"]).read_text(encoding="utf-8"),
                                           corpus.read_job_body(datasets_dir / p["job"])),
                   common.QUESTIONS, common.STATE_BUDGET, common.QUESTIONS_BUDGET, tokenizer.pad_token_id)
        for p in pairs
    ]
    candidate_ids, candidate_counts = jev.candidate_ids(tokenizer, common.QUESTIONS)
    path = common.cache_path(run_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "pairs": pairs,
        **{key: torch.tensor([r[key] for r in rows], dtype=torch.long) for key in common.MODEL_INPUTS},
        "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
        "candidate_counts": candidate_counts,
    }, path)
    n = len(pairs)
    print(f"{name}: {n} pair(s) -> {path} (state truncated {sum(r['truncated'] for r in rows)}/{n})")


def _pii_scan(datasets_dir):
    import pii_scrub

    return pii_scrub.scan_paths([datasets_dir / "cvs", datasets_dir / "jobs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    common.add_seed_arg(parser)
    args = common.parse_args(parser, "prepare")

    with common.stage_span("prepare"):
        splits = corpus.load_splits(args.datasets_dir)
        if not any(splits.values()):
            raise SystemExit(f"no splits found under {args.datasets_dir / 'splits'}")
        try:
            assert_leakage_free(splits)
        except LeakageError as exc:
            raise SystemExit(f"LEAKAGE CHECK FAILED: {exc}")
        print("leakage check passed: train/test disjoint on both CV and company axes")

        if not _pii_scan(args.datasets_dir):
            raise SystemExit("PII CHECK FAILED: see findings above")
        print("PII scan passed: no findings")

        tokenizer = common.load_tokenizer(args.model)
        questions_len = sum(len(tokenizer.encode(jev.render_question(q), add_special_tokens=False))
                            for q in common.QUESTIONS.values())
        print(f"questions block: {questions_len}/{common.QUESTIONS_BUDGET} tokens")
        for name, pairs in splits.items():
            if pairs:
                _cache_split(tokenizer, args.datasets_dir, args.run_dir, name, pairs)
        if splits["test"]:
            all_jobs = [job for jobs in corpus.list_jobs(args.datasets_dir).values() for job in jobs]
            shuffled = derange_pairs(splits["test"], all_jobs, seed=args.seed)
            _cache_split(tokenizer, args.datasets_dir, args.run_dir, "test_shuffled", shuffled)


if __name__ == "__main__":
    main()
