#!/usr/bin/env python3
"""LLM-as-judge labeling of the corpus (specs §5).

The 13 pairwise questions and the 2 clarity questions (questions.json) are judged via
`claude -p`; the 2 LLM-generated questions use an open-source AI-text detector
(ai_text_detector.py).

Usage:
    ./data/label_dataset.py --out-dir datasets_smoke
    ./data/label_dataset.py --out-dir datasets_smoke --double-label-sample 3
    ./data/label_dataset.py --out-dir datasets_full --double-label-sample 300 --workers 8

Labels every pair in <out-dir>/splits/ plus every CV/job, writing
<out-dir>/labels/{cvs,jobs,pairs}.jsonl. Each record is appended as soon as its judge
call returns, and already-labeled docs/pairs are kept as-is and skipped -- so a run
that got interrupted (a bad `claude` auth, a rate limit, a crash, ...) can just be
re-run to pick up where it left off, and a run after adding new CVs/jobs only labels
the new ones. Pass --force to relabel
everything from scratch (e.g. after switching --model). --double-label-sample N re-judges
N test pairs (and their CVs/jobs) with a second model into labels/*_double.jsonl for
inter-rater QC: the bulk pass runs on Sonnet, the few-hundred-call QC pass on the
stronger Opus.
Needs an authenticated `claude` CLI in the calling shell.
"""

import argparse
import random
from pathlib import Path

import corpus
from claude_json import run_parallel
from label_docs import label_doc
from label_pairs import label_pairs_for_cv
from rubric import load_questions


def _key(item):
    """Identity of a work item or its label record: items are {"cv"}, {"job"} or {"cv", "job"}."""
    return item.get("cv"), item.get("job")


def _chunk_by_cv(items, max_size):
    """Groups consecutive same-CV items into chunks of at most `max_size` (a CV's run
    longer than `max_size` splits into multiple same-CV chunks; a chunk never spans two
    CVs). Assumes `items` is already CV-grouped, which build_dataset.py's splits are."""
    chunks = []
    for item in items:
        if chunks and chunks[-1][0]["cv"] == item["cv"] and len(chunks[-1]) < max_size:
            chunks[-1].append(item)
        else:
            chunks.append([item])
    return chunks


def _label_file(path, what, items, fn, force, workers, chunk_size=1):
    """Labels the items not yet in `path` and rewrites it in `items` order. `fn(chunk)`
    returns one record per item of a chunk of up to `chunk_size` items. Each new record
    is appended to `path` as soon as it completes, so an interrupted run (a bad `claude`
    auth, a rate limit, a crash) resumes where it stopped."""
    if force:
        path.unlink(missing_ok=True)
    by_key = {_key(r): r for r in corpus.load_jsonl(path)}
    todo = [item for item in items if _key(item) not in by_key]
    if len(todo) < len(items):
        print(f"{what}: {len(items) - len(todo)} already labeled, {len(todo)} to label")

    path.parent.mkdir(parents=True, exist_ok=True)

    def keep(_, records):
        for record in records:
            corpus.append_jsonl(path, record)
            by_key[_key(record)] = record

    chunks = _chunk_by_cv(todo, chunk_size)
    if chunks and not run_parallel(what, chunks, fn, workers, keep):
        raise SystemExit(f"every {what} labeling call failed -- check `claude` auth and the model id")
    records = [by_key[_key(item)] for item in items if _key(item) in by_key]
    corpus.write_jsonl(path, records)
    print(f"wrote {len(records)} record(s) -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="claude-sonnet-5-5", help="model for the primary judging pass")
    parser.add_argument("--force", action="store_true",
                        help="relabel everything, ignoring existing labels/*.jsonl (e.g. after switching --model)")
    parser.add_argument("--double-label-sample", type=int, default=0,
                        help="also label this many test pairs with a second judge model for inter-rater QC")
    parser.add_argument("--double-label-model", default="claude-opus-5-5",
                        help="a different model than --model, so the QC pass is a genuine second rater")
    parser.add_argument("--workers", type=int, default=1, help="parallel `claude` calls")
    parser.add_argument("--jobs-per-call", type=int, default=10,
                        help="judge up to this many of a CV's jobs in one claude call for the primary "
                             "pairs.jsonl pass; the double-label QC pass always uses one job per call")
    parser.add_argument("--seed", type=int, default=corpus.DEFAULT_SEED)
    args = parser.parse_args()

    out_dir, labels_dir = args.out_dir, args.out_dir / "labels"
    cv_ids = corpus.list_cvs(out_dir)
    if not cv_ids:
        raise SystemExit(f"no CVs found under {out_dir / 'cvs'}")
    job_ids = [job_id for ids in corpus.list_jobs(out_dir).values() for job_id in ids]
    if not job_ids:
        raise SystemExit(f"no jobs found under {out_dir / 'jobs'}")
    splits = corpus.load_splits(out_dir)
    pairs = [pair for split in splits.values() for pair in split]
    if not pairs:
        raise SystemExit(f"no pairs found under {out_dir / 'splits'} -- run build_dataset.py first")
    texts = corpus.load_texts(out_dir)
    questions = load_questions("pairwise")

    def doc_labeler(kind, model, double_label=False):
        return lambda chunk: [label_doc(kind, item[kind], texts[item[kind]], model=model, double_label=double_label)
                              for item in chunk]

    def pair_labeler(model):
        def fn(chunk):
            cv_id = chunk[0]["cv"]
            jobs = [(pair["job"], texts[pair["job"]]) for pair in chunk]
            return label_pairs_for_cv(cv_id, texts[cv_id], jobs, questions, model=model)
        return fn

    def label(name, what, items, fn, chunk_size=1):
        _label_file(labels_dir / name, what, items, fn, args.force, args.workers, chunk_size)

    label("cvs.jsonl", "CV", [{"cv": cv_id} for cv_id in cv_ids], doc_labeler("cv", args.model))
    label("jobs.jsonl", "job", [{"job": job_id} for job_id in job_ids], doc_labeler("job", args.model))
    label("pairs.jsonl", "pair", pairs, pair_labeler(args.model), args.jobs_per_call)

    test_pairs = splits["test"]
    if not args.double_label_sample or not test_pairs:
        return
    model = args.double_label_model
    sample = random.Random(args.seed).sample(test_pairs, min(args.double_label_sample, len(test_pairs)))

    label("pairs_double.jsonl", "pair", sample, pair_labeler(model))
    # The sampled pairs' own CVs/jobs, so the doc-level QC covers the same documents.
    label("cvs_double.jsonl", "CV", [{"cv": cv_id} for cv_id in sorted({p["cv"] for p in sample})],
          doc_labeler("cv", model, double_label=True))
    label("jobs_double.jsonl", "job", [{"job": job_id} for job_id in sorted({p["job"] for p in sample})],
          doc_labeler("job", model, double_label=True))


if __name__ == "__main__":
    main()
