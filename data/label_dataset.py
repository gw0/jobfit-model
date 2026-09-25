#!/usr/bin/env python3
"""LLM-as-judge labeling of the corpus (specs §5).

The 13 pairwise questions and the 2 clarity questions (questions.json) are judged via
`claude -p`; the 2 LLM-generated questions use an open-source AI-text detector
(ai_text_detector.py).

Usage:
    ./data/label_dataset.py --out-dir datasets_smoke
    ./data/label_dataset.py --out-dir datasets_smoke --double-label --double-label-sample 3
    ./data/label_dataset.py --out-dir datasets_full --double-label --double-label-sample 300 --workers 8

Labels every pair in <out-dir>/splits/ plus every CV/job, writing
<out-dir>/labels/{cvs,jobs,pairs}.jsonl. Each record is appended as soon as its judge
call returns, and already-labeled docs/pairs are kept as-is and skipped -- so a run
that got interrupted (a bad `claude` auth, a rate limit, a crash, ...) can just be
re-run to pick up where it left off, and a run after adding new CVs/jobs only labels
the new ones. Pass --force to relabel
everything from scratch (e.g. after switching --model). --double-label re-judges a
sample of one split with a second model into labels/*_double.jsonl for inter-rater QC:
the bulk pass runs on Sonnet, the few-hundred-call QC pass on the stronger Opus.
Needs an authenticated `claude` CLI in the calling shell.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import corpus
from claude_json import UsageLimitError
from label_docs import label_doc
from label_pairs import label_pairs_for_cv, load_pairwise_questions

singleton_chunks = lambda todo: [[item] for item in todo]  # noqa: E731 -- one item per call


def _chunk_by_cv(pairs, max_size):
    """Groups consecutive same-CV pairs into chunks of at most `max_size` (a CV's run
    longer than `max_size` splits into multiple same-CV chunks; a chunk never spans two
    CVs). Assumes `pairs` is already CV-grouped, which build_dataset.py's splits are."""
    chunks = []
    for pair in pairs:
        if chunks and chunks[-1][0]["cv"] == pair["cv"] and len(chunks[-1]) < max_size:
            chunks[-1].append(pair)
        else:
            chunks.append([pair])
    return chunks


def _label_all(what, units, fn, workers=1, on_record=None):
    """Labels each chunk in `units` on `workers` threads; `fn(chunk)` returns a list of
    records, one per item in the chunk. Calls `on_record` on the main thread for each
    record as its chunk completes. Skips a chunk's records on failure (one bad judge
    reply shouldn't lose the whole batch) but raises if every chunk failed -- that is a
    setup problem -- and stops at once on the usage limit, keeping records already
    passed on."""
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, chunk): chunk for chunk in units}
        for n, future in enumerate(as_completed(futures), 1):
            chunk = futures[future]
            try:
                chunk_records = future.result()
            except UsageLimitError as exc:
                pool.shutdown(cancel_futures=True)
                raise SystemExit(f"{exc} -- stopped after {len(records)} new {what} label(s); "
                                 "re-run once it resets to resume")
            except Exception as exc:  # noqa: BLE001
                print(f"[{n}/{len(units)}] {what} {chunk} SKIPPED ({exc})")
                continue
            print(f"[{n}/{len(units)}] labeled {what} {chunk}")
            records.extend(chunk_records)
            if on_record is not None:
                for record in chunk_records:
                    on_record(record)
    if units and not records:
        raise SystemExit(f"every {what} labeling call failed -- check `claude` auth and the model id")
    return records


def _label_file(path, what, items, item_key, record_key, fn, force, workers, chunk_fn):
    """Labels the items not yet in `path` and rewrites it in `items` order. Each new
    record is appended to `path` as soon as it completes, so an interrupted run (a bad
    `claude` auth, a rate limit, a crash) resumes where it stopped. `item_key` and
    `record_key` extract the same identity from an input item and a record -- they
    differ in shape (e.g. a bare CV id string vs. its {"cv": ...} label record).
    `chunk_fn` groups the not-yet-labeled items into the chunks `fn` is called with."""
    by_key = {} if force else {record_key(r): r for r in corpus.load_jsonl(path)}
    todo = [item for item in items if item_key(item) not in by_key]
    if len(todo) < len(items):
        print(f"{what}: {len(items) - len(todo)} already labeled, {len(todo)} to label")

    path.parent.mkdir(parents=True, exist_ok=True)

    def append(record):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

    for record in _label_all(what, chunk_fn(todo), fn, workers, on_record=append):
        by_key[record_key(record)] = record
    records = [by_key[item_key(item)] for item in items if item_key(item) in by_key]
    corpus.write_jsonl(path, records)
    print(f"wrote {len(records)} record(s) -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=corpus.DEFAULT_DATASET_DIR)
    parser.add_argument("--model", default="claude-sonnet-5", help="model for the primary judging pass")
    parser.add_argument("--force", action="store_true",
                        help="relabel everything, ignoring existing labels/*.jsonl (e.g. after switching --model)")
    parser.add_argument("--double-label", action="store_true",
                        help="also label a sample with a second judge model for inter-rater QC")
    parser.add_argument("--double-label-sample", type=int, default=5)
    parser.add_argument("--double-label-from", default="test", choices=corpus.SPLIT_NAMES)
    parser.add_argument("--double-label-model", default="claude-opus-5-5",
                        help="a different model than --model, so the QC pass is a genuine second rater")
    parser.add_argument("--workers", type=int, default=1, help="parallel `claude` calls")
    parser.add_argument("--jobs-per-call", type=int, default=5,
                        help="judge up to this many of a CV's jobs in one claude call for the primary "
                             "pairs.jsonl pass; the double-label QC pass always uses one job per call")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir, labels_dir = args.out_dir, args.out_dir / "labels"
    cv_bodies = {cv_id: (out_dir / cv_id).read_text(encoding="utf-8") for cv_id in corpus.list_cvs(out_dir)}
    if not cv_bodies:
        raise SystemExit(f"no CVs found under {out_dir / 'cvs'}")
    job_bodies = {
        job_id: corpus.read_job_body(out_dir / job_id)
        for job_ids in corpus.list_jobs(out_dir).values() for job_id in job_ids
    }
    if not job_bodies:
        raise SystemExit(f"no jobs found under {out_dir / 'jobs'}")
    splits = corpus.load_splits(out_dir)
    pairs = list({(p["cv"], p["job"]): p for split in splits.values() for p in split}.values())
    if not pairs:
        raise SystemExit(f"no pairs found under {out_dir / 'splits'} -- run build_dataset.py first")
    pairwise_questions = load_pairwise_questions()
    pair_key = lambda p: (p["cv"], p["job"])  # noqa: E731 -- shared by primary and double-label pairs

    def pair_labeler(judge_model=None):
        def fn(chunk):
            cv_id = chunk[0]["cv"]
            jobs = [(pair["job"], job_bodies[pair["job"]]) for pair in chunk]
            return label_pairs_for_cv(cv_id, cv_bodies[cv_id], jobs, pairwise_questions, judge_model=judge_model)
        return fn

    identity = lambda item: item  # noqa: E731 -- CV/job items are bare id strings
    cv_key, job_key = (lambda r: r["cv"]), (lambda r: r["job"])  # noqa: E731

    def label(name, what, items, item_key, record_key, fn, chunk_fn=singleton_chunks):
        _label_file(labels_dir / name, what, items, item_key, record_key, fn, args.force, args.workers, chunk_fn)

    label("cvs.jsonl", "CV", list(cv_bodies), identity, cv_key,
          lambda chunk: [label_doc("cv", cv_id, cv_bodies[cv_id], model=args.model) for cv_id in chunk])
    label("jobs.jsonl", "job", list(job_bodies), identity, job_key,
          lambda chunk: [label_doc("job", job_id, job_bodies[job_id], model=args.model) for job_id in chunk])
    label("pairs.jsonl", "pair", pairs, pair_key, pair_key, pair_labeler(args.model),
          chunk_fn=lambda todo: _chunk_by_cv(todo, args.jobs_per_call))

    if not args.double_label:
        return
    split_pairs = splits[args.double_label_from]
    if not split_pairs:
        print(f"WARNING: {args.double_label_from!r} split is empty, nothing to double-label")
        return
    model = args.double_label_model
    sample = random.Random(args.seed).sample(split_pairs, min(args.double_label_sample, len(split_pairs)))
    sample_cvs, sample_jobs = sorted({p["cv"] for p in sample}), sorted({p["job"] for p in sample})

    label("pairs_double.jsonl", "pair", sample, pair_key, pair_key, pair_labeler(model))
    # The sampled pairs' own CVs/jobs, so the doc-level QC covers the same documents.
    label("cvs_double.jsonl", "CV", sample_cvs, identity, cv_key,
          lambda chunk: [label_doc("cv", cv_id, cv_bodies[cv_id], model=model, double_label=True)
                        for cv_id in chunk])
    label("jobs_double.jsonl", "job", sample_jobs, identity, job_key,
          lambda chunk: [label_doc("job", job_id, job_bodies[job_id], model=model, double_label=True)
                        for job_id in chunk])

if __name__ == "__main__":
    main()
