#!/usr/bin/env python3
"""LLM-as-judge labeling of the corpus (specs §5).

The 13 pairwise aspects and the 2 clarity aspects are judged via `claude -p`; the 2
LLM-generated aspects use an open-source AI-text detector (ai_text_detector.py).

Usage:
    ./data/label_dataset.py --out-dir datasets_smoke
    ./data/label_dataset.py --out-dir datasets_smoke --double-label --double-label-sample 3

Labels every pair in <out-dir>/splits/ plus every CV/job, writing
<out-dir>/labels/{cvs,jobs,pairs}.jsonl. --double-label re-judges a sample of one split
with a second model into labels/*_double.jsonl for inter-rater QC. Needs an
authenticated `claude` CLI in the calling shell.
"""

import argparse
import random
from pathlib import Path

import corpus
from label_docs import label_doc
from label_pairs import label_pair, load_pairwise_aspects


def _label_all(what, items, fn):
    """Labels each item, skipping individual failures (one bad judge reply shouldn't
    lose the batch) but raising if every item failed -- that is a setup problem."""
    records = []
    for item in items:
        print(f"labeling {what} {item}...")
        try:
            records.append(fn(item))
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIPPED ({exc})")
    if items and not records:
        raise SystemExit(f"every {what} labeling call failed -- check `claude` auth and the judge model id")
    return records


def _write(path, records):
    corpus.write_jsonl(path, records)
    print(f"wrote {len(records)} record(s) -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=corpus.DEFAULT_DATASET_DIR)
    parser.add_argument("--double-label", action="store_true",
                        help="also label a sample with a second judge model for inter-rater QC")
    parser.add_argument("--double-label-sample", type=int, default=5)
    parser.add_argument("--double-label-from", default="test", choices=corpus.SPLIT_NAMES)
    parser.add_argument("--double-label-model", default="claude-opus-5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir, labels_dir = args.out_dir, args.out_dir / "labels"
    cv_bodies = {cv_id: (out_dir / cv_id).read_text(encoding="utf-8") for cv_id in corpus.list_cvs(out_dir)}
    if not cv_bodies:
        raise SystemExit(f"no CVs found under {out_dir / 'cvs'}")
    job_docs = {
        job_id: corpus.parse_frontmatter((out_dir / job_id).read_text(encoding="utf-8"))
        for job_ids in corpus.list_jobs(out_dir).values() for job_id in job_ids
    }
    if not job_docs:
        raise SystemExit(f"no jobs found under {out_dir / 'jobs'}")
    splits = corpus.load_splits(out_dir)
    pairs = list({(p["cv"], p["job"]): p for split in splits.values() for p in split}.values())
    if not pairs:
        raise SystemExit(f"no pairs found under {out_dir / 'splits'} -- run build_dataset.py first")
    pairwise_aspects = load_pairwise_aspects()

    def pair_labeler(judge_model=None):
        def fn(pair):
            meta, body = job_docs[pair["job"]]
            return label_pair(pair["cv"], cv_bodies[pair["cv"]], pair["job"], body, meta,
                              pairwise_aspects, judge_model=judge_model)
        return fn

    _write(labels_dir / "cvs.jsonl",
           _label_all("CV", list(cv_bodies), lambda cv_id: label_doc("cv", cv_id, cv_bodies[cv_id])))
    _write(labels_dir / "jobs.jsonl",
           _label_all("job", list(job_docs), lambda job_id: label_doc("job", job_id, job_docs[job_id][1])))
    _write(labels_dir / "pairs.jsonl", _label_all("pair", pairs, pair_labeler()))

    if not args.double_label:
        return
    split_pairs = splits[args.double_label_from]
    if not split_pairs:
        print(f"WARNING: {args.double_label_from!r} split is empty, nothing to double-label")
        return
    model = args.double_label_model
    sample = random.Random(args.seed).sample(split_pairs, min(args.double_label_sample, len(split_pairs)))
    _write(labels_dir / "pairs_double.jsonl", _label_all("pair", sample, pair_labeler(model)))
    # The sampled pairs' own CVs/jobs, so the doc-level QC covers the same documents.
    _write(labels_dir / "cvs_double.jsonl", _label_all(
        "CV", sorted({p["cv"] for p in sample}),
        lambda cv_id: label_doc("cv", cv_id, cv_bodies[cv_id], judge_model=model)))
    _write(labels_dir / "jobs_double.jsonl", _label_all(
        "job", sorted({p["job"] for p in sample}),
        lambda job_id: label_doc("job", job_id, job_docs[job_id][1], judge_model=model)))


if __name__ == "__main__":
    main()
