# `data/` -- offline dataset-build toolset

One-time scripts (specs §5) that produce the corpora `datasets_smoke/` (committed here) and
`datasets_full/` (a symlink to the sibling checkout
[gw0/jobfit-jevbench](https://github.com/gw0/jobfit-jevbench)). None of this runs in the Argo pipeline;
`pipeline/prepare.py` only reads the result.

## Building a full-scale corpus

`datasets_smoke/` (10 CVs, 10 jobs, 58 pairs, 4 of them in `test`) exercises the
pipeline logic but is too small for meaningful quality or calibration numbers.
`datasets_full/` comes from the same scripts at `SCALE=full` (Makefile):

```sh
make jobs SCALE=full      # fetch_all.py: ~400 posts, at most 10 per company
make cvs SCALE=full       # generate_cvs.py: 250 CVs over a seeded role/seniority/location/style grid
make dataset SCALE=full   # build_dataset.py: 20 jobs per CV, half nearest by TF-IDF, half random
make labels SCALE=full    # label_dataset.py: judge every doc and pair, double-label 300 test pairs
```

| | CVs | jobs | pairs (train / val / calib / test) |
|---|---|---|---|
| smoke | 10 | 10 | 58 (42 / 6 / 6 / 4), every CV x every job of its pool |
| full | 250 | ~400 | 5,000 (3,500 / 500 / 500 / 500) |

Sampling pairs instead of taking every combination keeps labeling affordable, and
taking half of each CV's jobs from its nearest matches keeps the fit labels spread over
the whole scale instead of piling up near "no fit". The ~25 distinct `test` CVs, rather
than ~1, are what make the bootstrap CIs over CV groups (specs §5) usable.

Labeling makes one `claude -p` judge call per document and one per `--jobs-per-call`
(10) pairs of a CV, ~1.5k calls at full scale: Sonnet for the bulk pass and Opus for the
300-pair double-label QC (one pair per call).
Document labels also run the local AI-text detector (`ai_text_detector.py`), and
`generate_cvs.py` scans every CV with `pii_scrub.py`. Every step keeps what already exists and the labeler
appends each record as its call returns; on the account's usage limit it stops at
once, so an interrupted step resumes by re-running it after the reset. `datasets_full/` is committed to
`jobfit-jevbench` once built: it is synthetic apart from the job posts (specs §10), and
losing it would mean paying for labeling again.

## Job-post redistribution: ToS review (specs §12, gate before going public)

The job posts are individual public pages fetched from Greenhouse, Lever, Ashby and four
Workday tenants. Every post keeps its `url`, `company` and `posted_at` in frontmatter, and
`jobboard.py`'s `USER_AGENT` is a descriptive, non-spoofed string (specs §12). The per-source
robots.txt and ToS review (engineering-level, not a legal opinion) lives with the corpus, in
the [jobfit-jevbench README](https://github.com/gw0/jobfit-jevbench#job-post-sources).
`discover_*.py` and `fetch_generic.py` are standalone fetchers run by hand, not part of
`fetch_all.py`.
