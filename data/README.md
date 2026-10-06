# `data/` -- offline dataset-build toolset

One-time scripts (specs §5) that produce the committed corpora `datasets_smoke/` and
`datasets_full/`. None of this runs in the Argo pipeline;
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
once, so an interrupted step resumes by re-running it after the reset. `datasets_full/` is committed once built:
it is synthetic apart from the job posts (specs §10), and losing it would mean paying
for labeling again.

## Job-post redistribution: ToS review (specs §12, gate before going public)

The curated `jobs/` corpus (`datasets_smoke/jobs/`, and `datasets_full/jobs/`)
consists of individual public job-posting pages fetched from third-party
applicant-tracking systems: Greenhouse (`job-boards.greenhouse.io`), Lever
(`jobs.lever.co`), Ashby (`jobs.ashbyhq.com`), and four Workday tenants (not covered by
this review). `discover_*.py` and `fetch_generic.py` are standalone fetchers for job
boards and arbitrary careers pages, run by hand and not part of `fetch_all.py`. Findings from a review done ahead
of this gate (2026-09-23, not a substitute for real legal counsel):

- **robots.txt**: none of the three reviewed disallow the job-listing paths this project
  fetches from (`/<org>/jobs/*` on Greenhouse boards, `/<org>/*` on Lever, the
  individual job pages on Ashby -- Ashby's robots.txt only disallows `/meeting/`,
  `/b/`, `/api/`, none of which this project touches).
- **Terms of service**: Lever's public ToS contains no clause on scraping,
  crawling, or redistribution of job postings by third parties -- its restrictions
  target the ATS's paying customers (the employers), not readers of the public
  listing pages. Greenhouse's terms-and-policies page did not yield accessible text
  covering this specific question in this review; no explicit anti-redistribution
  clause for individual public job listings was found in what was reachable. Ashby's
  ToS was not separately reviewed.
- Every fetched post already carries its `url`, `company`, and `posted_at` in
  frontmatter (written by `data/fetch_jobs/jobboard.py`) -- attribution back to the
  original listing travels with the data by construction, not as an afterthought.
- `jobboard.py`'s `USER_AGENT` is a descriptive, non-spoofed string (specs §12) --
  these are polite, identified fetches of publicly reachable pages, not disguised as a
  browser.

**Decision**: proceed with the current curated `datasets_smoke/jobs/` corpus (10
posts, non-commercial ML research/demo use, small scale, full attribution retained)
staying committed. Before publishing the large-scale (~400 post) `datasets_full/jobs/`
corpus or any commercial use, get a real legal review per-ATS rather than relying on this
engineering-level pass -- this review is deliberately scoped to what could be checked
via public robots.txt/ToS pages in the time available, not a certified legal opinion.
