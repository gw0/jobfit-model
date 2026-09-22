# `data/` -- offline dataset-build toolset

One-time scripts (specs §5) that produce the committed corpus `datasets/`
(`datasets_smoke/` at smoke scale). None of this runs in the Argo pipeline;
`pipeline/prepare.py` only reads the result.

## Building a full-scale corpus

`datasets_smoke/` (10 CVs, 10 jobs, 58 pairs) exercises the pipeline logic but is too
small for meaningful quality or calibration numbers. The full corpus (~100 CVs,
~500 jobs, ~4k pairs, specs §5) comes from the same scripts pointed at `datasets/`:

```sh
./data/fetch_jobs/fetch_all.py --out-dir datasets/jobs       # or fetch_<ats>.py per board
./data/generate_cvs.py --count 100 --out-dir datasets         # needs an authenticated `claude`
./data/pii_scrub.py datasets/cvs
./data/build_dataset.py --out-dir datasets                    # 70/10/10/10 needs this scale
./data/label_dataset.py --out-dir datasets --double-label     # one call does both passes
```

The labeling step makes one `claude -p` judge call per pair, so its token cost and
wall-clock time scale with the corpus: the 58-pair smoke corpus takes minutes, ~4k pairs
takes orders of magnitude longer. `datasets/` is committed once built: it is synthetic
(specs §10), and losing it would mean paying for labeling again.

## Job-post redistribution: ToS review (specs §12, gate before going public)

The curated `jobs/` corpus (`datasets_smoke/jobs/`, and any future `datasets/jobs/`)
consists of individual public job-posting pages fetched from three third-party
applicant-tracking systems: Greenhouse (`job-boards.greenhouse.io`), Lever
(`jobs.lever.co`), and Ashby (`jobs.ashbyhq.com`). Findings from a review done ahead
of this gate (2026-09-23, not a substitute for real legal counsel):

- **robots.txt**: none of the three disallow the job-listing paths this project
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
staying committed. Before any future large-scale (~500 post) `datasets/jobs/` corpus
or commercial use, get a real legal review per-ATS rather than relying on this
engineering-level pass -- this review is deliberately scoped to what could be checked
via public robots.txt/ToS pages in the time available, not a certified legal opinion.
