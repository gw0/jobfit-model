# Job post fetcher — design spec

Lives in `fetch_jobs/` (renamed from `fetcher/`, then briefly `job-fetcher/`, on 2026-09-21/22). Captures the agreed requirements for a set of small scripts that fetch **job posts** from company careers pages and a few aggregator job boards, keep only technical/software-engineering roles, and save each as a Markdown file. §2 below reflects the as-built architecture, which diverged from the original draft in a few places (called out inline) after a code-review pass on 2026-09-21.

## 1. Terminology

The items being fetched are called **job posts** throughout — in code, filenames, and this doc. Chosen because it matches the vocabulary ATS platforms themselves use in their APIs (Greenhouse/Lever/Ashby/Workday all call these "postings").

## 2. Architecture

With ~9 source-specific scripts in play (4 ATS + generic + 3 aggregators), the biggest real risk to "simple and maintainable" isn't any single script — it's the same boilerplate (HTTP retry/backoff, CLI parsing, markdown writing, dedup) getting copy-pasted into all nine and drifting out of sync. So the design pushes everything *not specific to one source* into one small shared library, and keeps every source-specific file down to "call the API, normalize the response":

- **`common.py`** — shared plumbing, written once. Everything here is source-agnostic; nothing specific to one provider or to orchestration lives here:
  - `request(method, url, **kwargs)` — HTTP session with the standard User-Agent, an unconditional 1s sleep before every call, and retry-with-backoff on connection errors / 429 / 5xx. Since every script only ever talks to one host (or, for Workday, one tenant host) at a time in a sequential loop, this unconditional sleep *is* the ≤1 req/sec-per-provider guarantee — no per-host timestamp tracking needed.
  - `extract_slug(target)` — if `target` is a URL, returns its first path segment (the org/board slug); otherwise returns it unchanged. Lets Greenhouse/Lever/Ashby `fetch()` accept either a bare slug or a full careers URL pasted straight from `careers-ats-list.md`.
  - `to_markdown(html_content)` — `html.unescape` then `markdownify`, or `""` if empty. Called by every `fetch_*`/`discover_*` function to build `post["description"]` (already Markdown by the time it reaches `write_post`).
  - `write_post(post)` — given one normalized post dict (with an already-Markdown `description` field), computes the filename (date + title/location slug), scans the company folder for a matching `url` to dedupe, and writes/overwrites the Markdown file.
  - `matches(post, include=None, exclude=None, location=None, sector=None)` — one merged filter predicate (see §4) plus its backing data: `DEFAULT_INCLUDE`, `DEFAULT_EXCLUDE`, `LOCATION_PATTERNS`. `None` on any dimension skips that check rather than falling back to a default — default resolution happens one level up, in `parse_keywords`.
  - `build_cli(target=False, sector=False)` / `parse_keywords(args)` / `save_posts(posts, location=None)` — CLI argument parsing, `--include`/`--exclude` default resolution, and the write-with-location-filter-and-summarize step. Deliberately *not* a single `run_cli(fetch_fn, mode)` wrapper (as originally drafted): once every `fetch_*`/`discover_*` function shares one signature, routing calls through a mode flag only adds indirection, so each script's `__main__` block calls its own function directly and then `common.save_posts(...)`.
- Originally drafted as separate `filters.py` and `companies.py` modules; in practice the filter logic stayed in `common.py` (it's genuinely source-agnostic shared plumbing), while the company list moved to `fetch_all.py` (see below) since nothing but that one file ever reads it.
- **`fetch_all.py`** — the orchestrator *and* the company data together: `COMPANIES`, a list of `{name, provider, fetch, target}` dicts where `fetch` holds the provider's fetch function directly (no `{provider_name: fetch_fn}` dispatch dict — `provider` is kept only as a human-readable label for the progress print). Loops over `COMPANIES`, calls `company["fetch"](company["target"], include=include, exclude=exclude)`, hands the result to `common.save_posts`. No provider-specific logic here.
- **One `fetch_<provider>.py` per known-company source format** — `fetch_greenhouse.py`, `fetch_lever.py`, `fetch_ashby.py`, `fetch_workday.py`, `fetch_generic.py` (JSON-LD `JobPosting` first, heuristic HTML link-scan second, for anything without a known structured API — always attempts structured extraction, never saves an unstructured page dump). Each is intentionally small: one `fetch(target, include=None, exclude=None) -> list[dict]` function, shared signature across all five, that calls the API/page, applies `common.matches(...)` per item before doing the (comparatively expensive) Markdown conversion, and returns a list of posts already normalized to the common shape (`title, company, location, url, posted_at, description`) — no CLI code, no writing code, no retry code duplicated here.
- **One `discover_<source>.py` per aggregator** — `discover_remoteok.py`, `discover_hn_whoishiring.py`, `discover_weworkremotely.py`. Named with a different verb, and a different function name, on purpose: a `fetch_*` script's `fetch()` pulls posts from one already-known company; a `discover_*` script's `discover(include=None, exclude=None, sector=None)` searches a marketplace of many companies' postings by keyword/tag to surface roles from companies *not* in `COMPANIES`. Same shape internally (one function, returns normalized posts, no shared plumbing duplicated).
- Every `fetch_<provider>.py` and `discover_<source>.py` is independently runnable (`python fetch_greenhouse.py <org>`, `python discover_remoteok.py --include ...`) via its own explicit `__main__` block (`build_cli` → `parse_keywords` → call the function → `save_posts`), and writes the same `.md` output as when called from `fetch_all.py` — one code path, no separate "debug mode."

Net effect: adding a 10th source later means writing one small normalize function, not touching CLI/writing/retry logic anywhere. Removing or reworking a source touches exactly one file.

## 3. Sources

- **Structured per-company ATS** (reliable, one request per company or a small fixed number): Greenhouse, Lever, Ashby, Workday.
- **Generic fallback** for any other company careers page, in order: (1) look for embedded `schema.org` `JobPosting` JSON-LD (many pages, even JS-rendered ones, embed this for SEO); (2) if absent, heuristically scan the HTML for repeated anchor/list patterns that look like job listings. If neither yields structured data, skip the company and say so on stdout.
- **Aggregators** (via `discover_*`), limited to sources with public, ToS-permitting programmatic access: RemoteOK (public JSON API), Hacker News "Who is Hiring" threads (public Algolia API), WeWorkRemotely (RSS). Scraping sites that prohibit it (LinkedIn, Indeed, Wellfound/AngelList) is explicitly out of scope.

## 4. Filtering

### Technical role filter (always applied, fully configurable)

Keyword include/exclude lists covering software engineering broadly, shipped as defaults in `common.py` but **overridable at the CLI**:

- `--include` — replaces the default include list for this run (comma-separated).
- `--exclude` — replaces the default exclude list for this run.
- With neither flag given, the built-in defaults below apply.

**Default include** (title contains, case-insensitive): `engineer, engineering, developer, programmer, sre, site reliability, devops, mlops, ml ops, platform, infrastructure, backend, frontend, front-end, back-end, full stack, full-stack, mobile, ios, android, security researcher, security engineer, appsec, penetration tester, pentester, data engineer, database engineer, machine learning, ml engineer, ai engineer, applied scientist, research engineer, ml researcher, qa engineer, sdet, test engineer, automation engineer, release engineer, build engineer, systems engineer, network engineer, cloud engineer, compiler, kernel, firmware, embedded, architect, staff engineer, principal engineer, distinguished engineer`

**Default exclude** (checked first, overrides an include match): `sales engineer, solutions engineer, support engineer, customer engineer, implementation engineer, field engineer, technical writer, technical recruiter, technical account manager, technical program manager, technical product manager`

Deliberately left **out** of the default include list, one `--include` flag away from being added:
- Engineering Manager / Director / VP Engineering (people management, not hands-on)
- Data Scientist (often analytics-leaning rather than engineering)
- Hardware Engineer / Robotics Engineer (physical hardware, not software)
- Developer Advocate / DevRel (technical but not building the product)

### Location filter (optional)

`--location remote-us` / `--location remote-eu`: simple case-insensitive substring match against the post's location text. No geocoding or normalization.

- US patterns: `remote - us`, `us remote`, `united states`, `usa`
- EU patterns: `remote - eu`, `eu remote`, `emea`, `europe`

### Sector/industry filter (optional, aggregator-only)

Supported **only where the source natively provides tag/category data**, which in practice means `discover_*` scripts: RemoteOK and WeWorkRemotely both expose tags/categories per listing that can be matched against `--sector` (e.g. `--sector ai`, `--sector fintech`). Per-company ATS sources (`fetch_*`) have no industry field in their data model, so `--sector` is accepted but a no-op there — we deliberately don't maintain our own company→sector mapping, to avoid a second data source to keep in sync.

### Efficiency principle

Every `fetch_*`/`discover_*` function applies `common.matches(item, include=..., exclude=...)` (and, where applicable, `sector=...`) **per item, before normalization** — role filtering is a first-class parameter on every one of these functions, not a step applied afterward by the caller. Two motivations, not just one:
- Fetchers/discoverers use server-side filtering/query params when the source supports it, instead of pulling everything and filtering client-side: `discover_*` scripts build their search query from `--include`/`--sector` where the aggregator supports keyword/tag search server-side (RemoteOK's `?tags=`).
- Per-company ATS APIs (Greenhouse/Lever/Ashby) don't support server-side title filtering, so `fetch_*` for those still pulls a company's full post list — but filtering *before* normalization still matters there: it skips the Markdown conversion (`common.to_markdown`) for every non-matching post, which is real work even when it doesn't save an HTTP call.
- Workday's list+detail split means the title filter is applied to the list response *before* the per-post detail request, so non-matching posts never trigger the extra call — this used to be documented as the one exception; now it's simply consistent with every other source.

Location filtering stays a thin, separate post-step in `common.save_posts` (applied after role-matched posts come back from `fetch`/`discover`), since no source offers it server-side and it doesn't gate any expensive call.

## 5. CLI

- `fetch_all.py [--include ...] [--exclude ...] [--location remote-us|remote-eu]`
- Each `fetch_<provider>.py <url-or-org> [--include ...] [--exclude ...] [--location ...]` for standalone runs.
- Each `discover_<source>.py [--include ...] [--exclude ...] [--location ...] [--sector ...]` for standalone runs.

Flag parsing (`common.build_cli`) and default-keyword resolution (`common.parse_keywords`) are shared helpers, not re-implemented per script; each script's own `__main__` block calls them directly rather than going through a mode-flag wrapper (see §2).

## 6. Output

### File path and naming

`jobs/<company-slug>/<post-date>-<job-title-slug>.md`

- **`<post-date>`** is `YYYYMMDD`, taken from the job post's own posting/update date **as reported by the source**, not the date the fetcher ran — so re-running later doesn't shift a post's filename. Per-provider field: Greenhouse `updated_at`, Lever `createdAt`, Ashby `publishedAt`, Workday `postedOn`, generic/JSON-LD `datePosted`. If a source gives no date at all, fall back to the fetch date (best available signal; not flagged specially, to keep frontmatter minimal — see below).
- On a same-day-and-title collision (e.g. the same role open in multiple locations), the **location slug** is appended: `<post-date>-<job-title-slug>-<location-slug>.md`. In the rare case that also collides (identical date, title, *and* location text), the later fetch overwrites the earlier file — accepted as a simple, extremely unlikely edge case rather than adding a further tiebreaker.

### Frontmatter — kept minimal

```yaml
---
company: Anthropic
title: Senior Backend Engineer
location: San Francisco, CA
url: https://job-boards.greenhouse.io/anthropic/jobs/12345
posted_at: 2026-09-15
---
```

Just enough to identify, dedupe, and link back to the post. No `source`/provider, no `fetched_at`, no fallback-flag field — anything else needed can be read from the body or re-derived. The body is `post["description"]` — the post description, already converted to Markdown by the `fetch_*`/`discover_*` function itself (via `common.to_markdown`) before it ever reaches `write_post`, rather than converted at write time.

### Deduplication via source URL (no state file)

Every post's own source `url` is unique and stable across runs. Instead of maintaining a separate state/index file, `common.write_post` **dedupes by scanning the frontmatter of existing `.md` files already in that company's folder** for a matching `url` before writing:

- If a file with the same `url` already exists, **overwrite that exact file** (same filename, refreshed content/frontmatter) rather than creating a new one — this also naturally handles a post whose title or location text changed slightly between runs, which would otherwise slip past the filename-based collision check above.
- If no match is found, it's a new post: write it to `<post-date>-<title-slug>[-<location-slug>].md` as described above.

This keeps "the filesystem is the source of truth" intact — no separate index to fall out of sync — while still giving real dedup instead of relying on filename coincidence. Cost is one small directory scan (read frontmatter of existing files) per company per run, which is cheap at this scale (tens of posts per company at most).

No report file — progress and a per-source summary print to stdout only.

## 7. Resilience & politeness

- Every source call runs inside the shared wrapper in `common.py`, so one failure (bad JSON, changed schema, HTTP error, no posts found) is logged to stdout and the run continues to the next source — this behavior lives once, not per script. In `fetch_all.py` specifically, each company's `fetch` call is wrapped in its own try/except so one dead board (confirmed live: two Greenhouse boards that 404) doesn't stop the run.
- An unconditional 1s sleep before every request, plus retry-with-backoff (a few attempts) on connection errors / 429 / 5xx, with a normal browser-like User-Agent — all in `common.request`.
- Each `fetch_<provider>`/`discover_<source>` normalize function reads fields defensively (`.get()` with fallback) so a minor upstream schema tweak degrades one field rather than crashing the whole run.

## 8. Setup

Everything installs into a standard Python venv — no other tooling required:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`: `requests`, `beautifulsoup4`, `markdownify`, `feedparser` (for the WeWorkRemotely RSS discoverer).

## 9. Explicit non-goals (v1)

- No JS rendering / headless browser for JS-only career pages.
- No scraping of sources whose ToS prohibits it.
- No cleanup/archival of posts that disappear from a source.
- No company→sector mapping for `fetch_*` sources — sector filtering only works where a `discover_*` source natively provides it.
- No plugin system / class hierarchy for sources — plain functions, each `COMPANIES` entry in `fetch_all.py` holding its fetch function directly; not worth any further indirection at ~9 sources.

## 10. Open questions for future refinement

- Exact keyword list wording — this doc ships a first draft, expected to be tuned after seeing real output.
- How to dedupe a post that's discoverable both via a company's own ATS *and* via an aggregator listing the same role (different source URLs for the same underlying job — the URL-based dedup above won't catch this).
