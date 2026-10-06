"""Shared plumbing for the job-board fetchers: polite HTTP, title/location filters,
writing posts into the corpus, and CLI wiring. File naming and the frontmatter format
themselves live in data/corpus.py."""

import argparse
import html
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from markdownify import markdownify

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data"))
from corpus import format_frontmatter, parse_frontmatter, slugify  # noqa: E402

FRONTMATTER_KEYS = ("company", "title", "location", "url", "posted_at")

EMAIL_RE = re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)+")
REDACTED_EMAIL = "redacted@example.com"

USER_AGENT = "JobFit/0.1 (+https://github.com/gw0/jobfit-model)"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


def request(method, url, **kwargs):
    """requests wrapper with an unconditional 1s pre-call sleep and retry-with-backoff.

    Every script only ever talks to one host at a time in a sequential loop, so
    sleeping 1s before each call already guarantees <=1 req/sec to whatever host
    this call hits -- no per-host bookkeeping needed.

    Retries on connection errors, timeouts, 429, and 5xx. All other HTTP
    errors raise immediately via raise_for_status().
    """
    kwargs.setdefault("timeout", 30)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        time.sleep(1.0)
        try:
            resp = session.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp


def extract_slug(target):
    """If target is a URL, return its first path segment (the org/board slug); else return it unchanged."""
    if target.startswith("http://") or target.startswith("https://"):
        path = urlparse(target).path.strip("/")
        return unquote(path.split("/")[0]) if path else target
    return target


def to_markdown(html_content):
    if not html_content:
        return ""
    return markdownify(html.unescape(html_content)).strip()


# ---------------------------------------------------------------------------
# Writing posts to Markdown
# ---------------------------------------------------------------------------

def _find_existing(folder, url):
    if not url:
        return None
    for md_path in folder.glob("*.md"):
        if parse_frontmatter(md_path.read_text(encoding="utf-8"))[0].get("url") == url:
            return md_path
    return None


def _date_slug(posted_at):
    """YYYYMMDD from a 'YYYY-MM-DD' posted_at, or today's date if absent/invalid."""
    if posted_at and len(posted_at) >= 10 and posted_at[4] == "-" and posted_at[7] == "-":
        return posted_at[:10].replace("-", "")
    return date.today().strftime("%Y%m%d")


def redact_emails(text):
    """Replace every email address with the fake one pii_scrub.py whitelists."""
    return EMAIL_RE.sub(REDACTED_EMAIL, text)


def write_post(post, jobs_root):
    """Write one normalized post dict to jobs/<company-slug>/, deduping by url.

    post: {title, company, location, url, posted_at, description}
    Returns (path, is_new).
    """
    folder = jobs_root / slugify(post.get("company"))
    folder.mkdir(parents=True, exist_ok=True)

    url = post.get("url") or ""
    existing = _find_existing(folder, url)

    title_slug = (slugify(post.get("title")) or "untitled")[:80].rstrip("-")
    date_part = _date_slug(post.get("posted_at"))

    if existing is not None:
        path = existing
        is_new = False
    else:
        path = folder / f"{date_part}-{title_slug}.md"
        if path.exists():
            location_slug = slugify(post.get("location"))
            if location_slug:
                path = folder / f"{date_part}-{title_slug}-{location_slug}.md"
        is_new = True

    meta = {key: post.get(key) for key in FRONTMATTER_KEYS}
    description = redact_emails(post.get("description", ""))
    path.write_text(format_frontmatter(meta) + "\n\n" + description + "\n", encoding="utf-8")
    return path, is_new


# ---------------------------------------------------------------------------
# Filters (pure functions, no I/O)
# ---------------------------------------------------------------------------

DEFAULT_INCLUDE = [
    "engineer", "engineering", "developer", "programmer", "sre", "site reliability",
    "devops", "mlops", "ml ops", "platform", "infrastructure", "backend", "frontend",
    "front-end", "back-end", "full stack", "full-stack", "mobile", "ios", "android",
    "security researcher", "security engineer", "appsec", "penetration tester", "pentester",
    "data engineer", "database engineer", "machine learning", "ml engineer", "ai engineer",
    "applied scientist", "research engineer", "ml researcher", "qa engineer", "sdet",
    "test engineer", "automation engineer", "release engineer", "build engineer",
    "systems engineer", "network engineer", "cloud engineer", "compiler", "kernel",
    "firmware", "embedded", "architect", "staff engineer", "principal engineer",
    "distinguished engineer",
]

DEFAULT_EXCLUDE = [
    "sales engineer", "solutions engineer", "support engineer", "customer engineer",
    "implementation engineer", "field engineer", "technical writer", "technical recruiter",
    "technical account manager", "technical program manager", "technical product manager",
]

LOCATION_PATTERNS = {
    "remote-us": ["remote - us", "us remote", "united states", "usa"],
    "remote-eu": ["remote - eu", "eu remote", "emea", "europe"],
}


def matches(post, include=None, exclude=None, location=None, sector=None):
    """True if post passes every filter dimension given. None on a dimension skips that check."""
    if include is not None or exclude is not None:
        title = (post.get("title") or "").lower()
        if exclude and any(kw in title for kw in exclude):
            return False
        if include and not any(kw in title for kw in include):
            return False
    if location:
        patterns = LOCATION_PATTERNS.get(location, [])
        loc_text = (post.get("location") or "").lower()
        if not any(p in loc_text for p in patterns):
            return False
    if sector:
        tags = post.get("tags")
        if tags and not any(sector.lower() in str(tag).lower() for tag in tags):
            return False
    return True


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def _csv_list(default):
    """Builds an argparse `type=` callable: comma-split -> lower -> filter blanks,
    falling back to `default` if that yields nothing (an omitted flag never reaches
    this -- argparse uses `default` directly in that case)."""
    def parse(value):
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
        return items or default
    return parse


def parse_cli(*, target=False, sector=False, count=False):
    """Builds and parses this script's CLI -- every fetch_*.py/discover_*.py script
    needs the same --include/--exclude/--location/--out-dir flags, plus a positional
    `target` or a --sector flag depending on which kind, or --count for fetch_all.py."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", type=_csv_list(DEFAULT_INCLUDE), default=DEFAULT_INCLUDE,
                         help="comma-separated list of title keywords to include")
    parser.add_argument("--exclude", type=_csv_list(DEFAULT_EXCLUDE), default=DEFAULT_EXCLUDE,
                         help="comma-separated list of title keywords to exclude")
    parser.add_argument("--location", choices=list(LOCATION_PATTERNS.keys()), default=None)
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="directory to write jobs/<company>/<file>.md into (datasets_<scale>/jobs)",
    )
    if target:
        parser.add_argument("target", help="org/board slug (or careers URL for workday)")
    if sector:
        parser.add_argument("--sector", default=None, help="tag/category keyword, e.g. ai, fintech")
    if count:
        parser.add_argument("--count", type=int, default=None,
                             help="target total job postings across all companies; stop once reached")
        parser.add_argument("--max-per-company", type=int, default=None,
                             help="write at most this many postings per company")
    return parser.parse_args()


def save_posts(posts, out_dir, location=None, limit=None):
    """Apply location filtering, write up to `limit` matching posts, and print a
    one-line summary. Returns the number of new posts.

    posts arriving here are already role-matched (fetch/discover apply include/exclude
    themselves), so this only checks location before writing.
    """
    matched = new = 0
    for post in posts:
        if not matches(post, location=location):
            continue
        if matched == limit:
            break
        matched += 1
        new += write_post(post, jobs_root=out_dir)[1]

    print(f"matched {matched}, written {new} new / {matched - new} updated")
    return new
