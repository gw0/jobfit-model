#!/usr/bin/env python3
"""Fetch job posts from a Workday (CXS) careers site.

Usage: ./fetch_workday.py <careers-url> [--include ...] [--exclude ...] [--location ...]

<careers-url> is the tenant's myworkdayjobs.com careers site, e.g.
https://bostondynamics.wd1.myworkdayjobs.com/Boston_Dynamics
"""

from urllib.parse import urlparse

import jobboard

# Workday's CXS API rejects any limit above 20 (confirmed live across tenants),
# so the full list needs an offset-paginated loop rather than one "generous limit" call.
PAGE_SIZE = 20
MAX_PAGES = 50  # safety cap: 1000 postings, far beyond any real career site


def _parse_target(target):
    parsed = urlparse(target)
    tenant, wd_host = parsed.hostname.split(".")[:2]
    site = parsed.path.strip("/").split("/")[0]
    return tenant, wd_host, site


def _list_postings(base):
    listings = []
    offset = 0
    for _ in range(MAX_PAGES):
        resp = jobboard.request(
            "POST",
            f"{base}/jobs",
            json={"appliedFacets": {}, "limit": PAGE_SIZE, "offset": offset, "searchText": ""},
        )
        page = resp.json().get("jobPostings", [])
        listings.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return listings


def fetch(target, include=None, exclude=None):
    tenant, wd_host, site = _parse_target(target)
    base = f"https://{tenant}.{wd_host}.myworkdayjobs.com/wday/cxs/{tenant}/{site}"

    listings = _list_postings(base)

    candidates = [
        posting for posting in listings
        if jobboard.matches({"title": posting.get("title")}, include=include, exclude=exclude)
    ]

    posts = []
    for posting in candidates:
        external_path = posting.get("externalPath")
        if not external_path:
            continue
        detail_resp = jobboard.request(
            "GET", f"{base}{external_path}", headers={"Accept": "application/json"}
        )
        info = detail_resp.json().get("jobPostingInfo", {})
        posts.append({
            "title": info.get("title") or posting.get("title"),
            "company": tenant,
            "location": info.get("location") or posting.get("locationsText"),
            "url": info.get("externalUrl") or f"https://{tenant}.{wd_host}.myworkdayjobs.com/{site}{external_path}",
            "posted_at": info.get("startDate"),
            "description": jobboard.to_markdown(info.get("jobDescription") or ""),
        })
    return posts


if __name__ == "__main__":
    args = jobboard.parse_cli(target=True)
    posts = fetch(args.target, include=args.include, exclude=args.exclude)
    jobboard.save_posts(posts, location=args.location, out_dir=args.out_dir)
