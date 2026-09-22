#!/usr/bin/env python3
"""Fetch job posts from a Greenhouse job board.

Usage: ./fetch_greenhouse.py <org-slug-or-careers-url> [--include ...] [--exclude ...] [--location ...]
"""

import jobboard


def fetch(target, include=None, exclude=None):
    target = jobboard.extract_slug(target)
    resp = jobboard.request(
        "GET", f"https://boards-api.greenhouse.io/v1/boards/{target}/jobs?content=true"
    )
    data = resp.json()
    posts = []
    for job in data.get("jobs", []):
        title = job.get("title")
        if not jobboard.matches({"title": title}, include=include, exclude=exclude):
            continue
        posts.append({
            "title": title,
            "company": job.get("company_name") or target,
            "location": (job.get("location") or {}).get("name"),
            "url": job.get("absolute_url"),
            "posted_at": (job.get("updated_at") or "")[:10] or None,
            "description": jobboard.to_markdown(job.get("content") or ""),
        })
    return posts


if __name__ == "__main__":
    args = jobboard.parse_cli(target=True)
    posts = fetch(args.target, include=args.include, exclude=args.exclude)
    jobboard.save_posts(posts, location=args.location, out_dir=args.out_dir)
