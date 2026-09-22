#!/usr/bin/env python3
"""Fetch job posts from a Lever postings board.

Usage: ./fetch_lever.py <org-slug-or-careers-url> [--include ...] [--exclude ...] [--location ...]
"""

from datetime import datetime, timezone

import jobboard


def _full_description_html(posting):
    parts = [posting.get("description") or ""]
    for section in posting.get("lists") or []:
        heading = section.get("text")
        content = section.get("content") or ""
        if heading:
            parts.append(f"<h3>{heading}</h3>")
        parts.append(content)
    if posting.get("additional"):
        parts.append(posting["additional"])
    return "\n".join(p for p in parts if p)


def fetch(target, include=None, exclude=None):
    target = jobboard.extract_slug(target)
    resp = jobboard.request("GET", f"https://api.lever.co/v0/postings/{target}?mode=json")
    postings = resp.json()
    posts = []
    for posting in postings:
        title = posting.get("text")
        if not jobboard.matches({"title": title}, include=include, exclude=exclude):
            continue
        created_at = posting.get("createdAt")
        posted_at = None
        if created_at:
            posted_at = datetime.fromtimestamp(created_at / 1000, tz=timezone.utc).date().isoformat()
        posts.append({
            "title": title,
            "company": target,
            "location": (posting.get("categories") or {}).get("location"),
            "url": posting.get("hostedUrl"),
            "posted_at": posted_at,
            "description": jobboard.to_markdown(_full_description_html(posting)),
        })
    return posts


if __name__ == "__main__":
    args = jobboard.parse_cli(target=True)
    posts = fetch(args.target, include=args.include, exclude=args.exclude)
    jobboard.save_posts(posts, location=args.location, out_dir=args.out_dir)
