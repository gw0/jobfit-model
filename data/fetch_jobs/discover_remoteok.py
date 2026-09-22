#!/usr/bin/env python3
"""Discover job posts from the RemoteOK public API.

Usage: ./discover_remoteok.py [--include ...] [--exclude ...] [--location ...] [--sector ...]
"""

import jobboard


def discover(include=None, exclude=None, sector=None):
    url = "https://remoteok.com/api"
    if sector:
        url += f"?tags={sector}"
    resp = jobboard.request("GET", url)
    data = resp.json()

    posts = []
    for item in data:
        if "legal" in item:
            continue  # first element is a legal-notice placeholder, not a listing
        title = item.get("position")
        tags = item.get("tags") or []
        if not jobboard.matches({"title": title, "tags": tags}, include=include, exclude=exclude, sector=sector):
            continue
        posts.append({
            "title": title,
            "company": item.get("company"),
            "location": item.get("location"),
            "url": item.get("url"),
            "posted_at": (item.get("date") or "")[:10] or None,
            "description": jobboard.to_markdown(item.get("description") or ""),
            "tags": tags,
        })
    return posts


if __name__ == "__main__":
    args = jobboard.parse_cli(sector=True)
    posts = discover(include=args.include, exclude=args.exclude, sector=args.sector)
    jobboard.save_posts(posts, location=args.location, out_dir=args.out_dir)
