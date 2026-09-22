#!/usr/bin/env python3
"""Discover job posts from the WeWorkRemotely combined RSS feed.

Usage: ./discover_weworkremotely.py [--include ...] [--exclude ...] [--location ...] [--sector ...]
"""

import time

import feedparser

import jobboard

FEED_URL = "https://weworkremotely.com/remote-jobs.rss"


def _location_text(entry):
    parts = [entry.get("region"), entry.get("country"), entry.get("state")]
    return ", ".join(p for p in parts if p)


def discover(include=None, exclude=None, sector=None):
    resp = jobboard.request("GET", FEED_URL)
    feed = feedparser.parse(resp.content)

    posts = []
    for entry in feed.entries:
        title = entry.get("title") or ""
        if ": " in title:
            company, role = title.split(": ", 1)
        else:
            company, role = "WeWorkRemotely", title

        tags = [tag.get("term") for tag in entry.get("tags", []) if tag.get("term")]

        if not jobboard.matches({"title": role, "tags": tags}, include=include, exclude=exclude, sector=sector):
            continue

        posted_at = None
        if entry.get("published_parsed"):
            posted_at = time.strftime("%Y-%m-%d", entry.published_parsed)

        posts.append({
            "title": role,
            "company": company,
            "location": _location_text(entry),
            "url": entry.get("link"),
            "posted_at": posted_at,
            "description": jobboard.to_markdown(entry.get("summary") or ""),
            "tags": tags,
        })
    return posts


if __name__ == "__main__":
    args = jobboard.parse_cli(sector=True)
    posts = discover(include=args.include, exclude=args.exclude, sector=args.sector)
    jobboard.save_posts(posts, location=args.location, out_dir=args.out_dir)
