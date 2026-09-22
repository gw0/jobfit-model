#!/usr/bin/env python3
"""Discover job posts from the latest Hacker News "Who is hiring?" thread.

Usage: ./discover_hn_whoishiring.py [--include ...] [--exclude ...] [--location ...] [--sector ...]
"""

import html

from bs4 import BeautifulSoup

import jobboard

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date?tags=story,author_whoishiring"
ITEM_URL = "https://hn.algolia.com/api/v1/items/{id}"


def _latest_thread_id():
    resp = jobboard.request("GET", SEARCH_URL)
    hits = resp.json().get("hits", [])
    for hit in hits:
        if (hit.get("title") or "").startswith("Ask HN: Who is hiring?"):
            return hit.get("objectID")
    return None


def _comment_title(unescaped_text):
    soup = BeautifulSoup(unescaped_text, "html.parser")
    for line in soup.get_text("\n").split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def discover(include=None, exclude=None, sector=None):
    thread_id = _latest_thread_id()
    if not thread_id:
        return []
    resp = jobboard.request("GET", ITEM_URL.format(id=thread_id))
    thread = resp.json()
    posted_at = (thread.get("created_at") or "")[:10] or None

    posts = []
    for comment in thread.get("children") or []:
        text = comment.get("text")
        if not text:
            continue
        title = _comment_title(html.unescape(text))
        if not jobboard.matches({"title": title}, include=include, exclude=exclude):
            continue
        posts.append({
            "title": title,
            "company": "HN Who is Hiring",
            "location": None,
            "url": f"https://news.ycombinator.com/item?id={comment.get('id')}",
            "posted_at": posted_at,
            "description": jobboard.to_markdown(text),
        })
    return posts


if __name__ == "__main__":
    args = jobboard.parse_cli(sector=True)
    posts = discover(include=args.include, exclude=args.exclude, sector=args.sector)
    jobboard.save_posts(posts, location=args.location, out_dir=args.out_dir)
