#!/usr/bin/env python3
"""Generic fallback fetcher for company career pages with no known ATS.

Tries embedded schema.org JobPosting JSON-LD first; if that yields nothing,
falls back to a heuristic scan of anchor tags that look like job listings.
Returns [] (no page dump) if neither yields anything.

Usage: ./fetch_generic.py <careers-url> [--include ...] [--exclude ...] [--location ...]
"""

import json
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

import jobboard

LINK_KEYWORDS = ("job", "career", "position")


def _company_from_url(url):
    netloc = urlparse(url).netloc
    netloc = netloc.removeprefix("www.")
    name = netloc.split(".")[0]
    return name.replace("-", " ").title()


def _location_text(job_location):
    if isinstance(job_location, list):
        job_location = job_location[0] if job_location else None
    if isinstance(job_location, dict):
        address = job_location.get("address")
        if isinstance(address, dict):
            parts = [
                address.get(key)
                for key in ("addressLocality", "addressRegion", "addressCountry")
            ]
            parts = [p for p in parts if p]
            if parts:
                return ", ".join(parts)
        return job_location.get("name")
    if isinstance(job_location, str):
        return job_location
    return None


def _normalize_job_posting(node, page_url):
    org = node.get("hiringOrganization") or {}
    return {
        "title": node.get("title"),
        "company": org.get("name") or _company_from_url(page_url),
        "location": _location_text(node.get("jobLocation")),
        "url": node.get("url") or page_url,
        "posted_at": (node.get("datePosted") or "")[:10] or None,
        "description": jobboard.to_markdown(node.get("description") or ""),
    }


def _from_json_ld(soup, page_url, include, exclude):
    posts = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            nodes = data["@graph"] if "@graph" in data else [data]
        elif isinstance(data, list):
            nodes = data
        else:
            nodes = []
        for node in nodes:
            if isinstance(node, dict) and node.get("@type") == "JobPosting":
                if not jobboard.matches({"title": node.get("title")}, include=include, exclude=exclude):
                    continue
                posts.append(_normalize_job_posting(node, page_url))
    return posts


def _from_link_scan(soup, page_url, include, exclude):
    company = _company_from_url(page_url)
    posts = []
    seen_urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if not text or len(text) < 4:
            continue
        haystack = f"{href} {text}".lower()
        if not any(kw in haystack for kw in LINK_KEYWORDS):
            continue
        if not jobboard.matches({"title": text}, include=include, exclude=exclude):
            continue
        url = urljoin(page_url, href)
        if url in seen_urls or url == page_url:
            continue
        seen_urls.add(url)
        posts.append({
            "title": text,
            "company": company,
            "location": None,
            "url": url,
            "posted_at": None,
            "description": "",
        })
    return posts


def fetch(target, include=None, exclude=None):
    resp = jobboard.request("GET", target)
    soup = BeautifulSoup(resp.text, "html.parser")
    posts = _from_json_ld(soup, target, include, exclude)
    if posts:
        return posts
    return _from_link_scan(soup, target, include, exclude)


if __name__ == "__main__":
    args = jobboard.parse_cli(target=True)
    posts = fetch(args.target, include=args.include, exclude=args.exclude)
    jobboard.save_posts(posts, location=args.location, out_dir=args.out_dir)
