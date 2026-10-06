"""Unit tests for jobboard.py's pure helpers. No network."""

from datetime import date

import corpus
import jobboard


def test_extract_slug_from_url():
    assert jobboard.extract_slug("https://boards.greenhouse.io/acme/jobs/123") == "acme"


def test_extract_slug_from_url_trailing_slash_only():
    assert jobboard.extract_slug("https://jobs.lever.co/") == "https://jobs.lever.co/"


def test_extract_slug_passthrough_for_plain_slug():
    assert jobboard.extract_slug("acme-corp") == "acme-corp"


def test_extract_slug_unquotes_path():
    assert jobboard.extract_slug("https://api.ashbyhq.com/acme%20corp/jobs") == "acme corp"


def test_date_slug_valid():
    assert jobboard._date_slug("2026-06-15") == "20260615"


def test_date_slug_valid_ignores_trailing_content():
    assert jobboard._date_slug("2026-06-15T10:00:00Z") == "20260615"


def test_date_slug_missing_falls_back_to_today():
    assert jobboard._date_slug(None) == date.today().strftime("%Y%m%d")


def test_date_slug_malformed_falls_back_to_today():
    assert jobboard._date_slug("not-a-date") == date.today().strftime("%Y%m%d")


def test_date_slug_too_short_falls_back_to_today():
    assert jobboard._date_slug("2026-06") == date.today().strftime("%Y%m%d")


def test_matches_include_keyword():
    post = {"title": "Senior Backend Engineer"}
    assert jobboard.matches(post, include=["engineer"], exclude=None) is True


def test_matches_no_include_keyword_fails():
    post = {"title": "Product Manager"}
    assert jobboard.matches(post, include=["engineer"], exclude=None) is False


def test_matches_exclude_keyword_wins():
    post = {"title": "Sales Engineer"}
    assert jobboard.matches(post, include=["engineer"], exclude=["sales engineer"]) is False


def test_matches_location_filter():
    post = {"title": "Engineer", "location": "Remote - US"}
    assert jobboard.matches(post, location="remote-us") is True
    assert jobboard.matches(post, location="remote-eu") is False


def test_matches_sector_filter():
    post = {"title": "Engineer", "tags": ["fintech", "backend"]}
    assert jobboard.matches(post, sector="fintech") is True
    assert jobboard.matches(post, sector="healthcare") is False


def test_matches_no_filters_passes():
    assert jobboard.matches({"title": "Anything"}) is True


def test_write_post_dedupes_by_url(tmp_path):
    post = {"title": "Backend Engineer", "company": "Acme", "url": "https://x/1",
            "posted_at": "2026-06-15", "description": "v1"}
    path, is_new = jobboard.write_post(post, jobs_root=tmp_path)
    assert is_new and path == tmp_path / "acme" / "20260615-backend-engineer.md"
    path2, is_new2 = jobboard.write_post({**post, "description": "v2"}, jobs_root=tmp_path)
    assert path2 == path and not is_new2
    assert path.read_text().endswith("v2\n")
    assert corpus.parse_frontmatter(path.read_text())[0]["url"] == "https://x/1"


def test_write_post_redacts_emails(tmp_path):
    post = {"title": "Backend Engineer", "company": "Acme", "url": "https://x/1", "posted_at": "2026-06-15",
            "description": "Mail [jane.doe@acme.com](mailto:jane.doe@acme.com) or hr+jobs@acme.co.uk."}
    path, _ = jobboard.write_post(post, jobs_root=tmp_path)
    body = path.read_text()
    assert "acme.com" not in body and "acme.co.uk" not in body
    assert body.endswith("Mail [redacted@example.com](mailto:redacted@example.com) or redacted@example.com.\n")


def test_save_posts_limit_caps_matching_posts(tmp_path):
    posts = [{"title": f"Engineer {i}", "company": "Acme", "url": f"https://x/{i}",
              "posted_at": "2026-06-15", "description": "d"} for i in range(5)]
    assert jobboard.save_posts(posts, out_dir=tmp_path, limit=2) == 2
    assert len(list((tmp_path / "acme").glob("*.md"))) == 2
