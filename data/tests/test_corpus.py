"""Unit tests for corpus.py."""

import corpus


def test_read_job_body_strips_frontmatter(tmp_path):
    path = tmp_path / "job.md"
    path.write_text("---\ncompany: acme\nurl: https://x/1\n---\n\n# Engineer\n\nBuild things.\n")
    assert corpus.read_job_body(path) == "# Engineer\n\nBuild things.\n"
    assert corpus.parse_frontmatter(path.read_text())[0] == {"company": "acme", "url": "https://x/1"}


def test_parse_frontmatter_without_block_returns_text_unchanged():
    assert corpus.parse_frontmatter("# Jane Doe\n") == ({}, "# Jane Doe\n")


def test_jsonl_roundtrip_and_missing_file(tmp_path):
    records = [{"cv": "cvs/a.md", "score": None}, {"cv": "cvs/b.md", "score": 0.5}]
    corpus.write_jsonl(tmp_path / "sub" / "x.jsonl", records)
    assert corpus.load_jsonl(tmp_path / "sub" / "x.jsonl") == records
    assert corpus.load_jsonl(tmp_path / "missing.jsonl") == []


def test_list_cvs_and_jobs(tmp_path):
    (tmp_path / "cvs").mkdir()
    (tmp_path / "cvs" / "b.md").write_text("b")
    (tmp_path / "cvs" / "a.md").write_text("a")
    (tmp_path / "jobs" / "acme").mkdir(parents=True)
    (tmp_path / "jobs" / "acme" / "1.md").write_text("j")
    (tmp_path / "jobs" / "empty").mkdir()
    assert corpus.list_cvs(tmp_path) == ["cvs/a.md", "cvs/b.md"]
    assert corpus.list_jobs(tmp_path) == {"acme": ["jobs/acme/1.md"]}
    assert corpus.company_of("jobs/acme/1.md") == "acme"


def test_slugify():
    assert corpus.slugify("Senior Backend Engineer!") == "senior-backend-engineer"
    assert corpus.slugify("  Foo___Bar--Baz  ") == "foo-bar-baz"
    assert corpus.slugify("Marcus Albrecht-Thorne, Jr.") == "marcus-albrecht-thorne-jr"
    assert corpus.slugify("") == corpus.slugify(None) == ""


def test_frontmatter_roundtrip_flattens_newlines():
    meta = {"company": "Acme", "title": "Staff\nEngineer", "location": None}
    text = corpus.format_frontmatter(meta) + "\n\nBody\n"
    assert corpus.parse_frontmatter(text) == (
        {"company": "Acme", "title": "Staff Engineer", "location": ""}, "Body\n")
