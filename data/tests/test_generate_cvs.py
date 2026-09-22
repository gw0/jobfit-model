"""Unit tests for generate_cvs.py's file writing. No claude calls."""

from generate_cvs import write_cv


def test_write_cv_numbers_name_collisions(tmp_path):
    paths = [write_cv(tmp_path, "jane-doe", f"# CV {i}") for i in range(3)]
    assert [p.name for p in paths] == ["jane-doe.md", "jane-doe-2.md", "jane-doe-3.md"]
    assert paths[2].read_text() == "# CV 2\n"
