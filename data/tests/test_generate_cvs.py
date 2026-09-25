"""Unit tests for generate_cvs.py's file writing. No claude calls."""

from generate_cvs import PROMPT_TEMPLATE, profile, write_cv


def test_write_cv_numbers_name_collisions(tmp_path):
    paths = [write_cv(tmp_path, "jane-doe", f"# CV {i}") for i in range(3)]
    assert [p.name for p in paths] == ["jane-doe.md", "jane-doe-2.md", "jane-doe-3.md"]
    assert paths[2].read_text() == "# CV 2\n"


def test_profile_is_deterministic_and_varied():
    assert profile(3) == profile(3)
    assert profile(3, seed=1) != profile(3, seed=2)
    assert len({(p["role"], p["seniority"], p["location"]) for p in map(profile, range(50))}) > 40
    assert len({(p["first_initial"], p["last_initial"]) for p in map(profile, range(50))}) > 40
    fields = profile(0)
    assert f'starting with "{fields["first_initial"]}"' in PROMPT_TEMPLATE.format(**fields)
