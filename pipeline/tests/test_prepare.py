"""Unit tests for prepare.py's pure logic (encoding itself is tested in test_jev.py)."""

import pytest

from prepare import LeakageError, assert_leakage_free, derange_pairs


def test_leakage_check():
    assert_leakage_free({})
    assert_leakage_free({
        "train": [{"cv": "cvs/a.md", "job": "jobs/acme/1.md"}],
        "test": [{"cv": "cvs/b.md", "job": "jobs/other-co/2.md"}],
    })
    with pytest.raises(LeakageError, match="cvs/a.md"):
        assert_leakage_free({
            "train": [{"cv": "cvs/a.md", "job": "jobs/acme/1.md"}],
            "test": [{"cv": "cvs/a.md", "job": "jobs/other-co/2.md"}],
        })
    with pytest.raises(LeakageError, match="acme"):
        assert_leakage_free({
            "train": [{"cv": "cvs/a.md", "job": "jobs/acme/1.md"}],
            "test": [{"cv": "cvs/b.md", "job": "jobs/acme/2.md"}],
        })


def test_derange_pairs_gives_every_cv_a_job_it_is_not_paired_with():
    # A full CV x job cross product: no in-split permutation could mismatch anything.
    jobs = [f"jobs/co/{i}.md" for i in range(4)]
    pairs = [{"cv": cv, "job": job} for cv in ("cvs/a.md", "cvs/b.md") for job in jobs]
    all_jobs = jobs + ["jobs/other/1.md", "jobs/other/2.md"]
    shuffled = derange_pairs(pairs, all_jobs, seed=1)
    originals = {(p["cv"], p["job"]) for p in pairs}
    assert [p["cv"] for p in shuffled] == [p["cv"] for p in pairs]
    assert not any((p["cv"], p["job"]) in originals for p in shuffled)
    assert shuffled == derange_pairs(pairs, all_jobs, seed=1)
    with pytest.raises(ValueError):
        derange_pairs(pairs, jobs)
