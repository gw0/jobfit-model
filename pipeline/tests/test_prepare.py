"""Unit tests for prepare.py's pure logic, with a whitespace stand-in tokenizer."""

import pytest

from prepare import LeakageError, assert_leakage_free, derange_pairs, encode_pair


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


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


def test_encode_pair_short_inputs_jd_last_and_padded():
    ids, mask, cv_trunc, jd_trunc = encode_pair(
        FakeTokenizer(), "cv one two", "jd three four", cv_budget=5, jd_budget=5, max_length=10)
    assert not cv_trunc and not jd_trunc
    assert ids == ["cv", "one", "two", "jd", "three", "four"] + [0] * 4
    assert mask == [1] * 6 + [0] * 4


def test_encode_pair_truncates_each_side_then_caps():
    cv = " ".join(f"cv{i}" for i in range(10))
    jd = " ".join(f"jd{i}" for i in range(10))
    ids, mask, cv_trunc, jd_trunc = encode_pair(FakeTokenizer(), cv, jd, cv_budget=3, jd_budget=4, max_length=20)
    assert cv_trunc and jd_trunc
    assert ids[:7] == ["cv0", "cv1", "cv2", "jd0", "jd1", "jd2", "jd3"] and sum(mask) == 7
    ids, mask, _, _ = encode_pair(FakeTokenizer(), cv, jd, cv_budget=8, jd_budget=8, max_length=12)
    assert len(ids) == 12 and sum(mask) == 12


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
