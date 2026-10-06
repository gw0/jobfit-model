"""Unit tests for the pure split function in build_dataset.py. In-memory fixture, no I/O."""

from build_dataset import cosine, make_splits, tfidf_vectors
from corpus import SPLIT_NAMES

CV_IDS = [f"cvs/cv-{i}.md" for i in range(5)]  # 5 CVs
JOBS_BY_COMPANY = {
    f"company-{i}": [f"jobs/company-{i}/job-{i}.md"] for i in range(8)
}  # 8 companies, 1 job each

# The production default ratios (70/10/10/10, tuned for ~250 CVs) round every
# non-train group to 0 on a fixture this small (round(0.1*5) == 0). Use ratios sized
# for 5 CVs so every split gets at least one, to exercise all four groups here.
FIXTURE_CV_RATIOS = (0.4, 0.2, 0.2, 0.2)  # -> 2, 1, 1, 1 of 5


def _all_pairs(splits):
    return [p for pairs in splits.values() for p in pairs]


def test_every_split_key_present():
    splits = make_splits(CV_IDS, JOBS_BY_COMPANY, cv_ratios=FIXTURE_CV_RATIOS)
    assert set(splits.keys()) == set(SPLIT_NAMES)


def test_every_cv_used_at_most_in_one_split_group():
    """Each CV's pairs all land in exactly one split (CV-identity grouping)."""
    splits = make_splits(CV_IDS, JOBS_BY_COMPANY, cv_ratios=FIXTURE_CV_RATIOS)
    cv_to_splits = {}
    for split_name, pairs in splits.items():
        for pair in pairs:
            cv_to_splits.setdefault(pair["cv"], set()).add(split_name)
    for cv_id, split_set in cv_to_splits.items():
        assert len(split_set) == 1, f"{cv_id} appears in multiple splits: {split_set}"


def test_test_split_disjoint_from_train_on_both_axes():
    splits = make_splits(CV_IDS, JOBS_BY_COMPANY, cv_ratios=FIXTURE_CV_RATIOS)

    train_cvs = {p["cv"] for p in splits["train"]}
    train_companies = {p["job"].split("/")[1] for p in splits["train"]}
    test_cvs = {p["cv"] for p in splits["test"]}
    test_companies = {p["job"].split("/")[1] for p in splits["test"]}

    assert train_cvs.isdisjoint(test_cvs)
    assert train_companies.isdisjoint(test_companies)


def test_no_pair_duplicated_across_splits():
    pairs = [(p["cv"], p["job"]) for p in _all_pairs(make_splits(CV_IDS, JOBS_BY_COMPANY, cv_ratios=FIXTURE_CV_RATIOS))]
    assert len(pairs) == len(set(pairs))


def test_deterministic_given_seed():
    a = make_splits(CV_IDS, JOBS_BY_COMPANY, seed=7, cv_ratios=FIXTURE_CV_RATIOS)
    b = make_splits(CV_IDS, JOBS_BY_COMPANY, seed=7, cv_ratios=FIXTURE_CV_RATIOS)
    assert a == b


def test_different_seed_can_change_split():
    a = make_splits(CV_IDS, JOBS_BY_COMPANY, seed=1, cv_ratios=FIXTURE_CV_RATIOS)
    b = make_splits(CV_IDS, JOBS_BY_COMPANY, seed=2, cv_ratios=FIXTURE_CV_RATIOS)
    assert a != b


def test_nonempty_splits_on_this_fixture_size():
    """5 CVs / 8 companies is large enough that every split gets at least one CV."""
    splits = make_splits(CV_IDS, JOBS_BY_COMPANY, cv_ratios=FIXTURE_CV_RATIOS)
    for name in SPLIT_NAMES:
        assert len(splits[name]) > 0, f"{name} split is empty"


# --- --jobs-per-cv sampling ----------------------------------------------------------

JOB_IDS = [job_id for job_ids in JOBS_BY_COMPANY.values() for job_id in job_ids]
TEXTS = {
    **{cv_id: f"python backend engineer cv{i}" for i, cv_id in enumerate(CV_IDS)},
    **{job_id: ("python backend role" if i % 2 else "frontend react role") for i, job_id in enumerate(JOB_IDS)},
}


def _sampled(**kwargs):
    return make_splits(CV_IDS, JOBS_BY_COMPANY, cv_ratios=FIXTURE_CV_RATIOS, jobs_per_cv=2, texts=TEXTS, **kwargs)


def test_sampling_pairs_each_cv_with_k_jobs_of_its_own_pool():
    full, sampled = make_splits(CV_IDS, JOBS_BY_COMPANY, cv_ratios=FIXTURE_CV_RATIOS), _sampled()
    full_pairs = {(p["cv"], p["job"]) for p in _all_pairs(full)}
    per_cv = {}
    for p in _all_pairs(sampled):
        assert (p["cv"], p["job"]) in full_pairs  # same pools, so the same leakage guarantees
        per_cv[p["cv"]] = per_cv.get(p["cv"], 0) + 1
    assert set(per_cv.values()) == {2}


def test_sampling_takes_the_nearest_job_first():
    """k=2: one nearest by cosine (a "python backend" job for these CVs), one random."""
    pairs = _all_pairs(_sampled())
    for cv_id in CV_IDS:
        jobs = [p["job"] for p in pairs if p["cv"] == cv_id]
        assert any("backend" in TEXTS[job_id] for job_id in jobs)


def test_sampling_is_deterministic_given_seed():
    assert _sampled(seed=3) == _sampled(seed=3)


def test_tfidf_ignores_terms_shared_by_every_document():
    vectors = tfidf_vectors({"a": "role python", "b": "role react"})
    assert vectors["a"]["role"] == 0 and vectors["a"]["python"] > 0
    assert cosine(vectors["a"], vectors["b"]) == 0
