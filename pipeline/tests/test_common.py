"""Unit tests for common.py."""

import math

import numpy as np

import common


def test_build_targets_joins_pairwise_and_single_doc_labels():
    pairs = [{"cv": "cvs/a.md", "job": "jobs/acme/1.md"}]
    labels = (
        {"cvs/a.md": {"clarity_score": 0.8, "clarity_confidence": 0.9,
                      "llm_generated_score": 0.1, "llm_generated_confidence": 0.7}},
        {"jobs/acme/1.md": {"clarity_score": 0.6, "clarity_confidence": 0.5,
                            "llm_generated_score": 0.2, "llm_generated_confidence": 0.4}},
        {("cvs/a.md", "jobs/acme/1.md"): {"skills_match_score": 0.7, "skills_match_confidence": 0.9,
                                          "overall_fit_score_score": None}},
    )
    targets, confidences = common.build_targets(pairs, labels)
    assert targets.shape == confidences.shape == (1, common.NUM_ASPECTS)

    def at(aid):
        j = common.ASPECT_IDS.index(aid)
        return targets[0, j], confidences[0, j]

    assert at("skills_match") == (0.7, 0.9)
    assert at("cv_clarity_structure_quality") == (0.8, 0.9)
    assert at("job_post_likely_llm_generated") == (0.2, 0.4)
    assert all(math.isnan(v) for v in at("culture_company_alignment_score"))


def test_build_targets_empty_and_unlabeled():
    assert common.build_targets([], ({}, {}, {}))[0].shape == (0, common.NUM_ASPECTS)
    targets, confidences = common.build_targets([{"cv": "cvs/x.md", "job": "jobs/y/1.md"}], ({}, {}, {}))
    assert np.isnan(targets).all() and np.isnan(confidences).all()


def test_json_roundtrip_writes_null_and_restores_nan(tmp_path):
    path = tmp_path / "out.json"
    common.write_json(path, {"aspect": {"delta": float("nan"), "list": [1.0, float("nan"), None]}})
    assert "NaN" not in path.read_text() and "null" in path.read_text()
    result = common.read_json(path)
    assert math.isnan(result["aspect"]["delta"])
    assert result["aspect"]["list"][0] == 1.0
    assert all(math.isnan(v) for v in result["aspect"]["list"][1:])


def test_model_slug():
    assert common.model_slug("Qwen/Qwen3-0.6B") == "qwen3-0.6b"
