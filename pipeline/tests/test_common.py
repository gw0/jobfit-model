"""Unit tests for common.py. torch-dependent tests skip when torch is absent (CI)."""

import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import common


def test_build_targets_joins_pairwise_and_single_doc_labels():
    pairs = [{"cv": "cvs/a.md", "job": "jobs/acme/1.md"}]
    labels = (
        {"cvs/a.md": {"cv_clarity_structure_quality_score": 0.8, "cv_clarity_structure_quality_confidence": 0.9,
                      "cv_likely_llm_generated_score": 0.1, "cv_likely_llm_generated_confidence": 0.7}},
        {"jobs/acme/1.md": {"job_post_clarity_structure_quality_score": 0.6, "job_post_clarity_structure_quality_confidence": 0.5,
                            "job_post_likely_llm_generated_score": 0.2, "job_post_likely_llm_generated_confidence": 0.4}},
        {("cvs/a.md", "jobs/acme/1.md"): {"skills_match_score": 0.7, "skills_match_confidence": 0.9,
                                          "overall_fit_score": None}},
    )
    targets, confidences = common.build_targets(pairs, labels)
    assert targets.shape == confidences.shape == (1, common.NUM_QUESTIONS)

    def at(aid):
        j = common.QUESTION_IDS.index(aid)
        return targets[0, j], confidences[0, j]

    assert at("skills_match") == (0.7, 0.9)
    assert at("cv_clarity_structure_quality") == (0.8, 0.9)
    assert at("job_post_likely_llm_generated") == (0.2, 0.4)
    assert all(math.isnan(v) for v in at("culture_company_alignment"))


def test_build_targets_empty_and_unlabeled():
    assert common.build_targets([], ({}, {}, {}))[0].shape == (0, common.NUM_QUESTIONS)
    targets, confidences = common.build_targets([{"cv": "cvs/x.md", "job": "jobs/y/1.md"}], ({}, {}, {}))
    assert np.isnan(targets).all() and np.isnan(confidences).all()


def test_json_roundtrip_writes_null_and_restores_nan(tmp_path):
    path = tmp_path / "out.json"
    common.write_json(path, {"question": {"temperature": float("nan"), "list": [1.0, float("nan"), None]}})
    assert "NaN" not in path.read_text() and "null" in path.read_text()
    result = common.read_json(path)
    assert math.isnan(result["question"]["temperature"])
    assert result["question"]["list"][0] == 1.0
    assert all(math.isnan(v) for v in result["question"]["list"][1:])


def test_model_slug():
    assert common.model_slug("Qwen/Qwen3-0.6B") == "qwen3-0.6b"


def test_git_sha_prefers_env(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "abc123")
    assert common.git_sha() == "abc123"
    monkeypatch.delenv("GIT_SHA")
    assert isinstance(common.git_sha(), str) and common.git_sha()


def test_stage_span_without_collector_is_a_passthrough(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    with common.stage_span("unit-test"):
        pass
    with pytest.raises(ValueError):  # a failing stage keeps its own exception
        with common.stage_span("unit-test"):
            raise ValueError("boom")


def test_utilization_gauges_observe_percentages_for_the_current_stage(monkeypatch):
    pytest.importorskip("opentelemetry.metrics")
    monkeypatch.setattr(common, "_current_stage", "unit-test")
    for _, read in common._GAUGES.values():
        (observation,) = common._observer(read)(None)
        assert 0 <= observation.value <= 100
        assert observation.attributes == {"stage": "unit-test"}


def test_utilization_gauge_reader_failure_reads_zero():
    pytest.importorskip("opentelemetry.metrics")

    def broken():
        raise RuntimeError("no device")

    (observation,) = common._observer(broken)(None)
    assert observation.value == 0.0


# --- parse_args / candidate ------------------------------------------------------------

def _parse(tmp_path, *argv, stage=None):
    parser = argparse.ArgumentParser()
    common.add_common_args(parser)
    common.add_inference_args(parser)
    common.add_seed_arg(parser)
    parser.add_argument("--lr", type=float, default=1e-4)
    return common.parse_args(parser, stage, ["--datasets-dir", str(tmp_path), "--runs-dir", str(tmp_path), *argv])


def test_parse_args_candidate_defaults_to_the_model_slug(tmp_path):
    args = _parse(tmp_path)
    assert args.candidate == "smollm2-135m-instruct" and args.run_dir == tmp_path / "smollm2-135m-instruct"
    args = _parse(tmp_path, "--candidate", "smollm2-135m-instruct-lr3e-4", "--lr", "3e-4")
    assert args.run_dir == tmp_path / "smollm2-135m-instruct-lr3e-4"
    assert not (args.run_dir / "config.json").exists()  # only stages that build the candidate record it


def test_parse_args_merges_each_stage_into_config_json(tmp_path):
    _parse(tmp_path, "--run-group", "wf-1", stage="prepare")
    _parse(tmp_path, "--lr", "3e-4", stage="finetune")
    config = common.read_json(tmp_path / "smollm2-135m-instruct" / "config.json")
    assert set(config) == {"prepare", "finetune"} and config["finetune"]["lr"] == 3e-4
    assert config["prepare"]["model"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert "runs_dir" not in config["prepare"] and "run_group" not in config["prepare"]


# --- mlflow_run parent lookup ------------------------------------------------------------

def _args(candidate="smollm2-135m-instruct", run_group=None):
    return types.SimpleNamespace(model=common.DEFAULT_MODEL, candidate=candidate, run_group=run_group,
                                 runs_dir=Path("/runs"))


class _FakeMlflow(types.SimpleNamespace):
    """Records start_run calls; search_runs finds whatever parent `create_run` made."""

    def __init__(self):
        super().__init__(started=[], created=[])
        fake = self

        class Client:
            def create_run(self, experiment_id, tags):
                fake.created.append(tags)
                return types.SimpleNamespace(info=types.SimpleNamespace(run_id=f"parent-{len(fake.created)}"))

        self.MlflowClient = Client

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        pass

    def get_experiment_by_name(self, name):
        return types.SimpleNamespace(experiment_id="1")

    def search_runs(self, experiment_ids, filter_string, max_results, output_format):
        group = filter_string.split("'")[1]
        return [types.SimpleNamespace(info=types.SimpleNamespace(run_id=f"parent-{i + 1}"))
                for i, tags in enumerate(self.created) if tags["run_group"] == group][:1]

    def start_run(self, **kwargs):
        self.started.append(kwargs)
        return _NullContext(types.SimpleNamespace(info=types.SimpleNamespace(run_id="child")))


class _NullContext:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


def test_mlflow_run_nests_every_stage_under_one_parent_per_group(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    for stage in ("prepare", "finetune"):
        with common.mlflow_run(stage, _args(run_group="wf-1")):
            pass
    with common.mlflow_run("prepare", _args(run_group="wf-2")):
        pass

    assert [c["run_group"] for c in fake.created] == ["wf-1", "wf-2"]
    parents = [s["run_id"] for s in fake.started if "run_id" in s]
    assert parents == ["parent-1", "parent-1", "parent-2"]
    assert all(s["nested"] for s in fake.started if "run_id" not in s)


def test_mlflow_run_without_group_is_top_level(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with common.mlflow_run("calibrate", _args(candidate="smollm2-135m-instruct-r16"), tags={"extra": "x"}):
        pass
    assert fake.tracking_uri == "file:///runs/mlflow-local" and fake.created == []
    (call,) = fake.started
    assert call["run_name"] == "calibrate-smollm2-135m-instruct-r16" and call["tags"]["extra"] == "x"
    assert call["tags"]["candidate"] == "smollm2-135m-instruct-r16" and call["tags"]["model"] == common.DEFAULT_MODEL


def test_label_weights_mask_missing_and_below_floor():
    targets = np.array([[0.5, np.nan, 0.5, 0.5]])
    confidences = np.array([[0.9, 0.9, 0.1, np.nan]])
    assert common.usable(targets, confidences, 0.3).tolist() == [[True, False, False, False]]
    assert common.label_weights(targets, confidences, 0.3).tolist() == [[0.9, 0.0, 0.0, 0.0]]


def test_answer_targets_two_hot_on_the_level_scale():
    targets = np.full((1, common.NUM_QUESTIONS), np.nan)
    targets[0, 0] = 0.3  # 5 levels: index 1.2
    dists = common.answer_targets(targets)
    assert dists.shape == (1, common.NUM_QUESTIONS, 10)
    assert np.allclose(dists[0, 0, :3], [0.0, 0.8, 0.2]) and (dists[0, 1:] == 0).all()


def test_read_scores_normalizes_and_applies_temperature():
    logits = np.full((1, common.NUM_QUESTIONS, 10), -1e9)
    logits[0, :, 4] = 0.0  # all mass on the top level of every 5-level question
    scores, confidences = common.read_scores(logits)
    assert np.allclose(scores, 1.0) and np.allclose(confidences, 1.0)
    flat = np.zeros((1, common.NUM_QUESTIONS, 10))
    flat[0, :, 4] = 1.0
    sharp, _ = common.read_scores(flat, temperature=0.01)
    soft, _ = common.read_scores(flat, temperature=100.0)
    assert np.allclose(sharp, 1.0) and np.allclose(soft, 0.5, atol=0.01)


def test_temperature_defaults_to_one():
    assert common.temperature(None) == 1.0
    assert common.temperature({"temperature": math.nan}) == 1.0
    assert common.temperature({"temperature": 2.0}) == 2.0


# --- torch ---------------------------------------------------------------------------------

def test_state_dict_hash_changes_with_any_weight():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    h = common.state_dict_hash(model)
    assert common.state_dict_hash(model) == h
    with torch.no_grad():
        model.bias += 1
    assert common.state_dict_hash(model) != h


def test_predict_answer_logits_batches_torch_and_onnx_sessions_alike():
    torch = pytest.importorskip("torch")
    n = 3
    cache = {"pairs": [{}] * n, "input_ids": torch.arange(n * 4).reshape(n, 4),
             "segment_ids": torch.zeros(n, 4, dtype=torch.long), "answer_positions": torch.zeros(n, 2, dtype=torch.long),
             "candidate_ids": torch.zeros(2, 10, dtype=torch.long)}

    class Model(torch.nn.Linear):
        def forward(self, input_ids, segment_ids, answer_positions, candidate_ids):
            return input_ids[:, :1, None].float().expand(-1, 2, 10)

    class Session:
        def run(self, names, feeds):
            assert names == ["answer_logits"] and set(feeds) == {*common.MODEL_INPUTS, "candidate_ids"}
            return [np.broadcast_to(feeds["input_ids"][:, :1, None], (len(feeds["input_ids"]), 2, 10)).astype(float)]

    for model in (Model(1, 1), Session()):
        logits = common.predict_answer_logits(model, cache, batch_size=2)
        assert logits.shape == (n, 2, 10) and logits[:, 0, 0].tolist() == [0, 4, 8]
