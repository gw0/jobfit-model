"""Unit tests for common.py. torch-dependent tests skip when torch is absent (CI)."""

import math
import sys
import types

import numpy as np
import pytest

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


# --- mlflow_run parent lookup ------------------------------------------------------------

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
    for stage in ("prepare", "headtrain"):
        with common.mlflow_run(stage, "Qwen/Qwen3-0.6B", run_group="wf-1"):
            pass
    with common.mlflow_run("prepare", "Qwen/Qwen3-0.6B", run_group="wf-2"):
        pass

    assert [c["run_group"] for c in fake.created] == ["wf-1", "wf-2"]
    parents = [s["run_id"] for s in fake.started if "run_id" in s]
    assert parents == ["parent-1", "parent-1", "parent-2"]
    assert all(s["nested"] for s in fake.started if "run_id" not in s)


def test_mlflow_run_without_group_is_top_level(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    with common.mlflow_run("calibrate", "Qwen/Qwen3-0.6B", tags={"extra": "x"}):
        pass
    assert fake.created == []
    (call,) = fake.started
    assert call["run_name"] == "calibrate-qwen3-0.6b" and call["tags"]["extra"] == "x"


# --- torch ---------------------------------------------------------------------------------

def _scalar_pinball_reference(preds, targets, confidences, floor):
    """The original per-element loop the vectorized loss replaced."""
    total = weight_sum = 0.0
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            t, c = targets[i][j], confidences[i][j]
            if math.isnan(t) or math.isnan(c) or c < floor:
                continue
            for k, q in enumerate(common.QUANTILES):
                diff = t - preds[i][j][k]
                total += c * max(q * diff, (q - 1) * diff)
            weight_sum += c * len(common.QUANTILES)
    return total / weight_sum if weight_sum else 0.0


def test_pinball_loss_matches_scalar_reference():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    preds = rng.normal(0.5, 0.3, size=(5, common.NUM_ASPECTS, 3))
    targets = rng.uniform(0, 1, size=(5, common.NUM_ASPECTS))
    confidences = rng.uniform(0, 1, size=(5, common.NUM_ASPECTS))
    targets[0, :4] = np.nan
    confidences[1, 2] = np.nan
    expected = _scalar_pinball_reference(preds.tolist(), targets.tolist(), confidences.tolist(), 0.3)
    loss = common.pinball_loss(torch.tensor(preds, requires_grad=True), torch.tensor(targets),
                               torch.tensor(confidences), 0.3)
    assert math.isclose(loss.item(), expected, rel_tol=1e-9)
    loss.backward()  # finite, differentiable despite NaN targets


def test_pinball_loss_all_masked_is_zero():
    torch = pytest.importorskip("torch")
    preds = torch.zeros(2, common.NUM_ASPECTS, 3, requires_grad=True)
    loss = common.pinball_loss(preds, torch.full((2, common.NUM_ASPECTS), 0.5),
                               torch.zeros(2, common.NUM_ASPECTS), 0.3)
    assert loss.item() == 0.0


class _TinyModel:
    base_model_prefix = "backbone"

    def __init__(self, torch):
        self.backbone = torch.nn.Linear(2, 2)
        self.score = torch.nn.Linear(2, 1)

    def named_parameters(self):
        yield from (("backbone." + n, p) for n, p in self.backbone.named_parameters())
        yield from (("score." + n, p) for n, p in self.score.named_parameters())


def test_freeze_backbone_and_hash():
    torch = pytest.importorskip("torch")
    model = _TinyModel(torch)
    common.freeze_backbone(model)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.score.parameters())

    h = common.backbone_state_dict_hash(model)
    with torch.no_grad():
        model.score.weight += 1
    assert common.backbone_state_dict_hash(model) == h
    with torch.no_grad():
        model.backbone.weight += 1
    assert common.backbone_state_dict_hash(model) != h


def test_predict_quantiles_batched_sorts_crossed_quantiles():
    torch = pytest.importorskip("torch")
    logits = torch.arange(common.NUM_LABELS, 0, -1, dtype=torch.float32).repeat(3, 1)  # every triple descending

    def model(input_ids, attention_mask):
        return types.SimpleNamespace(logits=logits[: len(input_ids)])

    preds = common.predict_quantiles_batched(model, torch.zeros(3, 4), torch.ones(3, 4), batch_size=2)
    assert preds.shape == (3, common.NUM_ASPECTS, 3)
    assert (np.diff(preds, axis=-1) >= 0).all()
