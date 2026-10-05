"""Unit tests for report.py and publish.py's local (non-network) logic."""

import math

import common
import publish
import report


def _eval_result(mean_mae, baseline_mae, n_beat, skipped=False, model="HuggingFaceTB/SmolLM2-135M-Instruct"):
    metrics, floor = {}, {}
    for i, qid in enumerate(common.QUESTION_IDS):
        metrics[qid] = {"mae": mean_mae - 0.05 if i < n_beat else mean_mae + 0.05,
                        "spearman_rho": 0.5, "confidence_error_spearman": -0.4, "insufficient_rate": 0.1}
        floor[qid] = {"mae": baseline_mae}
    shuffle = {qid: {"mean_abs_shift": 0.1, "mae": 0.4} for qid in common.QUESTION_IDS}
    return {"skipped": skipped, "model": model, "metrics": metrics,
            "baselines": {"train_mean": floor}, "shuffle_control": shuffle}


def test_beats_floor_count():
    assert report.beats_floor_count(_eval_result(0.2, 0.25, n_beat=10)) == 10
    assert report.beats_floor_count({"skipped": True}) is None


def test_pick_winner_across_candidates_uses_most_advanced_stage():
    candidates = {
        "cand-a": {"zeroshot": _eval_result(0.05, 0.4, 15), "quantized": _eval_result(0.3, 0.4, 5)},
        "cand-b": {"finetuned": _eval_result(0.2, 0.4, 10), "calibrated": _eval_result(0.1, 0.4, 12, skipped=True)},
    }
    assert report.pick_winner(candidates) == "cand-b"
    assert report.pick_winner({"cand": {"finetuned": {"skipped": True}}}) is None


def test_load_candidates_and_report_files(tmp_path):
    for slug, mae in (("smollm2-135m-instruct", 0.3), ("llama-3.2-1b", 0.2)):
        common.write_json(tmp_path / slug / "eval" / "quantized.json", _eval_result(mae, 0.4, 3, model=slug))
    candidates = publish.load_candidates(tmp_path)
    assert set(candidates) == {"smollm2-135m-instruct", "llama-3.2-1b"}
    export_dir = tmp_path / "smollm2-135m-instruct" / "export"
    (export_dir / "quantized").mkdir(parents=True)
    (export_dir / "model.onnx").write_bytes(b"x" * 10)
    (export_dir / "model.onnx_data").write_bytes(b"x" * 90)  # fp32 weights live out of the graph

    built = publish.build_report(tmp_path, "smollm2-135m-instruct", candidates, "abc123")
    assert built["deployability"]["onnx_fp32_bytes"] == 100
    assert built["deployability"]["onnx_quantized_bytes"] is None
    assert built["winner"] == "llama-3.2-1b"
    assert set(built["candidates"]) == {"smollm2-135m-instruct", "llama-3.2-1b"}
    assert math.isclose(built["quality"]["quantized"]["shuffle_mean_abs_shift"], 0.1)
    assert math.isnan(built["quality"]["zeroshot"]["mean_mae"])

    json_path, md_path = publish.write_report(built, tmp_path / "smollm2-135m-instruct" / "reports")
    assert common.read_json(json_path)["git_sha"] == "abc123"
    md = md_path.read_text()
    assert "abc123" in md and "`llama-3.2-1b`" in md and "17/17" in md
    assert "Acceptance criteria" not in md


def test_render_markdown_calibration_mean():
    built = report.assemble_report("smollm2-135m-instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "abc", {"calibrated": _eval_result(0.2, 0.3, 1)},
                                   {"temperature": 1.5, "confidence_threshold": 0.4}, {})
    built["calibration"]["calibrated"][common.QUESTION_IDS[0]]["insufficient_rate"] = 1.0
    rows = {r["stage"]: r for r in report.stage_rows(built)}
    assert math.isclose(rows["calibrated"]["mean_insufficient_rate"], (0.1 * 16 + 1.0) / 17)
    assert math.isclose(rows["calibrated"]["mean_confidence_error_spearman"], -0.4)
    md = report.render_markdown(built)
    assert "temperature 1.5000" in md and "n/a" in md  # stages without results


def test_hf_repo_id_and_model_card():
    assert publish.hf_repo_id("smollm2-135m-instruct", "jobfit") == "jobfit/jobfit-smollm2-135m-instruct"
    assert publish.hf_repo_id("smollm2-135m-instruct", "jobfit", "me/custom") == "me/custom"

    built = report.assemble_report("smollm2-135m-instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "abc123",
                                   {"quantized": _eval_result(0.2, 0.25, 12)}, None, {})
    card = publish.build_model_card(built, "jobfit/jobfit-smollm2-135m-instruct")
    assert card.startswith("---\n") and "license: apache-2.0" in card
    assert "abc123" in card and "12/17 questions" in card

    built = report.assemble_report("minicpm5-2b", "openbmb/MiniCPM5-2B", "abc123",
                                   {"quantized": _eval_result(0.2, 0.25, 12)}, None, {})
    card = publish.build_model_card(built, "jobfit/jobfit-minicpm5-2b")
    assert "license: other" in card and "verify before use" in card


def test_variant_candidate_reports_its_config_and_base_licence(tmp_path):
    common.write_json(tmp_path / "smollm2-135m-instruct-r16" / "eval" / "quantized.json", _eval_result(0.2, 0.25, 12))
    common.write_json(tmp_path / "smollm2-135m-instruct-r16" / "config.json", {"finetune": {"lora_rank": 16}})
    built = publish.build_report(tmp_path, "smollm2-135m-instruct-r16", publish.load_candidates(tmp_path), "abc123")
    assert built["candidate"] == "smollm2-135m-instruct-r16" and built["config"] == {"finetune": {"lora_rank": 16}}
    assert "- finetune: lora_rank=16" in report.render_markdown(built)
    assert "license: apache-2.0" in publish.build_model_card(built, "jobfit/jobfit-smollm2-135m-instruct-r16")
