"""Unit tests for report.py and publish.py's local (non-network) logic."""

import math

import common
import publish
import report


def _eval_result(mean_mae, baseline_mae, n_beat, skipped=False, model="Qwen/Qwen3-0.6B"):
    metrics, floor = {}, {}
    for i, aid in enumerate(common.ASPECT_IDS):
        metrics[aid] = {"mae": mean_mae - 0.05 if i < n_beat else mean_mae + 0.05,
                        "spearman_rho": 0.5, "coverage": 0.9, "mean_width": 0.3}
        floor[aid] = {"mae": baseline_mae}
    shuffle = {aid: {"mean_abs_shift": 0.1, "mae": 0.4} for aid in common.ASPECT_IDS}
    return {"skipped": skipped, "model": model, "metrics": metrics,
            "baselines": {"train_mean": floor}, "shuffle_control": shuffle}


def test_beats_floor_count():
    assert report.beats_floor_count(_eval_result(0.2, 0.25, n_beat=10)) == 10
    assert report.beats_floor_count({"skipped": True}) is None


def test_pick_winner_across_candidates_uses_most_advanced_stage():
    candidates = {
        "cand-a": {"headtrained": _eval_result(0.05, 0.4, 15), "quantized": _eval_result(0.3, 0.4, 5)},
        "cand-b": {"finetuned": _eval_result(0.2, 0.4, 10), "calibrated": _eval_result(0.1, 0.4, 12, skipped=True)},
    }
    assert report.pick_winner(candidates) == "cand-b"
    assert report.pick_winner({"cand": {"finetuned": {"skipped": True}}}) is None


def test_load_candidates_and_report_files(tmp_path):
    for slug, mae in (("qwen3-0.6b", 0.3), ("llama-3.2-1b", 0.2)):
        common.write_json(tmp_path / slug / "eval" / "quantized.json", _eval_result(mae, 0.4, 3, model=slug))
    candidates = publish.load_candidates(tmp_path)
    assert set(candidates) == {"qwen3-0.6b", "llama-3.2-1b"}
    export_dir = tmp_path / "qwen3-0.6b" / "export"
    (export_dir / "quantized").mkdir(parents=True)
    (export_dir / "model.onnx").write_bytes(b"x" * 10)
    (export_dir / "model.onnx_data").write_bytes(b"x" * 90)  # fp32 weights live out of the graph

    built = publish.build_report(tmp_path, "qwen3-0.6b", candidates, "abc123")
    assert built["deployability"]["onnx_fp32_bytes"] == 100
    assert built["deployability"]["onnx_quantized_bytes"] is None
    assert built["winner"] == "llama-3.2-1b"
    assert set(built["candidates"]) == {"qwen3-0.6b", "llama-3.2-1b"}
    assert math.isclose(built["quality"]["quantized"]["shuffle_mean_abs_shift"], 0.1)
    assert math.isnan(built["quality"]["headtrained"]["mean_mae"])

    json_path, md_path = publish.write_report(built, tmp_path / "qwen3-0.6b" / "reports")
    assert common.read_json(json_path)["git_sha"] == "abc123"
    md = md_path.read_text()
    assert "abc123" in md and "`llama-3.2-1b`" in md and "17/17" in md
    assert "Acceptance criteria" not in md


def test_render_markdown_calibration_mean():
    built = report.assemble_report("qwen3-0.6b", "Qwen/Qwen3-0.6B", "abc", {"calibrated": _eval_result(0.2, 0.3, 1)},
                                   None, {})
    built["calibration"]["calibrated"][common.ASPECT_IDS[0]]["coverage"] = 0.0
    rows = {r["stage"]: r for r in report.stage_rows(built)}
    assert math.isclose(rows["calibrated"]["mean_coverage"], 0.9 * 16 / 17)
    assert "n/a" in report.render_markdown(built)  # stages without results


def test_hf_repo_id_and_model_card():
    assert publish.hf_repo_id("qwen3-0.6b", "jobfit") == "jobfit/jobfit-qwen3-0.6b"
    assert publish.hf_repo_id("qwen3-0.6b", "jobfit", "me/custom") == "me/custom"

    built = report.assemble_report("qwen3-0.6b", "Qwen/Qwen3-0.6B", "abc123",
                                   {"quantized": _eval_result(0.2, 0.25, 12)}, None, {})
    card = publish.build_model_card(built, "jobfit/jobfit-qwen3-0.6b")
    assert card.startswith("---\n") and "license: apache-2.0" in card
    assert "abc123" in card and "12/17 aspects" in card

    built = report.assemble_report("minicpm5-2b", "openbmb/MiniCPM5-2B", "abc123",
                                   {"quantized": _eval_result(0.2, 0.25, 12)}, None, {})
    card = publish.build_model_card(built, "jobfit/jobfit-minicpm5-2b")
    assert "license: other" in card and "verify before use" in card
