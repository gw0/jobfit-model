#!/usr/bin/env python3
"""`export` stage (specs §4): ONNX export of the finetuned JevModel plus dynamic int8
quantization, and the bundle the browser app loads.

The graph takes `input_ids`, `segment_ids`, `answer_positions` and `candidate_ids` and
returns `answer_logits`; it builds the block-diagonal mask itself, so it answers any
Jev question set, not just the one it was trained on. The shipped calibration is
re-fit on the quantized model's `calib` outputs, since users should get the confidences
of the model they actually run. (`evaluate --stage quantized` still reports the
unchanged pre-quantization fit, so the report shows what quantization cost.)

Usage:
    ./pipeline/export.py --dataset-dir datasets_smoke --runs-dir runs_smoke

Reads <runs-dir>/<candidate>/checkpoints/finetune/, writes under <runs-dir>/<candidate>/export/:
    model.onnx    fp32 ONNX (weights in model.onnx_data)
    quantized/    model_quantized.onnx, dynamic int8
    web/          transformers.js layout: tokenizer + config files, onnx/model_quantized.onnx,
                  calibration.json and parity.json (a sample state with JobFit's questions
                  plus one choice and one noul question: its encoding, answer logits and
                  answers, checked by frontend/scripts/verify-parity.mjs)
"""

import argparse
import math
import shutil

import common
import jev
import metrics
from common import corpus

INPUT_NAMES = [*common.MODEL_INPUTS, "candidate_ids"]
PARITY_ATOL = 1e-2
# Question types JobFit never trains, checked end to end all the same.
PARITY_EXTRA_QUESTIONS = {
    "parity_choice": {"type": "choice", "question": "Which kind of role is the job?",
                      "criteria": {"ic": "Individual contributor", "manager": "People manager", "other": "Other"}},
    "parity_noul": {"type": "noul", "question": "Does the job description state a salary range?",
                    "criteria": {"true": "It does", "false": "It does not"}},
}


def export_onnx(model, cache, path):
    """fp32 ONNX of `model`, with batch, sequence, question and candidate axes dynamic,
    its weights in one `<name>_data` file (the exporter writes one file per tensor)."""
    import onnx
    import torch

    sample = (*[cache[name][:1] for name in common.MODEL_INPUTS], cache["candidate_ids"])
    dynamic_axes = {"input_ids": {0: "batch", 1: "sequence"}, "segment_ids": {0: "batch", 1: "sequence"},
                    "answer_positions": {0: "batch", 1: "questions"},
                    "candidate_ids": {0: "questions", 1: "candidates"},
                    "answer_logits": {0: "batch", 1: "questions", 2: "candidates"}}
    raw = path.parent / "raw"
    raw.mkdir()
    torch.onnx.export(model, sample, str(raw / path.name), input_names=INPUT_NAMES, output_names=["answer_logits"],
                      dynamic_axes=dynamic_axes, opset_version=17, dynamo=False, external_data=True)
    onnx.save_model(onnx.load(str(raw / path.name)), str(path), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=f"{path.name}_data")
    shutil.rmtree(raw)


def refit_calibration(session, args):
    calib_cache = common.load_cache(args.run_dir, "calib")
    if calib_cache is None:
        print("0 calib pairs -- shipping an empty calibration")
        return {"temperature": math.nan, "confidence_threshold": math.nan}
    answer_logits = common.predict_answer_logits(session, calib_cache, args.batch_size)
    targets, confidences = common.build_targets(calib_cache["pairs"], common.load_labels(args.dataset_dir))
    return metrics.fit_calibration(answer_logits, targets, confidences, args.confidence_floor)


def _first_pair(run_dir):
    for name in ("test", "calib", "val", "train"):
        cache = common.load_cache(run_dir, name)
        if cache is not None:
            return cache["pairs"][0]
    raise SystemExit("no cached split found -- run prepare.py first")


def parity_fixture(session, tokenizer, pair, dataset_dir, calibration):
    """One state, encoded and answered, for the frontend to reproduce. The questions
    budget is widened to fit the extra questions; the graph has no fixed length."""
    import numpy as np

    state = common.jobfit_state((dataset_dir / pair["cv"]).read_text(encoding="utf-8"),
                         corpus.read_job_body(dataset_dir / pair["job"]))
    questions = {**common.QUESTIONS, **PARITY_EXTRA_QUESTIONS}
    questions_budget = 2 * common.QUESTIONS_BUDGET
    encoded = jev.encode(tokenizer, state, questions, common.STATE_BUDGET, questions_budget, tokenizer.pad_token_id)
    candidate_ids, _ = jev.candidate_ids(tokenizer, questions)
    feeds = {name: np.array([encoded[name]]) for name in common.MODEL_INPUTS}
    answer_logits = session.run(["answer_logits"], {**feeds, "candidate_ids": np.array(candidate_ids)})[0][0]
    temperature = common.temperature(calibration)
    return {
        "cv": pair["cv"],
        "job": pair["job"],
        "state": state,
        "questions": questions,
        "state_budget": common.STATE_BUDGET,
        "questions_budget": questions_budget,
        "pad_token_id": tokenizer.pad_token_id,
        **{name: encoded[name] for name in common.MODEL_INPUTS},
        "candidate_ids": candidate_ids,
        "answer_logits": answer_logits.tolist(),
        "temperature": temperature,
        "answers": jev.answers(questions, answer_logits, temperature),
        "atol": PARITY_ATOL,
    }


def _run(args):
    import mlflow
    import onnxruntime
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    checkpoint_dir = args.run_dir / "checkpoints" / "finetune"
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"no finetune checkpoint under {checkpoint_dir} -- run train.py first")
    export_dir = args.run_dir / "export"
    quantized_dir, web_dir = export_dir / "quantized", export_dir / "web"
    train_cache = common.load_cache(args.run_dir, "train")
    if train_cache is None:
        raise SystemExit("no train cache -- run prepare.py first")

    tokenizer = common.load_tokenizer(checkpoint_dir)
    model = common.load_jev_model(checkpoint_dir, attn_implementation="eager", device="cpu").eval()
    shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.mkdir(parents=True)
    export_onnx(model, train_cache, export_dir / "model.onnx")
    model.lm.config.save_pretrained(export_dir)
    print(f"exported fp32 ONNX -> {export_dir}")

    quantizer = ORTQuantizer.from_pretrained(export_dir, file_name="model.onnx")
    quantizer.quantize(save_dir=quantized_dir,
                       quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=False))
    session = onnxruntime.InferenceSession(str(quantized_dir / "model_quantized.onnx"))
    print(f"quantized (dynamic int8) -> {quantized_dir}")

    (web_dir / "onnx").mkdir(parents=True)
    tokenizer.save_pretrained(web_dir)
    model.lm.config.save_pretrained(web_dir)
    shutil.copy(quantized_dir / "model_quantized.onnx", web_dir / "onnx" / "model_quantized.onnx")
    calibration = refit_calibration(session, args)
    common.write_json(web_dir / "calibration.json", calibration)
    common.write_json(web_dir / "parity.json", parity_fixture(
        session, tokenizer, _first_pair(args.run_dir), args.dataset_dir, calibration))
    print(f"browser bundle (re-fit calibration, parity fixture) -> {web_dir}")

    with common.mlflow_run("export", args):
        mlflow.log_metrics({
            "onnx_fp32_bytes": common.onnx_bytes(export_dir),
            "onnx_quantized_bytes": common.onnx_bytes(quantized_dir),
        })
        common.log_model_signature(session, train_cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    args = common.parse_args(parser, "export")
    with common.stage_span("export"):
        _run(args)


if __name__ == "__main__":
    main()
