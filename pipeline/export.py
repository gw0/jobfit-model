#!/usr/bin/env python3
"""`export` stage (specs §4): ONNX export of the finetuned JevModel plus int8
quantization (`quantize_onnx`), and the bundle the browser app loads.

The graph takes `input_ids`, `segment_ids`, `answer_positions` and `candidate_ids` and
returns `answer_logits`; it builds the block-diagonal mask itself, so it answers any
Jev question set, not just the one it was trained on. The shipped calibration is
re-fit on the quantized model's `calib` outputs, since users should get the confidences
of the model they actually run. (`evaluate --stage quantized` still reports the
unchanged pre-quantization fit, so the report shows what quantization cost.)

Usage:
    ./pipeline/export.py --datasets-dir datasets_smoke --runs-dir runs_smoke

Reads <runs-dir>/<candidate>/checkpoints/finetune/, writes under <runs-dir>/<candidate>/export/:
    model.onnx    fp32 ONNX (weights in model.onnx_data)
    quantized/    model_quantized.onnx, int8 (see `quantize_onnx`)
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


def matmul_names(path):
    """(weight, activation) MatMul names of an ONNX graph: those whose B operand is an
    initializer, and the rest (attention scores, rotary angles)."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    initializers = {i.name for i in graph.initializer}
    matmuls = [n for n in graph.node if n.op_type == "MatMul"]
    return ([n.name for n in matmuls if n.input[1] in initializers],
            [n.name for n in matmuls if n.input[1] not in initializers])


def dequantize_weights(model, names):
    """Stores each named MatMul's weight as int8 (symmetric, per output channel) behind a
    DequantizeLinear node: the weight is small on disk, the activations stay fp32."""
    import numpy as np
    from onnx import helper, numpy_helper

    graph = model.graph
    weights = {i.name: i for i in graph.initializer}
    nodes = []
    for node in graph.node:
        if node.name in names:
            weight = numpy_helper.to_array(weights[node.input[1]])
            scale = np.maximum(np.abs(weight).max(axis=0), 1e-8) / 127
            key = node.name.strip("/").replace("/", "_")
            graph.initializer.extend([
                numpy_helper.from_array(np.clip(np.round(weight / scale), -127, 127).astype(np.int8), f"{key}_int8"),
                numpy_helper.from_array(scale.astype(np.float32), f"{key}_scale")])
            nodes.append(helper.make_node("DequantizeLinear", [f"{key}_int8", f"{key}_scale"], [f"{key}_weight"], axis=1))
            graph.initializer.remove(weights[node.input[1]])
            node.input[1] = f"{key}_weight"
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)


def quantize_onnx(export_dir, quantized_dir):
    """int8 `export_dir/model.onnx` -> `quantized_dir/model_quantized.onnx`. Dynamic int8
    for every weight MatMul except `down_proj`, whose inputs are outlier-heavy: its weight
    is int8 but its activations stay fp32. The activation-by-activation MatMuls (attention
    scores, rotary angles) stay fp32 too; quantizing either set collapses the answers
    (measured: MAE 0.29 vs 0.13 on Qwen3-0.6B)."""
    import onnx
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    weight_names, activation_names = matmul_names(export_dir / "model.onnx")
    down_proj = {n for n in weight_names if "/down_proj/" in n}
    ORTQuantizer.from_pretrained(export_dir, file_name="model.onnx").quantize(
        save_dir=quantized_dir, quantization_config=AutoQuantizationConfig.avx2(
            is_static=False, per_channel=False, nodes_to_exclude=[*activation_names, *down_proj]))
    path = quantized_dir / "model_quantized.onnx"
    model = onnx.load(str(path))
    dequantize_weights(model, down_proj)
    onnx.save(model, str(path))


def logit_error(reference, session, cache, n=8):
    """Mean |answer logit| difference of two ONNX sessions on the first `n` pairs of `cache`."""
    import numpy as np

    head = {**cache, "pairs": cache["pairs"][:n], **{name: cache[name][:n] for name in common.MODEL_INPUTS}}
    a, b = (common.predict_answer_logits(s, head, n) for s in (reference, session))
    return float(np.abs(a - b).mean())


def refit_calibration(session, args):
    calib_cache = common.load_cache(args.run_dir, "calib")
    if calib_cache is None:
        print("0 calib pairs -- shipping an empty calibration")
        return {"temperature": math.nan, "confidence_threshold": math.nan}
    answer_logits = common.predict_answer_logits(session, calib_cache, args.batch_size)
    targets, confidences = common.build_targets(calib_cache["pairs"], common.load_labels(args.datasets_dir))
    return metrics.fit_calibration(answer_logits, targets, confidences, args.confidence_floor)


def _first_pair(run_dir):
    for name in ("test", "calib", "val", "train"):
        cache = common.load_cache(run_dir, name)
        if cache is not None:
            return cache["pairs"][0]
    raise SystemExit("no cached split found -- run prepare.py first")


def parity_fixture(session, tokenizer, pair, datasets_dir, calibration):
    """One state, encoded and answered, for the frontend to reproduce. The questions
    budget is widened to fit the extra questions; the graph has no fixed length."""
    import numpy as np

    state = common.jobfit_state((datasets_dir / pair["cv"]).read_text(encoding="utf-8"),
                         corpus.read_job_body(datasets_dir / pair["job"]))
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

    quantize_onnx(export_dir, quantized_dir)
    session = onnxruntime.InferenceSession(str(quantized_dir / "model_quantized.onnx"))
    print(f"quantized (int8) -> {quantized_dir}")

    (web_dir / "onnx").mkdir(parents=True)
    tokenizer.save_pretrained(web_dir)
    model.lm.config.save_pretrained(web_dir)
    shutil.copy(quantized_dir / "model_quantized.onnx", web_dir / "onnx" / "model_quantized.onnx")
    calibration = refit_calibration(session, args)
    common.write_json(web_dir / "calibration.json", calibration)
    common.write_json(web_dir / "parity.json", parity_fixture(
        session, tokenizer, _first_pair(args.run_dir), args.datasets_dir, calibration))
    print(f"browser bundle (re-fit calibration, parity fixture) -> {web_dir}")

    with common.mlflow_run("export", args):
        mlflow.log_metrics({
            "onnx_fp32_bytes": common.onnx_bytes(export_dir),
            "onnx_quantized_bytes": common.onnx_bytes(quantized_dir),
            "quantized_logit_error": logit_error(onnxruntime.InferenceSession(str(export_dir / "model.onnx")),
                                                 session, train_cache),
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
