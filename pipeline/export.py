#!/usr/bin/env python3
"""`export` stage (specs §4): ONNX export of the finetuned checkpoint plus dynamic int8
quantization, and the bundle the browser app loads.

The shipped conformal calibration is re-fit on the quantized model's `calib` outputs,
since users shouldn't be served intervals the pre-quantization fit no longer covers.
(`evaluate --stage quantized` still reports the unchanged pre-quantization fit, so
the report shows what quantization cost.)

Usage:
    ./pipeline/export.py --dataset-dir datasets_smoke --runs-dir runs_smoke

Reads <runs-dir>/<slug>/checkpoints/finetune/, writes under <runs-dir>/<slug>/export/:
    ./            fp32 ONNX (optimum ORTModel dir)
    quantized/    int8 ONNX (optimum ORTModel dir)
    web/          transformers.js layout: tokenizer + config files, onnx/model_quantized.onnx,
                  calibration.json and parity.json (a sample pair with its expected
                  token ids and logits, checked by frontend/scripts/verify-parity.mjs)
"""

import argparse
import math
import shutil

import common
import metrics
from common import corpus

PARITY_ATOL = 1e-2


def _first_cache(runs_dir, slug):
    for name in ("test", "calib", "val", "train"):
        cache = common.load_cache(runs_dir, slug, name)
        if cache is not None:
            return cache
    raise SystemExit("no cached split found -- run prepare.py first")


def refit_calibration(model, args, slug):
    calib_cache = common.load_cache(args.runs_dir, slug, "calib")
    if calib_cache is None:
        print("0 calib pairs -- shipping an empty calibration")
        return {aid: {"delta": math.nan, "insufficient_data_threshold": math.nan} for aid in common.ASPECT_IDS}
    preds_q = common.predict_quantiles_batched(
        model, calib_cache["input_ids"], calib_cache["attention_mask"], args.batch_size)
    targets, confidences = common.build_targets(calib_cache["pairs"], common.load_labels(args.dataset_dir))
    return metrics.fit_calibration(preds_q, targets, confidences, args.confidence_floor)


def parity_fixture(model, tokenizer, cache, dataset_dir):
    """The first cached pair's raw text, token ids and raw logits (not quantile-sorted),
    for the frontend to reproduce."""
    import torch

    pair = cache["pairs"][0]
    input_ids, attention_mask = cache["input_ids"][:1], cache["attention_mask"][:1]
    with torch.no_grad():
        logits = torch.as_tensor(model(input_ids=input_ids, attention_mask=attention_mask).logits)
    return {
        "cv": pair["cv"],
        "job": pair["job"],
        "cv_text": (dataset_dir / pair["cv"]).read_text(encoding="utf-8"),
        "jd_text": corpus.read_job_body(dataset_dir / pair["job"]),
        "pad_token_id": tokenizer.pad_token_id,
        "input_ids": input_ids[0][attention_mask[0].bool()].tolist(),
        "logits": logits[0].tolist(),
        "atol": PARITY_ATOL,
    }


def _run(args):
    import mlflow
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    slug = common.model_slug(args.model)
    checkpoint_dir = args.runs_dir / slug / "checkpoints" / "finetune"
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"no finetune checkpoint under {checkpoint_dir} -- run train.py --stage finetune first")
    export_dir = args.runs_dir / slug / "export"
    quantized_dir, web_dir = export_dir / "quantized", export_dir / "web"

    tokenizer = common.load_tokenizer(checkpoint_dir)
    ort_model = ORTModelForSequenceClassification.from_pretrained(checkpoint_dir, export=True)
    ort_model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    print(f"exported fp32 ONNX -> {export_dir}")

    quantizer = ORTQuantizer.from_pretrained(export_dir)
    quantizer.quantize(save_dir=quantized_dir,
                       quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=False))
    tokenizer.save_pretrained(quantized_dir)
    quantized = ORTModelForSequenceClassification.from_pretrained(quantized_dir)
    print(f"quantized (dynamic int8) -> {quantized_dir}")

    shutil.rmtree(web_dir, ignore_errors=True)
    (web_dir / "onnx").mkdir(parents=True)
    tokenizer.save_pretrained(web_dir)
    ort_model.config.save_pretrained(web_dir)
    shutil.copy(quantized_dir / "model_quantized.onnx", web_dir / "onnx" / "model_quantized.onnx")
    common.write_json(web_dir / "calibration.json", refit_calibration(quantized, args, slug))
    cache = _first_cache(args.runs_dir, slug)
    common.write_json(web_dir / "parity.json", parity_fixture(quantized, tokenizer, cache, args.dataset_dir))
    print(f"browser bundle (re-fit calibration, parity fixture) -> {web_dir}")

    with common.mlflow_run("export", args.model, args.run_group):
        mlflow.log_metrics({
            "onnx_fp32_bytes": common.onnx_bytes(export_dir),
            "onnx_quantized_bytes": common.onnx_bytes(quantized_dir),
        })
        common.log_model_signature(quantized, cache["input_ids"], cache["attention_mask"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    args = parser.parse_args()
    with common.stage_span("export"):
        _run(args)


if __name__ == "__main__":
    main()
