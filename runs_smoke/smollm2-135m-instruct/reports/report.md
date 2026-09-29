# JobFit benchmark report -- smollm2-135m-instruct

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Git SHA: `8cee385b818b495f3680e28662f6787896f08355`
- Winner across candidates: `smollm2-135m-instruct`

## Settings

- prepare: model=HuggingFaceTB/SmolLM2-135M-Instruct, candidate=smollm2-135m-instruct, confidence_floor=0.3, batch_size=4, seed=42
- finetune: model=HuggingFaceTB/SmolLM2-135M-Instruct, candidate=smollm2-135m-instruct, confidence_floor=0.3, batch_size=4, seed=42, epochs=5, lr=0.0001, lora_rank=8, lora_alpha=16, lora_layers=8, lora_target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
- calibrate: model=HuggingFaceTB/SmolLM2-135M-Instruct, candidate=smollm2-135m-instruct, confidence_floor=0.3, batch_size=4, seed=42
- export: model=HuggingFaceTB/SmolLM2-135M-Instruct, candidate=smollm2-135m-instruct, confidence_floor=0.3, batch_size=4, seed=42

## Candidates (mean MAE, most advanced stage)

| Candidate | Mean MAE |
|---|---|
| smollm2-135m-instruct | 0.2669 |

## Quality (`test`)

| Stage | Mean MAE | Beats train-mean floor | Shuffled-pair shift |
|---|---|---|---|
| zero-shot (base model, prompted) | 0.3469 | 12/17 | 0.0427 |
| fine-tuned (LoRA, end-to-end) | 0.2695 | 12/17 | 0.0687 |
| calibrated (temperature, pre-quantization) | 0.2638 | 11/17 | 0.0037 |
| quantized (shipped, int8) | 0.2669 | 11/17 | 0.0075 |

Shuffled-pair shift: mean |prediction change| when each test CV is paired with an unrelated JD; near 0 means the model ignores the JD. Per-question numbers and the keyword-overlap baseline are in the per-stage `eval/*.json`.

## Calibration

- Shipped (fit on the quantized model): temperature 20.0854, confidence threshold 0.2864

| Stage | Spearman(confidence, abs. error) | Insufficient-data rate |
|---|---|---|
| calibrated (temperature, pre-quantization) | -0.0882 | 0.1225 |
| quantized (shipped, int8) | -0.3647 | 0.0000 |

Means over questions. The Spearman correlation should be clearly negative: the more confident an answer, the smaller its error.

## Deployability

- Quantized ONNX: 136.6 MB (fp32: 539.2 MB)

Measured in a browser, not by the pipeline:

- Cold load: n/a
- Per-inference latency: WebGPU n/a (budget < 6 s), WASM n/a (budget < 40 s)
- Loads in a browser: n/a

## Drift (JevBench, informational)

- Base weights: n/a
- Post-finetune: n/a

## Limitations

- The test split covers few distinct CV profiles, so candidate-axis generalisation is the weakest claim here; read test metrics with the bootstrap CIs in the per-stage eval JSONs, not as point estimates.
