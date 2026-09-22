# JobFit benchmark report -- qwen3-0.6b

- Model: `Qwen/Qwen3-0.6B`
- Git SHA: `731456aa0a4e1dbaad4c7ad4051cc651528fb1ce`
- Winner across candidates: `qwen3-0.6b`

## Candidates (mean MAE, most advanced stage)

| Candidate | Mean MAE |
|---|---|
| qwen3-0.6b | 1.1753 |

## Quality (`test`)

| Stage | Mean MAE | Beats train-mean floor | Shuffled-pair shift |
|---|---|---|---|
| head-trained (backbone frozen) | 1.4099 | 0/17 | 1.4383 |
| fine-tuned (LoRA, end-to-end) | 1.2021 | 0/17 | 1.2377 |
| calibrated (conformal, pre-quantization) | 1.2021 | 0/17 | 1.2377 |
| quantized (shipped, int8) | 1.1753 | 0/17 | 1.1066 |

Shuffled-pair shift: mean |prediction change| when each test CV is paired with an unrelated JD; near 0 means the model ignores the JD. Per-aspect numbers and the keyword-overlap baseline are in the per-stage `eval/*.json`.

## Calibration (target coverage 0.90)

| Stage | Mean coverage | Mean interval width |
|---|---|---|
| calibrated (conformal, pre-quantization) | 0.6373 | 3.6753 |
| quantized (shipped, int8) | 0.8137 | 3.8241 |

## Deployability

- Quantized ONNX: 598.5 MB (fp32: 2385.8 MB)

Measured in a browser, not by the pipeline:

- Cold load: n/a
- Per-inference latency: WebGPU n/a (budget < 5 s), WASM n/a (budget < 30 s)
- Loads in a browser: n/a

## Drift (JevBench, informational)

- Base weights: n/a
- Post-finetune: n/a

## Limitations

- The test split covers few distinct CV profiles, so candidate-axis generalisation is the weakest claim here; read test metrics with the bootstrap CIs in the per-stage eval JSONs, not as point estimates.
