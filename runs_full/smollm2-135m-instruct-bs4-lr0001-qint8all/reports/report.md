# JobFit benchmark report -- smollm2-135m-instruct-bs4-lr0001-qint8all

[![GitHub](https://img.shields.io/badge/GitHub-gw0%2Fjobfit--model-181717?logo=github)](https://github.com/gw0/jobfit-model)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HF-dataset-orange)](https://huggingface.co/datasets/gw0/jobfit-jevbench)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-model-yellow)](https://huggingface.co/gw0/jobfit-model)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-space-blue)](https://huggingface.co/spaces/gw0/jobfit-app)
[![Sponsor](https://img.shields.io/badge/sponsor-%E2%9D%A4-red?logo=github-sponsors)](https://github.com/sponsors/gw0)

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Git SHA: `31751c92c0a849361dea1b79fdbbc221bb8d0ad7`
- Settings: confidence_floor=0.3, batch_size=4, seed=42, epochs=5, lr=0.0001, lora_rank=8, lora_alpha=16, lora_layers=8

## Candidates (mean MAE on `test`)

| Candidate | zero-shot | fine-tuned | calibrated | quantized |
|---|---|---|---|---|
| qwen3-0.6b-qint8all | 0.2870 | 0.1309 | 0.1315 | 0.2931 |
| **smollm2-135m-instruct** (winner) | 0.3063 | 0.1545 | 0.1548 | 0.1568 |
| smollm2-135m-instruct-bs4-lr0001-qint8all | 0.3063 | 0.1915 | 0.1919 | 0.2901 |

The winner has the lowest mean MAE at its most advanced stage.

## Quality (`test`, 500 pairs)

| | Mean MAE | Beats train-mean baseline | Shuffled-pair shift |
|---|---|---|---|
| train-mean baseline | 0.2046 | | |
| keyword-overlap baseline | 0.1961 | | |
| zero-shot (base model, prompted) | 0.3063 | 3/17 | 0.0444 |
| fine-tuned (LoRA, end-to-end) | 0.1915 | 12/17 | 0.1120 |
| calibrated (temperature, pre-quantization) | 0.1919 | 12/17 | 0.1090 |
| quantized (shipped, int8) | 0.2901 | 4/17 | 0.1184 |

Shuffled-pair shift: mean |prediction change| when each test CV is paired with an unrelated JD; near 0 means the model ignores the JD.

## Per question (quantized)

| Question | MAE [90% CI] | Train-mean MAE | Spearman | Confidence vs error | Insufficient-data rate |
|---|---|---|---|---|---|
| skills_match | 0.2251 [0.2153, 0.2357] | 0.2224 | 0.1584 | -0.1495 | 0.2180 |
| experience_level_match | 0.2712 [0.2474, 0.2978] | 0.2877 | 0.0524 | -0.0840 | 0.2220 |
| domain_industry_relevance | 0.2108 [0.1931, 0.2296] | 0.1700 | 0.1144 | -0.1097 | 0.2180 |
| education_qualifications_match | 0.3597 [0.3285, 0.3940] | 0.2063 | 0.0893 | 0.1080 | 0.2380 |
| seniority_role_level_fit | 0.2211 [0.2031, 0.2391] | 0.2191 | 0.1125 | -0.0853 | 0.2960 |
| career_trajectory_fit | 0.2318 [0.2204, 0.2443] | 0.1962 | 0.1157 | -0.1940 | 0.2120 |
| responsibilities_overlap | 0.2313 [0.2186, 0.2444] | 0.2197 | 0.1286 | -0.1710 | 0.2460 |
| location_work_arrangement_fit | 0.2399 [0.2235, 0.2588] | 0.2429 | 0.1084 | -0.1286 | 0.2380 |
| soft_skills_match | 0.1901 [0.1722, 0.2089] | 0.1660 | 0.1078 | -0.0044 | 0.2640 |
| recency_of_relevant_experience | 0.2994 [0.2794, 0.3198] | 0.3492 | 0.1217 | -0.1771 | 0.2480 |
| cv_clarity_structure_quality | 0.5427 [0.5026, 0.5826] | 0.0550 | -0.0737 | 0.2671 | 0.3200 |
| job_post_clarity_structure_quality | 0.4988 [0.4726, 0.5215] | 0.0528 | -0.0056 | 0.2745 | 0.3320 |
| overall_fit_score | 0.2416 [0.2299, 0.2540] | 0.1773 | 0.1642 | -0.1216 | 0.3340 |
| cv_likely_llm_generated | 0.4457 [0.3845, 0.5072] | 0.2384 | -0.0387 | 0.0854 | 0.2480 |
| job_post_likely_llm_generated | 0.3521 [0.3318, 0.3708] | 0.3795 | 0.1556 | -0.0810 | 0.2530 |
| interview_readiness_score | 0.2299 [0.2177, 0.2428] | 0.1660 | 0.2069 | -0.2177 | 0.2460 |
| culture_company_alignment_score | 0.1410 [0.1303, 0.1524] | 0.1292 | 0.1553 | 0.0067 | 0.2200 |

## Calibration

- Shipped (fit on the quantized model): temperature 3.9630, confidence threshold 0.2742

| Stage | Spearman(confidence, abs. error) | Insufficient-data rate |
|---|---|---|
| calibrated (temperature, pre-quantization) | -0.2191 | 0.1235 |
| quantized (shipped, int8) | -0.0460 | 0.2561 |

Means over questions. The Spearman correlation should be clearly negative: the more confident an answer, the smaller its error.

## Deployability

- Quantized ONNX: 136.6 MB (fp32: 539.2 MB)

## Limitations

- The test split covers few distinct CV profiles, so candidate-axis generalisation is the weakest claim here; read the per-question table with its bootstrap CIs, not the mean as a point estimate.
