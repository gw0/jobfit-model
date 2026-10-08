# JobFit benchmark report -- qwen3-0.6b-qint8all

[![GitHub](https://img.shields.io/badge/GitHub-gw0%2Fjobfit--model-181717?logo=github)](https://github.com/gw0/jobfit-model)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HF-dataset-orange)](https://huggingface.co/datasets/gw0/jobfit-jevbench)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-model-yellow)](https://huggingface.co/gw0/jobfit-model)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-space-blue)](https://huggingface.co/spaces/gw0/jobfit-app)
[![Sponsor](https://img.shields.io/badge/sponsor-%E2%9D%A4-red?logo=github-sponsors)](https://github.com/sponsors/gw0)

- Model: `Qwen/Qwen3-0.6B`
- Git SHA: `b977f9286efd59f41fdcefbf1fa52d81f8076666`
- Settings: confidence_floor=0.3, batch_size=16, seed=42, epochs=5, lr=0.0004, lora_rank=8, lora_alpha=16, lora_layers=8

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
| zero-shot (base model, prompted) | 0.2870 | 4/17 | 0.0682 |
| fine-tuned (LoRA, end-to-end) | 0.1309 | 16/17 | 0.1243 |
| calibrated (temperature, pre-quantization) | 0.1315 | 16/17 | 0.1185 |
| quantized (shipped, int8) | 0.2931 | 3/17 | 0.0791 |

Shuffled-pair shift: mean |prediction change| when each test CV is paired with an unrelated JD; near 0 means the model ignores the JD.

## Per question (quantized)

| Question | MAE [90% CI] | Train-mean MAE | Spearman | Confidence vs error | Insufficient-data rate |
|---|---|---|---|---|---|
| skills_match | 0.2313 [0.2211, 0.2411] | 0.2224 | -0.0638 | -0.2021 | 0.1340 |
| experience_level_match | 0.2648 [0.2404, 0.2925] | 0.2877 | -0.0252 | -0.1546 | 0.1120 |
| domain_industry_relevance | 0.2035 [0.1874, 0.2198] | 0.1700 | -0.0253 | -0.2023 | 0.1060 |
| education_qualifications_match | 0.3509 [0.3221, 0.3798] | 0.2063 | 0.0028 | 0.1654 | 0.1260 |
| seniority_role_level_fit | 0.2284 [0.2145, 0.2424] | 0.2191 | -0.0923 | -0.1035 | 0.1420 |
| career_trajectory_fit | 0.2320 [0.2216, 0.2423] | 0.1962 | -0.0097 | -0.1976 | 0.1620 |
| responsibilities_overlap | 0.2390 [0.2301, 0.2483] | 0.2197 | -0.0357 | -0.2017 | 0.1360 |
| location_work_arrangement_fit | 0.2447 [0.2283, 0.2638] | 0.2429 | -0.0170 | -0.1635 | 0.1160 |
| soft_skills_match | 0.2073 [0.1908, 0.2238] | 0.1660 | -0.0754 | 0.0234 | 0.1200 |
| recency_of_relevant_experience | 0.3042 [0.2884, 0.3208] | 0.3492 | -0.0766 | -0.1817 | 0.1060 |
| cv_clarity_structure_quality | 0.5645 [0.5336, 0.5949] | 0.0550 | -0.0755 | 0.3819 | 0.1360 |
| job_post_clarity_structure_quality | 0.5290 [0.5199, 0.5376] | 0.0528 | 0.0490 | 0.4332 | 0.1220 |
| overall_fit_score | 0.2161 [0.2075, 0.2255] | 0.1773 | -0.0296 | -0.2034 | 0.1100 |
| cv_likely_llm_generated | 0.4325 [0.3736, 0.4886] | 0.2384 | -0.0303 | 0.2307 | 0.1680 |
| job_post_likely_llm_generated | 0.3660 [0.3485, 0.3821] | 0.3795 | -0.0665 | -0.0068 | 0.1446 |
| interview_readiness_score | 0.2341 [0.2198, 0.2489] | 0.1660 | -0.0045 | -0.2113 | 0.1740 |
| culture_company_alignment_score | 0.1345 [0.1208, 0.1472] | 0.1292 | -0.0571 | -0.1160 | 0.1300 |

## Calibration

- Shipped (fit on the quantized model): temperature 20.0854, confidence threshold 0.3014

| Stage | Spearman(confidence, abs. error) | Insufficient-data rate |
|---|---|---|
| calibrated (temperature, pre-quantization) | -0.2894 | 0.1350 |
| quantized (shipped, int8) | -0.0418 | 0.1320 |

Means over questions. The Spearman correlation should be clearly negative: the more confident an answer, the smaller its error.

## Deployability

- Quantized ONNX: 598.4 MB (fp32: 2385.6 MB)

## Limitations

- The test split covers few distinct CV profiles, so candidate-axis generalisation is the weakest claim here; read the per-question table with its bootstrap CIs, not the mean as a point estimate.
