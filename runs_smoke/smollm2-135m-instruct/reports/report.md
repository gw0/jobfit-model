# JobFit benchmark report -- smollm2-135m-instruct

[![GitHub](https://img.shields.io/badge/GitHub-gw0%2Fjobfit--model-181717?logo=github)](https://github.com/gw0/jobfit-model)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HF-dataset-orange)](https://huggingface.co/datasets/gw0/jobfit-jevbench)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-model-yellow)](https://huggingface.co/gw0/jobfit-model)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-space-blue)](https://huggingface.co/spaces/gw0/jobfit-app)
[![Sponsor](https://img.shields.io/badge/sponsor-%E2%9D%A4-red?logo=github-sponsors)](https://github.com/sponsors/gw0)

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Git SHA: `c96549259af3eba667181729e62e03509bf72eea`
- Settings: confidence_floor=0.3, batch_size=16, seed=42, epochs=5, lr=0.0004, lora_rank=8, lora_alpha=16, lora_layers=8

## Candidates (mean MAE on `test`)

| Candidate | zero-shot | fine-tuned | calibrated | quantized |
|---|---|---|---|---|
| **smollm2-135m-instruct** (winner) | 0.3469 | 0.2695 | 0.2638 | 0.2655 |

The winner has the lowest mean MAE at its most advanced stage.

## Quality (`test`, 4 pairs)

| | Mean MAE | Beats train-mean baseline | Shuffled-pair shift |
|---|---|---|---|
| train-mean baseline | 0.2563 | | |
| keyword-overlap baseline | 0.2731 | | |
| zero-shot (base model, prompted) | 0.3469 | 12/17 | 0.0427 |
| fine-tuned (LoRA, end-to-end) | 0.2695 | 12/17 | 0.0696 |
| calibrated (temperature, pre-quantization) | 0.2638 | 11/17 | 0.0040 |
| quantized (shipped, int8) | 0.2655 | 11/17 | 0.0035 |

Shuffled-pair shift: mean |prediction change| when each test CV is paired with an unrelated JD; near 0 means the model ignores the JD.

## Per question (quantized)

| Question | MAE [90% CI] | Train-mean MAE | Spearman | Confidence vs error | Insufficient-data rate |
|---|---|---|---|---|---|
| skills_match | 0.1558 [n/a, n/a] | 0.2798 | 0.2000 | -0.4000 | 0.7500 |
| experience_level_match | 0.2549 [n/a, n/a] | 0.3646 | -0.4000 | -0.4000 | 0.0000 |
| domain_industry_relevance | 0.1922 [n/a, n/a] | 0.3780 | 0.2108 | 0.4000 | 0.2500 |
| education_qualifications_match | 0.2568 [n/a, n/a] | 0.1071 | n/a | -0.2000 | 0.2500 |
| seniority_role_level_fit | 0.0937 [n/a, n/a] | 0.2366 | 0.3162 | 0.8000 | 0.2500 |
| career_trajectory_fit | 0.1885 [n/a, n/a] | 0.3065 | -0.8000 | 0.8000 | 0.2500 |
| responsibilities_overlap | 0.2187 [n/a, n/a] | 0.3348 | 0.4000 | 0.4000 | 0.5000 |
| location_work_arrangement_fit | 0.3098 [n/a, n/a] | 0.3125 | 0.6000 | 0.4000 | 0.2500 |
| soft_skills_match | 0.2526 [n/a, n/a] | 0.3274 | -0.3162 | -0.8000 | 1.0000 |
| recency_of_relevant_experience | 0.4127 [n/a, n/a] | 0.5506 | -0.6000 | -0.4000 | 0.0000 |
| cv_clarity_structure_quality | 0.5024 [n/a, n/a] | 0.0714 | n/a | -0.6000 | 1.0000 |
| job_post_clarity_structure_quality | 0.4082 [n/a, n/a] | 0.0938 | 0.2582 | 0.4000 | 0.0000 |
| overall_fit | 0.2141 [n/a, n/a] | 0.2247 | 0.8000 | -0.8000 | 0.2500 |
| cv_likely_llm_generated | 0.3688 [n/a, n/a] | 0.0474 | n/a | -0.4000 | 1.0000 |
| job_post_likely_llm_generated | 0.3705 [n/a, n/a] | 0.3374 | 0.5000 | 1.0000 | 1.0000 |
| interview_readiness | 0.2471 [n/a, n/a] | 0.2113 | 0.2108 | -0.4000 | 0.2500 |
| culture_company_alignment | 0.0664 [n/a, n/a] | 0.1726 | -0.6325 | 0.2000 | 0.2500 |

## Calibration

- Shipped (fit on the quantized model): temperature 20.0854, confidence threshold 0.2854

| Stage | Spearman(confidence, abs. error) | Insufficient-data rate |
|---|---|---|
| calibrated (temperature, pre-quantization) | -0.1118 | 0.1225 |
| quantized (shipped, int8) | -0.0000 | 0.4265 |

Means over questions. The Spearman correlation should be clearly negative: the more confident an answer, the smaller its error.

## Deployability

- Quantized ONNX: 136.6 MB (fp32: 539.2 MB)

## Limitations

- The test split covers few distinct CV profiles, so candidate-axis generalisation is the weakest claim here; read the per-question table with its bootstrap CIs, not the mean as a point estimate.
