# JobFit benchmark report -- smollm2-135m-instruct

[![GitHub](https://img.shields.io/badge/GitHub-gw0%2Fjobfit--model-181717?logo=github)](https://github.com/gw0/jobfit-model)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HF-dataset-orange)](https://huggingface.co/datasets/gw0/jobfit-jevbench)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-model-yellow)](https://huggingface.co/gw0/jobfit-model)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-space-blue)](https://huggingface.co/spaces/gw0/jobfit-app)
[![Sponsor](https://img.shields.io/badge/sponsor-%E2%9D%A4-red?logo=github-sponsors)](https://github.com/sponsors/gw0)

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Git SHA: `e418ab5eae994a5bc2cf14345d5e4822d23d9d08`
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
| zero-shot (base model, prompted) | 0.3063 | 3/17 | 0.0444 |
| fine-tuned (LoRA, end-to-end) | 0.1545 | 15/17 | 0.1267 |
| calibrated (temperature, pre-quantization) | 0.1548 | 15/17 | 0.1242 |
| quantized (shipped, int8) | 0.1568 | 15/17 | 0.1345 |

Shuffled-pair shift: mean |prediction change| when each test CV is paired with an unrelated JD; near 0 means the model ignores the JD.

## Per question (quantized)

| Question | MAE [90% CI] | Train-mean MAE | Spearman | Confidence vs error | Insufficient-data rate |
|---|---|---|---|---|---|
| skills_match | 0.1479 [0.1354, 0.1608] | 0.2224 | 0.5981 | -0.3837 | 0.0640 |
| experience_level_match | 0.2161 [0.1939, 0.2400] | 0.2877 | 0.5505 | -0.3232 | 0.4520 |
| domain_industry_relevance | 0.1485 [0.1332, 0.1665] | 0.1700 | 0.4321 | -0.1562 | 0.0260 |
| education_qualifications_match | 0.1688 [0.1553, 0.1824] | 0.2063 | 0.5046 | -0.2039 | 0.1060 |
| seniority_role_level_fit | 0.1847 [0.1683, 0.2007] | 0.2191 | 0.4883 | -0.1802 | 0.0980 |
| career_trajectory_fit | 0.1485 [0.1363, 0.1616] | 0.1962 | 0.5417 | -0.3592 | 0.0480 |
| responsibilities_overlap | 0.1538 [0.1438, 0.1647] | 0.2197 | 0.5491 | -0.3004 | 0.0980 |
| location_work_arrangement_fit | 0.2135 [0.1902, 0.2436] | 0.2429 | 0.3050 | 0.0110 | 0.5420 |
| soft_skills_match | 0.1439 [0.1343, 0.1541] | 0.1660 | 0.5314 | -0.1076 | 0.0020 |
| recency_of_relevant_experience | 0.2256 [0.2075, 0.2446] | 0.3492 | 0.5469 | -0.4296 | 0.5180 |
| cv_clarity_structure_quality | 0.0740 [0.0578, 0.0918] | 0.0550 | 0.4935 | 0.1311 | 0.0000 |
| job_post_clarity_structure_quality | 0.0593 [0.0562, 0.0620] | 0.0528 | 0.2478 | 0.3006 | 0.0000 |
| overall_fit | 0.1326 [0.1230, 0.1432] | 0.1773 | 0.5494 | -0.3533 | 0.0020 |
| cv_likely_llm_generated | 0.1249 [0.0943, 0.1583] | 0.2384 | 0.7119 | -0.5336 | 0.2920 |
| job_post_likely_llm_generated | 0.2894 [0.2719, 0.3045] | 0.3795 | 0.4157 | -0.3897 | 0.5639 |
| interview_readiness | 0.1178 [0.1070, 0.1288] | 0.1660 | 0.5540 | -0.4983 | 0.0020 |
| culture_company_alignment | 0.1164 [0.1082, 0.1249] | 0.1292 | 0.4389 | 0.0516 | 0.0000 |

## Calibration

- Shipped (fit on the quantized model): temperature 1.1337, confidence threshold 0.4279

| Stage | Spearman(confidence, abs. error) | Insufficient-data rate |
|---|---|---|
| calibrated (temperature, pre-quantization) | -0.2502 | 0.1553 |
| quantized (shipped, int8) | -0.2191 | 0.1655 |

Means over questions. The Spearman correlation should be clearly negative: the more confident an answer, the smaller its error.

## Deployability

- Quantized ONNX: 136.6 MB (fp32: 539.2 MB)

## Limitations

- The test split covers few distinct CV profiles, so candidate-axis generalisation is the weakest claim here; read the per-question table with its bootstrap CIs, not the mean as a point estimate.
