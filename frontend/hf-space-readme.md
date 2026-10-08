---
title: JobFit
emoji: 🎯
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
license: agpl-3.0
short_description: A CV/job-post fit-scoring web app, private in you browser.
models:
- gw0/jobfit-model
tags:
- transformers.js
- onnx
- cv
- jobs
---

# JobFit

[![GitHub](https://img.shields.io/badge/GitHub-gw0%2Fjobfit--model-181717?logo=github)](https://github.com/gw0/jobfit-model)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HF-dataset-orange)](https://huggingface.co/datasets/gw0/jobfit-jevbench)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-model-yellow)](https://huggingface.co/gw0/jobfit-model)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-space-blue)](https://huggingface.co/spaces/gw0/jobfit-app)
[![Sponsor](https://img.shields.io/badge/sponsor-%E2%9D%A4-red?logo=github-sponsors)](https://github.com/sponsors/gw0)

A CV/job-post fit-scoring typed-decision model, training pipeline, and web app -- [JobFit](https://github.com/gw0/jobfit-model).

Paste or upload a CV and paste a job description to get 17 scores with calibrated
confidences. Scoring runs entirely client-side (transformers.js + ONNX Runtime Web) with
the [gw0/jobfit-model](https://huggingface.co/gw0/jobfit-model) weights: nothing you paste or
upload leaves your browser.

Research benchmark, not a hiring tool. Source, pipeline, dataset and benchmark report:
[github.com/gw0/jobfit-model](https://github.com/gw0/jobfit-model) (AGPL-3.0-or-later).

## License

Copyright &copy; 2026 [gw0](https://gw.tnode.com/) &lt;<gw.2026@ena.one>&gt;. Licensed under [AGPL-3.0-or-later](https://www.gnu.org/licenses/agpl-3.0).

The weights on the HF Model carry their base model's licence (see the [model card](https://huggingface.co/gw0/jobfit-model)).

CVs are synthetic. Job posts come from public Greenhouse, Lever, Ashby and Workday pages, keep their source `url`, and remain under their publishers' rights and licenses. Not a legal opinion.

If my work saves you time, money, or tokens, consider [sponsoring](https://github.com/sponsors/gw0) the work that keeps it maintained — and tell whoever else benefits, human or agent.
