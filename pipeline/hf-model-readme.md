---
base_model: $base_model
license: $license
library_name: transformers.js
pipeline_tag: text-classification
language: en
tags:
- jobfit
- cv-job-fit-scoring
- typed-decision
- onnx
---

# $repo_id

[![GitHub](https://img.shields.io/badge/GitHub-gw0%2Fjobfit--model-181717?logo=github)](https://github.com/gw0/jobfit-model)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HF-dataset-orange)](https://huggingface.co/datasets/gw0/jobfit-jevbench)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-model-yellow)](https://huggingface.co/gw0/jobfit-model)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-space-blue)](https://huggingface.co/spaces/gw0/jobfit-app)
[![Sponsor](https://img.shields.io/badge/sponsor-%E2%9D%A4-red?logo=github-sponsors)](https://github.com/sponsors/gw0)

A CV/job-post fit-scoring typed-decision model, training pipeline, and web app -- [JobFit](https://github.com/gw0/jobfit-model). Try it in the [HF Space](https://huggingface.co/spaces/gw0/jobfit-app) with model weights from [HF Model](https://huggingface.co/gw0/jobfit-model).

This model is a *Jev-shaped typed-decision model*: a state (CV and job description) plus typed questions (score / choice / noul) in, one typed answer per question id out, each read off a causal LM's own `lm_head` in one masked forward pass. It answers the 17 JobFit questions with a 5-level score and a calibrated confidence.

- base model: `$base_model`$licence_note
- git SHA: `$git_sha`
- mean MAE (quantized, `test`): $mean_mae
- beats train-mean floor on $beats/$n_questions questions

## Usage

Ships as dynamic-int8 ONNX for [transformers.js](https://github.com/huggingface/transformers.js), loaded through the generic `PreTrainedModel` class. The graph takes `input_ids`, `segment_ids`, `answer_positions` and `candidate_ids` and returns `answer_logits`; `calibration.json` holds the model's temperature and JobFit's insufficient-data confidence threshold.

```js
import { AutoTokenizer, PreTrainedModel } from "@huggingface/transformers";
const tokenizer = await AutoTokenizer.from_pretrained("$repo_id");
const model = await PreTrainedModel.from_pretrained("$repo_id", { dtype: "q8" });
```

The input encoding and answer decoding are in the [JobFit source](https://github.com/gw0/jobfit-model) (`frontend/src/jev.mjs`, `frontend/src/infer.ts`).

## Limitations

A research benchmark, not a hiring tool: don't use it as the sole basis for decisions about people. Training CVs are synthetic and labels come from an LLM judge. The full benchmark report is in the [repository](https://github.com/gw0/jobfit-model).

## License

Copyright &copy; 2026 [gw0](https://gw.tnode.com/) &lt;<gw.2026@ena.one>&gt;. Licensed under [AGPL-3.0-or-later](https://www.gnu.org/licenses/agpl-3.0).

The weights on the HF Model carry their base model's licence (see the [model card](https://huggingface.co/gw0/jobfit-model)).

CVs are synthetic. Job posts come from public Greenhouse, Lever, Ashby and Workday pages, keep their source `url`, and remain under their publishers' rights and licenses. Not a legal opinion.

If my work saves you time, money, or tokens, consider [sponsoring](https://github.com/sponsors/gw0) the work that keeps it maintained — and tell whoever else benefits, human or agent.
