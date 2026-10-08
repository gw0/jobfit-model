# JobFit

[![GitHub](https://img.shields.io/badge/GitHub-gw0%2Fjobfit--model-181717?logo=github)](https://github.com/gw0/jobfit-model)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HF-dataset-orange)](https://huggingface.co/datasets/gw0/jobfit-jevbench)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-model-yellow)](https://huggingface.co/gw0/jobfit-model)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-space-blue)](https://huggingface.co/spaces/gw0/jobfit-app)
[![CI](https://img.shields.io/github/actions/workflow/status/gw0/jobfit-model/test.yml?label=tests)](https://github.com/gw0/jobfit-model/actions/workflows/test.yml)
[![Sponsor](https://img.shields.io/badge/sponsor-%E2%9D%A4-red?logo=github-sponsors)](https://github.com/sponsors/gw0)

A CV/job-post fit-scoring typed-decision model, training pipeline, and web app.

JobFit answers 17 questions about a CV and a job description ([`questions.json`](questions.json)), each with a score and a calibrated confidence. All scoring runs client-side in the browser: no inference API, accounts, or data collection. Try it in the [HF Space](https://huggingface.co/spaces/gw0/jobfit-app) which loads weights from [HF Model](https://huggingface.co/gw0/jobfit-model).

This model is a *Jev-shaped typed-decision model*: a state (CV and job description) plus typed questions (score / choice / noul) in, one typed answer per question id out, each read off a causal LM's own `lm_head` in one masked forward pass. It answers the 17 JobFit questions with a 5-level score and a calibrated confidence. The core is `pipeline/jev.py`, `pipeline/jev_model.py` and the exact JS mirror `frontend/src/jev.mjs`; the normative design is in [`specs/`](specs/).

## Results

Winner of the full benchmark (`runs_full/smollm2-135m-instruct`, SmolLM2-135M-Instruct + LoRA, int8 ONNX, 136.6 MB). Mean MAE over the 17 questions on 500 test pairs, lower is better:

| | Mean MAE | Beats train-mean baseline |
|---|---|---|
| train-mean baseline | 0.2046 | |
| keyword-overlap baseline | 0.1961 | |
| zero-shot (base model, prompted) | 0.3063 | 3/17 |
| fine-tuned | 0.1545 | 15/17 |
| quantized (shipped) | 0.1568 | 15/17 |

The test split covers few distinct CV profiles. See the [full report](runs_full/smollm2-135m-instruct/reports/report.md).

## Layout

- `data/` builds the dataset offline: job posts, synthetic CVs, PII scan, splits, LLM-judge labels. One corpus per scale: `datasets_smoke/` (10 CVs, exercises the logic) and `datasets_full/` (250 CVs, the real benchmark), kept in [gw0/jobfit-jevbench](https://github.com/gw0/jobfit-jevbench). See [`data/README.md`](data/README.md).
- `pipeline/` is the training pipeline, one script per stage: `prepare -> evaluate (zeroshot) -> train -> evaluate -> calibrate -> evaluate -> export -> evaluate -> publish`. Output goes to `runs_<scale>/<candidate>/`; only `reports/{report.json,report.md}` is committed there.
- `frontend/` is the Vue 3 + Vite + Bulma SPA; inference uses transformers.js.
- `infra/` and `docker/` hold the KinD cluster (Argo, MLflow, OTel collector) and the images.

## Setup

Requires Python 3.13, Node 22 (or bun) and Docker (pipeline targets).

```sh
make venv   # .venv: Python requirements, spaCy model, pinned argo/kind/kubectl
npm --prefix frontend install
```

Every `make` target below takes three knobs:

- `SCALE=smoke|full` picks the corpus `datasets_$SCALE/` and output `runs_$SCALE/` (default `smoke`).
- `CANDIDATE=<name>` names one pipeline run, `runs_$SCALE/<name>/`: the model slug (`smollm2-135m-instruct`) for default settings, plus a suffix per variant (`-r16`). Settings are recorded in the run's `config.json`; `publish` picks the winner across candidates.
- `GPU=1` runs on NVIDIA GPUs instead of CPU. Weights stay fp32 either way.

## Dataset

```sh
git clone https://github.com/gw0/jobfit-jevbench ../jobfit-jevbench
make jobs cvs dataset labels                # SCALE=smoke: a no-op on the committed corpus
make jobs cvs dataset labels SCALE=full     # resumable; see data/README.md for size and cost
```

`data/generate_cvs.py` and `data/label_dataset.py` call the `claude` CLI, so it must be authenticated in your shell; a nested call from inside another Claude Code session does not inherit that session's auth. Never edit a corpus in place: a change is a new tag, and each report's git SHA says which corpus it used.

## Pipeline

Locally, each stage runs in Docker:

```sh
make local-publish                                  # build the image, then prepare ... publish
make local-publish SCALE=full GPU=1 CANDIDATE=smollm2-135m-instruct
```

On a KinD cluster, `make cluster-run` runs the same DAG as one Argo workflow per candidate. `cluster-up` fixes the scale (it mounts `datasets_$SCALE/` and `runs_$SCALE/`) and port-forwards the MLflow and Argo UIs; pass the same `SCALE` to every `cluster-run`:

```sh
make cluster-up [SCALE=full] [GPU=1]    # GPU: infra/gpu/setup-node.sh
make cluster-run ARGO_PARAMS="-p lora-rank=16" CANDIDATE=smollm2-135m-instruct-r16
make cluster-check                      # list runs
make cluster-down
```

Without `MLFLOW_TRACKING_URI`, runs go to a local `runs_<scale>/mlflow-local/` store; the cluster's MLflow keeps its store in `runs_<scale>/mlflow/`.

For a production run, run one candidate at a time at a tagged commit (every model at default settings first, then variants of the best one or two), refresh the reports so each names the final winner, commit `runs_full/*/reports/` and push the winner:

```sh
for c in $(ls runs_full); do ./pipeline/publish.py --runs-dir runs_full --candidate $c; done
make push-hf-model SCALE=full CANDIDATE=<winner>
```

## Frontend

```sh
make copy-model                  # serve runs_<scale>/<candidate>/export/web at /models/default/
npm --prefix frontend run dev
node frontend/scripts/verify-parity.mjs frontend/public/models/default
```

`verify-parity.mjs` checks that transformers.js reproduces the pipeline's encoding, answer logits and answers for the sample state in `parity.json`. The model is chosen at deploy time: the app fetches `config.json`, which holds `JOBFIT_MODEL_URL` (an HF Model `resolve/<revision>` URL, or a path such as `/models/default`). This repository ships no weights.

## Publishing

Secrets live in gitignored `KEY=VALUE` files (`chmod 600`), one per scope, never on a command line:

| File | Contents | Used by |
|---|---|---|
| `.env.local` | `HF_TOKEN` (read) | `local-*` Docker targets |
| `.env.cluster` | `HF_TOKEN` (read) | `make cluster-secrets` (run by `cluster-up`; rerun to rotate) creates the `hf-token` Secret that stage pods read |
| `.env.publish` | `HF_TOKEN` (write); optional `HF_MODEL_REPO` (default `gw0/jobfit-model`), `HF_SPACE_REPO` (default `gw0/jobfit-app`), `WANDB_API_KEY` | `push-hf-model`, `push-hf-frontend`, `push-wandb-report` |

Only `.env.publish` is needed to publish; public models need no token otherwise.

```sh
make push-hf-model SCALE=full CANDIDATE=<winner>      # winner's bundle + model card to HF_MODEL_REPO
make push-hf-frontend                                 # SPA to the HF_SPACE_REPO Space, loading that model
make push-wandb-report SCALE=full CANDIDATE=<winner>  # benchmark report to W&B
```

The model card is generated by `pipeline/publish.py`; the Space README is `frontend/hf-space-readme.md`.

## Tests

```sh
make test    # hermetic: pytest + frontend type-check/build/vitest (what CI runs)
make smoke   # live: job fetching, claude labeling, Dockerized pipeline, parity check, KinD/Argo
```

Tests that need torch or transformers skip themselves when those packages are missing. Set `NPM=bun` to use bun instead of npm.

## Limitations

JobFit is a research benchmark, not a hiring tool: don't use it as the sole basis for decisions about people. Training CVs are synthetic and labels come from an LLM judge, so scores reflect that judge's view, not recruiter ground truth.

## License

Copyright &copy; 2026 [gw0](https://gw.tnode.com/) &lt;<gw.2026@ena.one>&gt;. Licensed under [AGPL-3.0-or-later](https://www.gnu.org/licenses/agpl-3.0).

The weights on the HF Model carry their base model's licence (see the [model card](https://huggingface.co/gw0/jobfit-model)).

CVs are synthetic. Job posts come from public Greenhouse, Lever, Ashby and Workday pages, keep their source `url`, and remain under their publishers' rights and licenses. Not a legal opinion.

If my work saves you time, money, or tokens, consider [sponsoring](https://github.com/sponsors/gw0) the work that keeps it maintained — and tell whoever else benefits, human or agent.
