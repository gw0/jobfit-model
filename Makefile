# make test    hermetic: every pytest suite. No network beyond package installs,
#              no credentials -- this is what CI runs.
# make build   the pipeline Docker image.
# smoke-*      the live chain against datasets_smoke/: job fetching, `claude -p` CV
#              generation and labeling (needs an authenticated `claude` CLI), the
#              dataset build and the Dockerized pipeline.

.PHONY: venv build test smoke-fetch-jobs smoke-cvs smoke-dataset smoke-labels smoke-prepare

VENV ?= .venv/bin
PY ?= $(VENV)/python
MODEL ?= Qwen/Qwen3-0.6B

# Entry-point scripts use `#!/usr/bin/env python3`; resolve that to the venv.
export PATH := $(CURDIR)/.venv/bin:$(PATH)

venv:
	python3 -m venv .venv
	$(PY) -m pip install -q pytest -r data/requirements.txt -r pipeline/requirements.txt
	$(PY) -m spacy download en_core_web_sm

build:
	docker build -t jobfit-pipeline -f docker/pipeline.Dockerfile .

test:
	$(PY) -m pytest -q
	@! grep -n "Mozilla" data/fetch_jobs/*.py || { echo "spoofed browser User-Agent in data/fetch_jobs" >&2; exit 1; }

# --- dataset toolset -----------------------------------------------------------------

smoke-fetch-jobs:
	./data/fetch_jobs/fetch_greenhouse.py grafanalabs --out-dir $$(mktemp -d) --include engineer

# --count 10: the 70/10/10/10 split ratios need 10 CVs before val/calib/test are non-empty.

smoke-cvs:
	./data/generate_cvs.py --count 10 --out-dir datasets_smoke

smoke-dataset:
	./data/build_dataset.py --out-dir datasets_smoke

smoke-labels:
	./data/label_dataset.py --out-dir datasets_smoke --double-label --double-label-sample 3

# --- pipeline (Docker, CPU) -----------------------------------------------------------
# The corpus is mounted read-only; every stage writes only into runs_smoke/.

DOCKER_RUN = docker run --rm -v $(CURDIR)/datasets_smoke:/data:ro -v $(CURDIR)/runs_smoke:/runs jobfit-pipeline
STAGE_ARGS = --dataset-dir /data --runs-dir /runs --model $(MODEL)

smoke-prepare: build
	mkdir -p runs_smoke
	$(DOCKER_RUN) ./pipeline/prepare.py $(STAGE_ARGS)
