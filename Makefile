# make test    hermetic: every pytest suite plus the frontend type-check/build/vitest.
#              No network beyond package installs, no credentials -- this is what CI runs.
# make smoke   the live chain at SCALE=smoke: job fetching, `claude -p` CV generation and
#              labeling (needs an authenticated `claude` CLI), the Dockerized pipeline, the
#              frontend parity check and the KinD/Argo run.
#
# Every other target works at either scale and on either device:
#   SCALE=smoke|full   corpus datasets_$(SCALE)/, pipeline output runs_$(SCALE)/
#   MODEL=<hf-id>      base model
#   CANDIDATE=<name>   runs_$(SCALE)/<name>/: the model slug for default settings, else the
#                      slug plus what changed, e.g. smollm2-135m-instruct-r16 (config.json has the rest)
#   GPU=1              run the pipeline on NVIDIA GPUs (Docker --gpus, or a GPU KinD cluster)
# Dataset steps: jobs cvs dataset labels. Pipeline steps run either locally in Docker,
# local-{prepare,zeroshot,finetune,calibrate,export,publish} (each depending on the
# previous one), or on the KinD cluster as one Argo workflow, cluster-run.

.PHONY: venv build test smoke fetch-check \
	jobs cvs dataset labels \
	local-prepare local-zeroshot local-finetune local-calibrate local-export local-publish \
	copy-model frontend-check \
	cluster-up cluster-secrets cluster-down cluster-check cluster-logs cluster-mlflow cluster-run \
	push-hf-model push-hf-frontend push-wandb-report

VENV ?= .venv/bin
PY ?= $(VENV)/python
NPM ?= npm

SCALE ?= smoke
DATASET := datasets_$(SCALE)
RUNS := runs_$(SCALE)
MODEL ?= HuggingFaceTB/SmolLM2-135M-Instruct
CANDIDATE ?= $(shell echo '$(notdir $(MODEL))' | tr A-Z a-z)
GPU ?= 0

# Corpus size per scale (specs §5). Smoke's 10 CVs are the fewest for which the
# 70/10/10/10 split ratios leave val/calib/test non-empty; it pairs every CV with
# every job of its pool. Full samples JOBS_PER_CV per CV instead of all ~400, 20 so
# that calib (25 CVs) still gets the >= 500 pairs specs §5 asks for.
ifeq ($(SCALE),full)
CVS ?= 250
JOBS ?= 400
JOBS_PER_COMPANY ?= 10
JOBS_PER_CV ?= 20
DOUBLE_LABEL_SAMPLE ?= 300
WORKERS ?= 8
else
CVS ?= 10
JOBS ?= 10
DOUBLE_LABEL_SAMPLE ?= 3
WORKERS ?= 1
endif

# Jobs judged per CV in one labeling call (pairs.jsonl only; the double-label QC pass
# always uses one job per call).
JOBS_PER_CALL ?= 10

# Entry-point scripts use `#!/usr/bin/env python3`; resolve that to the venv.
export PATH := $(CURDIR)/.venv/bin:$(PATH)

# Cluster CLIs go into the venv too, pinned: argo matches the controller in
# infra/argo/kustomization.yaml, kind's release decides the node's Kubernetes version,
# kubectl matches that node. socat (the cluster-up tunnel) and docker come from the system.
ARGO_VERSION ?= v3.6.2
KIND_VERSION ?= v0.27.0
KUBECTL_VERSION ?= v1.32.2
OS := $(shell uname -s | tr A-Z a-z)
ARCH := $(shell uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/')

venv:
	python3 -m venv .venv
	$(PY) -m pip install -q pytest scipy -r data/requirements.txt -r pipeline/requirements.txt
	$(PY) -m spacy download en_core_web_sm
	curl -fsSL https://github.com/argoproj/argo-workflows/releases/download/$(ARGO_VERSION)/argo-$(OS)-$(ARCH).gz | gunzip > $(VENV)/argo
	curl -fsSLo $(VENV)/kind https://kind.sigs.k8s.io/dl/$(KIND_VERSION)/kind-$(OS)-$(ARCH)
	curl -fsSLo $(VENV)/kubectl https://dl.k8s.io/release/$(KUBECTL_VERSION)/bin/$(OS)/$(ARCH)/kubectl
	chmod +x $(VENV)/argo $(VENV)/kind $(VENV)/kubectl

build:
	docker build -t jobfit-pipeline -f docker/pipeline.Dockerfile --build-arg GIT_SHA=$$(git rev-parse HEAD) .
	docker build -t jobfit-frontend -f docker/frontend.Dockerfile .

test:
	$(PY) -m pytest -q
	@! grep -n "Mozilla" data/fetch_jobs/*.py || { echo "spoofed browser User-Agent in data/fetch_jobs" >&2; exit 1; }
	cd frontend && $(NPM) install && $(NPM) run build && $(NPM) run test

smoke: test
	$(MAKE) SCALE=smoke fetch-check cvs dataset labels local-publish frontend-check cluster-run

# A live fetch into a throwaway directory: the committed smoke corpus already has its jobs.
fetch-check:
	./data/fetch_jobs/fetch_greenhouse.py grafanalabs --out-dir $$(mktemp -d) --include engineer

# --- dataset toolset -----------------------------------------------------------------
# Each step keeps what already exists (jobs/CVs up to the target count, finished labels),
# so an interrupted step is resumed by re-running it. Labeling re-spends real judge
# calls: never relabel a committed corpus in place.

jobs:
	./data/fetch_jobs/fetch_all.py --out-dir $(DATASET)/jobs --count $(JOBS) \
		$(if $(JOBS_PER_COMPANY),--max-per-company $(JOBS_PER_COMPANY))

cvs:
	./data/generate_cvs.py --count $(CVS) --out-dir $(DATASET) --workers $(WORKERS)

dataset:
	./data/build_dataset.py --out-dir $(DATASET) $(if $(JOBS_PER_CV),--jobs-per-cv $(JOBS_PER_CV))

labels:
	./data/label_dataset.py --out-dir $(DATASET) --workers $(WORKERS) --jobs-per-call $(JOBS_PER_CALL) \
		--double-label-sample $(DOUBLE_LABEL_SAMPLE)

# --- pipeline, local (Docker) -------------------------------------------------------
# The corpus is mounted read-only; every stage writes only into $(RUNS)/$(CANDIDATE)/.

DOCKER_RUN = docker run --rm $(if $(filter 1,$(GPU)),--gpus all) $(if $(wildcard .env.local),--env-file .env.local) \
	-v $(CURDIR)/$(DATASET):/datasets:ro -v $(CURDIR)/$(RUNS):/runs -v $(CURDIR)/.cache:/cache jobfit-pipeline
STAGE_ARGS = --datasets-dir /datasets --runs-dir /runs --model $(MODEL) --candidate $(CANDIDATE)

local-prepare: build
	mkdir -p $(RUNS)
	$(DOCKER_RUN) ./pipeline/prepare.py $(STAGE_ARGS)

local-zeroshot: local-prepare
	$(DOCKER_RUN) ./pipeline/evaluate.py --stage zeroshot $(STAGE_ARGS)

local-finetune: local-zeroshot
	$(DOCKER_RUN) ./pipeline/train.py $(STAGE_ARGS)
	$(DOCKER_RUN) ./pipeline/evaluate.py --stage finetuned $(STAGE_ARGS)

local-calibrate: local-finetune
	$(DOCKER_RUN) ./pipeline/calibrate.py $(STAGE_ARGS)
	$(DOCKER_RUN) ./pipeline/evaluate.py --stage calibrated $(STAGE_ARGS)

local-export: local-calibrate
	$(DOCKER_RUN) ./pipeline/export.py $(STAGE_ARGS)
	$(DOCKER_RUN) ./pipeline/evaluate.py --stage quantized $(STAGE_ARGS)

local-publish: local-export
	$(DOCKER_RUN) ./pipeline/publish.py $(STAGE_ARGS)

# --- frontend --------------------------------------------------------------------------

# Serves a pipeline export locally at /models/default/ (frontend/public/models is gitignored).
copy-model:
	rm -rf frontend/public/models/default
	mkdir -p frontend/public/models
	cp -r $(RUNS)/$(CANDIDATE)/export/web frontend/public/models/default

frontend-check: local-export copy-model build
	cd frontend && $(NPM) install
	node frontend/scripts/verify-parity.mjs frontend/public/models/default
	docker run --rm -d --name $(FRONTEND_CHECK_NAME) jobfit-frontend
	for i in 1 2 3 4 5; do \
		docker exec $(FRONTEND_CHECK_NAME) wget -qO /dev/null http://127.0.0.1:8080/ && status=0 && break; \
		status=$$?; sleep 1; \
	done; \
	[ $$status != 0 ] || echo "frontend served OK"; \
	docker stop $(FRONTEND_CHECK_NAME); exit $$status

# --- cluster (KinD + Argo + MLflow + OTel collector) ------------------------------------
# kind-config.yaml mounts $(DATASET) and $(RUNS) when the cluster is created, so
# `make cluster-up SCALE=... [GPU=1]` fixes the scale, then one
# `make cluster-run SCALE=... CANDIDATE=...` per candidate (same SCALE). The committed
# runs_*/<candidate>/reports/ is the durable cross-host comparison; MLflow's store,
# $(RUNS)/mlflow/ (local runs use $(RUNS)/mlflow-local/), is not.

# Two checkouts of this repo on the same host must not collide: derive a short hash from
# the checkout path and use it for the cluster name and every host port/name below, so
# each checkout gets its own by default -- still overridable via these ?= vars. Ports are
# a block of 10 starting at PORT_BASE (10000..19990, below the ephemeral range).
REPO_HASH := $(shell printf '%s' "$(CURDIR)" | md5sum | cut -c1-4)
PORT_BASE := $(shell echo $$((10000 + 10 * (0x$(REPO_HASH) % 1000))))

KIND_CLUSTER ?= jobfit-$(REPO_HASH)
TUNNEL_PORT ?= $(PORT_BASE)
MLFLOW_PORT ?= $(shell echo $$(($(PORT_BASE) + 1)))
ARGO_PORT ?= $(shell echo $$(($(PORT_BASE) + 2)))
FRONTEND_CHECK_NAME := jobfit-frontend-check-$(REPO_HASH)
KIND_DIR := .kind
export KUBECONFIG := $(CURDIR)/$(KIND_DIR)/kubeconfig.yaml

# kind's Docker daemon may not share our network namespace, so its published API port
# can be unreachable. Tunnel over the Docker socket instead: socat relays each
# connection through an alpine/socat container on the "kind" network. socat splits
# its address on every ':' inside EXEC:"...", hence the "\:" escapes.
# GPU=1 also readies the node's containerd and the NVIDIA device plugin (infra/gpu/).
cluster-up:
	mkdir -p $(RUNS)/mlflow .cache
	sed 's#@DATASET@#$(DATASET)#; s#@RUNS@#$(RUNS)#' infra/kind-config.yaml | kind create cluster --name $(KIND_CLUSTER) --config -
	kind get kubeconfig --name $(KIND_CLUSTER) | sed 's#server: https://127.0.0.1:[0-9]*#server: https://127.0.0.1:$(TUNNEL_PORT)#' > $(KIND_DIR)/kubeconfig.yaml
	socat TCP-LISTEN:$(TUNNEL_PORT),fork,reuseaddr EXEC:"docker run --rm -i --network kind alpine/socat - TCP\:$(KIND_CLUSTER)-control-plane\:6443" & echo $$! > $(KIND_DIR)/tunnel.pid
	until kubectl get nodes >/dev/null 2>&1; do sleep 2; done  # API server/tunnel warm-up
	[ "$(GPU)" != 1 ] || { infra/gpu/setup-node.sh $(KIND_CLUSTER)-control-plane && kubectl apply -k infra/gpu; }
	kubectl apply -k infra/kustomize/base
	$(MAKE) cluster-secrets
	kubectl apply -k infra/argo
	kubectl -n jobfit wait --for=condition=Ready pod --all --timeout=180s
	kubectl -n jobfit port-forward svc/mlflow $(MLFLOW_PORT):5000 >/dev/null 2>&1 & echo $$! > $(KIND_DIR)/forward.pid
	kubectl -n jobfit port-forward svc/argo-server $(ARGO_PORT):2746 >/dev/null 2>&1 & echo $$! >> $(KIND_DIR)/forward.pid
	@echo "MLflow UI: http://localhost:$(MLFLOW_PORT)  Argo UI: https://localhost:$(ARGO_PORT) (self-signed cert)"

# Stage pods read HF_TOKEN from the hf-token Secret (pipeline/workflow.yaml); it comes from
# .env.cluster. Idempotent: rerun after editing the file to rotate the token.
cluster-secrets:
	if [ -f .env.cluster ]; then \
		kubectl -n jobfit create secret generic hf-token --from-env-file=.env.cluster --dry-run=client -o yaml | kubectl apply -f -; \
	else echo "cluster-secrets: no .env.cluster, skipping (gated models need HF_TOKEN)"; fi

cluster-down:
	[ ! -f $(KIND_DIR)/forward.pid ] || kill $$(cat $(KIND_DIR)/forward.pid) 2>/dev/null || true
	[ ! -f $(KIND_DIR)/tunnel.pid ] || kill $$(cat $(KIND_DIR)/tunnel.pid) 2>/dev/null || true
	rm -f $(KIND_DIR)/forward.pid $(KIND_DIR)/tunnel.pid $(KIND_DIR)/kubeconfig.yaml
	kind delete cluster --name $(KIND_CLUSTER)

# In-cluster curl pods, so the check uses the same DNS names as the pipeline pods.
cluster-check:
	kubectl -n jobfit get pods
	kubectl -n jobfit get workflows
	kubectl -n jobfit run mlflow-check --rm -i --restart=Never --image=curlimages/curl -- \
		curl -sf http://mlflow.jobfit.svc.cluster.local:5000/health
	kubectl -n jobfit run otel-check --rm -i --restart=Never --image=curlimages/curl -- \
		curl -sf http://otel-collector.jobfit.svc.cluster.local:8889/metrics
	$(MAKE) cluster-mlflow

# Pipeline pods are the ones carrying the workflow label; the stage name is an annotation.
CLUSTER_LOG_LINES ?= 30
cluster-logs:
	for pod in $$(kubectl -n jobfit get pods -l workflows.argoproj.io/workflow -o name); do \
		stage=$$(kubectl -n jobfit get $$pod -o jsonpath='{.metadata.annotations.workflows\.argoproj\.io/node-name}'); \
		echo "=== $$stage ($$pod) ==="; \
		kubectl -n jobfit logs $$pod -c main --tail=$(CLUSTER_LOG_LINES) 2>&1 || true; \
		echo; \
	done

# Latest runs with status and summary metrics, via the standing port-forward from cluster-up.
MLFLOW_RUNS ?= 10
cluster-mlflow:
	MLFLOW_TRACKING_URI=http://localhost:$(MLFLOW_PORT) $(PY) -c "import mlflow; df = mlflow.search_runs(experiment_names=['jobfit-pipeline'], order_by=['start_time DESC'], max_results=$(MLFLOW_RUNS)); cols=['tags.mlflow.runName','status','start_time']+[c for c in df.columns if c.startswith('metrics.')]; print(df[cols].rename(columns=lambda c: c.split('.')[-1]).to_string(index=False)) if not df.empty else print('no runs yet')"

# Submits one workflow for $(CANDIDATE) and waits for it; expects `make cluster-up` done
# at this same SCALE (the cluster mounted $(DATASET) and $(RUNS) then).
# Hyperparameters beyond the defaults go in as workflow
# parameters, e.g. ARGO_PARAMS="-p lora-rank=16" CANDIDATE=smollm2-135m-instruct-r16.
# Polls the workflow phase rather than `argo submit --watch`: one long-lived stream over
# the socat tunnel is fragile across a run of hours.
ARGO_PARAMS ?=
RUN_TIMEOUT_MIN ?= 1440
cluster-run: build
	kubectl get nodes >/dev/null 2>&1 || { echo "cluster-run: no cluster up -- run 'make cluster-up' first" >&2; exit 1; }
	kind load docker-image jobfit-pipeline:latest --name $(KIND_CLUSTER)
	WF=$$(argo submit -n jobfit pipeline/workflow.yaml --generate-name jobfit-$(subst .,-,$(CANDIDATE))- \
		-p model=$(MODEL) -p candidate=$(CANDIDATE) -p gpus=$(GPU) $(ARGO_PARAMS) -o json \
		| $(PY) -c "import json,sys; print(json.load(sys.stdin)['metadata']['name'])") && \
		echo "workflow: $$WF" && \
		i=0; \
		while [ $$i -lt $$(( $(RUN_TIMEOUT_MIN) * 4 )) ]; do \
			PHASE=$$(kubectl -n jobfit get workflows $$WF -o jsonpath='{.status.phase}' 2>/dev/null); \
			echo "[$$i] phase=$${PHASE:-Pending}"; \
			case "$$PHASE" in \
				Succeeded) break ;; \
				Failed|Error) echo "workflow $$WF $$PHASE" >&2; exit 1 ;; \
			esac; \
			i=$$((i + 1)); \
			sleep 15; \
		done; \
		[ "$$PHASE" = Succeeded ] || { echo "workflow $$WF did not succeed within timeout (phase=$$PHASE)" >&2; exit 1; }
	$(MAKE) cluster-check
	# Spans show up in the collector's own counters (:8888); the GPU gauge in the
	# exported pipeline metrics (:8889).
	kubectl -n jobfit run otel-data-check --rm -i --restart=Never --image=curlimages/curl -- sh -c \
		'curl -sf http://otel-collector.jobfit.svc.cluster.local:8888/metrics | grep -q otelcol_receiver_accepted_spans && \
		 curl -sf http://otel-collector.jobfit.svc.cluster.local:8889/metrics | grep -q gpu_utilization_percent'
	echo "report: $(RUNS)/$(CANDIDATE)/reports/report.md"

# --- publishing: .env.publish holds HF_TOKEN (write), HF_SPACE_REPO, optionally HF_MODEL_REPO
# (default gw0/jobfit-model) and, for the W&B report, WANDB_API_KEY ----------------------

PUBLISH := set -a; . ./.env.publish; set +a;

push-hf-model:
	$(PUBLISH) ./pipeline/publish.py --runs-dir $(RUNS) --model $(MODEL) --candidate $(CANDIDATE) --push-hf

push-hf-frontend:
	$(PUBLISH) frontend/scripts/push-hf-frontend.sh

push-wandb-report:
	$(PUBLISH) ./pipeline/publish.py --runs-dir $(RUNS) --model $(MODEL) --candidate $(CANDIDATE) --push-wandb
