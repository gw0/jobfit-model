# make test    hermetic: every pytest suite. No network beyond package installs,
#              no credentials -- this is what CI runs.
# make build   the pipeline Docker image.
# smoke-*      the live chain against datasets_smoke/: job fetching, `claude -p` CV
#              generation and labeling (needs an authenticated `claude` CLI), the
#              dataset build, the Dockerized pipeline and the KinD/Argo run.

.PHONY: venv build test smoke-fetch-jobs smoke-cvs smoke-dataset smoke-labels \
	smoke-prepare smoke-headtrain smoke-finetune smoke-calibrate smoke-export smoke-publish \
	cluster-up cluster-down cluster-check cluster-logs cluster-mlflow smoke-cluster \
	publish-hf

VENV ?= .venv/bin
PY ?= $(VENV)/python
MODEL ?= Qwen/Qwen3-0.6B

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

smoke-headtrain: smoke-prepare
	$(DOCKER_RUN) ./pipeline/train.py --stage headtrain $(STAGE_ARGS)
	$(DOCKER_RUN) ./pipeline/evaluate.py --stage headtrained $(STAGE_ARGS)

smoke-finetune: smoke-headtrain
	$(DOCKER_RUN) ./pipeline/train.py --stage finetune $(STAGE_ARGS)
	$(DOCKER_RUN) ./pipeline/evaluate.py --stage finetuned $(STAGE_ARGS)

smoke-calibrate: smoke-finetune
	$(DOCKER_RUN) ./pipeline/calibrate.py $(STAGE_ARGS)
	$(DOCKER_RUN) ./pipeline/evaluate.py --stage calibrated $(STAGE_ARGS)

smoke-export: smoke-calibrate
	$(DOCKER_RUN) ./pipeline/export.py $(STAGE_ARGS)
	$(DOCKER_RUN) ./pipeline/evaluate.py --stage quantized $(STAGE_ARGS)

smoke-publish: smoke-export
	$(DOCKER_RUN) ./pipeline/publish.py $(STAGE_ARGS)

# --- cluster (KinD + Argo + MLflow + OTel collector) ------------------------------------
# The same flow runs a real training job: with ./datasets and ./runs in place, submit
# one workflow per candidate (`argo submit -n jobfit pipeline/workflow.yaml
# --generate-name jobfit-run-<slug-with-dots-as-hyphens>- -p model=...`). The committed
# runs/<slug>/reports/ is the durable cross-host comparison; MLflow's hostPath store is not.

KIND_CLUSTER ?= jobfit
TUNNEL_PORT ?= 16443
KIND_DIR := .kind
export KUBECONFIG := $(CURDIR)/$(KIND_DIR)/kubeconfig.yaml

# kind's Docker daemon may not share our network namespace, so its published API port
# can be unreachable. Tunnel over the Docker socket instead: socat relays each
# connection through an alpine/socat container on the "kind" network. socat splits
# its address on every ':' inside EXEC:"...", hence the "\:" escapes.
cluster-up:
	mkdir -p $(KIND_DIR)/mlflow-data
	kind create cluster --name $(KIND_CLUSTER) --config infra/kind-config.yaml
	kind get kubeconfig --name $(KIND_CLUSTER) | sed 's#server: https://127.0.0.1:[0-9]*#server: https://127.0.0.1:$(TUNNEL_PORT)#' > $(KIND_DIR)/kubeconfig.yaml
	socat TCP-LISTEN:$(TUNNEL_PORT),fork,reuseaddr EXEC:"docker run --rm -i --network kind alpine/socat - TCP\:$(KIND_CLUSTER)-control-plane\:6443" & echo $$! > $(KIND_DIR)/tunnel.pid
	until kubectl get nodes >/dev/null 2>&1; do sleep 2; done  # API server/tunnel warm-up
	kubectl apply -k infra/kustomize/base
	kubectl apply -k infra/argo
	kubectl -n jobfit wait --for=condition=Ready pod --all --timeout=180s

cluster-down:
	[ ! -f $(KIND_DIR)/tunnel.pid ] || kill $$(cat $(KIND_DIR)/tunnel.pid) 2>/dev/null || true
	rm -f $(KIND_DIR)/tunnel.pid $(KIND_DIR)/kubeconfig.yaml
	kind delete cluster --name $(KIND_CLUSTER)

# In-cluster curl pods, so the check uses the same DNS names as the pipeline pods.
cluster-check:
	kubectl -n jobfit get pods
	kubectl -n jobfit get workflows
	kubectl -n jobfit run mlflow-check --rm -i --restart=Never --image=curlimages/curl -- \
		curl -sf http://mlflow.jobfit.svc.cluster.local:5000/health
	kubectl -n jobfit run otel-check --rm -i --restart=Never --image=curlimages/curl -- \
		curl -sf http://otel-collector.jobfit.svc.cluster.local:8889/metrics

# Pipeline pods are the ones carrying the workflow label; the stage name is an annotation.
CLUSTER_LOG_LINES ?= 30
cluster-logs:
	for pod in $$(kubectl -n jobfit get pods -l workflows.argoproj.io/workflow -o name); do \
		stage=$$(kubectl -n jobfit get $$pod -o jsonpath='{.metadata.annotations.workflows\.argoproj\.io/node-name}'); \
		echo "=== $$stage ($$pod) ==="; \
		kubectl -n jobfit logs $$pod -c main --tail=$(CLUSTER_LOG_LINES) 2>&1 || true; \
		echo; \
	done

cluster-mlflow:
	kubectl -n jobfit port-forward svc/mlflow 5000:5000 >/dev/null 2>&1 & \
		PF_PID=$$!; \
		trap "kill $$PF_PID 2>/dev/null" EXIT; \
		sleep 2; \
		MLFLOW_TRACKING_URI=http://localhost:5000 $(PY) -c "import mlflow; df = mlflow.search_runs(experiment_names=['jobfit-pipeline'], order_by=['start_time DESC']); cols=[c for c in ['tags.mlflow.runName','tags.mlflow.parentRunId','status','start_time'] if c in df.columns]; print(df[cols].to_string(index=False)) if not df.empty else print('no runs yet')"

# Expects a cluster already up via `make cluster-up` -- doesn't create or tear one down
# itself, so repeated runs during iteration skip kind's bootstrap cost each time.
# kind-config.yaml mounts ./datasets and ./runs when the cluster is created, so they must
# already point at the smoke tree then (`ln -sfn datasets_smoke datasets; ln -sfn
# runs_smoke runs` before cluster-up); the check refuses to train on a real corpus.
# Polls the workflow phase rather than `argo submit --watch`: one long-lived stream over
# the socat tunnel is fragile across a run of hours. A CPU run takes 2h+.
SMOKE_TIMEOUT_MIN ?= 360
smoke-cluster: build
	kubectl get nodes >/dev/null 2>&1 || { echo "smoke-cluster: no cluster up -- run 'make cluster-up' first" >&2; exit 1; }
	[ "$$(readlink datasets)" = datasets_smoke ] && [ "$$(readlink runs)" = runs_smoke ] || \
		{ echo "smoke-cluster: ./datasets and ./runs must link to datasets_smoke and runs_smoke before cluster-up" >&2; exit 1; }
	kind load docker-image jobfit-pipeline:latest --name $(KIND_CLUSTER)
	WF=$$(argo submit -n jobfit pipeline/workflow.yaml --generate-name jobfit-run-$(subst .,-,$(SLUG))- \
		-p model=$(MODEL) -o json | $(PY) -c "import json,sys; print(json.load(sys.stdin)['metadata']['name'])") && \
		echo "workflow: $$WF" && \
		i=0; \
		while [ $$i -lt $$(( $(SMOKE_TIMEOUT_MIN) * 4 )) ]; do \
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
	echo "report: runs_smoke/$(SLUG)/reports/report.md"

# --- publishing (needs HF_TOKEN) -------------------------------------------------------

publish-hf:
	./pipeline/publish.py --runs-dir runs_smoke --model $(MODEL) --push-hf
