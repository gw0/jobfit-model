# make test    hermetic: every pytest suite. No network beyond package installs,
#              no credentials -- this is what CI runs.
# smoke-*      live steps against datasets_smoke/.

.PHONY: venv test smoke-fetch-jobs

VENV ?= .venv/bin
PY ?= $(VENV)/python

# Entry-point scripts use `#!/usr/bin/env python3`; resolve that to the venv.
export PATH := $(CURDIR)/.venv/bin:$(PATH)

venv:
	python3 -m venv .venv
	$(PY) -m pip install -q pytest -r data/requirements-fetch.txt

test:
	$(PY) -m pytest -q
	@! grep -n "Mozilla" data/fetch_jobs/*.py || { echo "spoofed browser User-Agent in data/fetch_jobs" >&2; exit 1; }

# --- dataset toolset -----------------------------------------------------------------

smoke-fetch-jobs:
	./data/fetch_jobs/fetch_greenhouse.py grafanalabs --out-dir $$(mktemp -d) --include engineer
