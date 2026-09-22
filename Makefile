# make test    hermetic: every pytest suite. No network beyond package installs,
#              no credentials -- this is what CI runs.
# smoke-*      the live chain against datasets_smoke/: job fetching, `claude -p` CV
#              generation (needs an authenticated `claude` CLI) and the dataset build.

.PHONY: venv test smoke-fetch-jobs smoke-cvs smoke-dataset

VENV ?= .venv/bin
PY ?= $(VENV)/python

# Entry-point scripts use `#!/usr/bin/env python3`; resolve that to the venv.
export PATH := $(CURDIR)/.venv/bin:$(PATH)

venv:
	python3 -m venv .venv
	$(PY) -m pip install -q pytest -r data/requirements.txt
	$(PY) -m spacy download en_core_web_sm

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
