#!/usr/bin/env bash
# Deploys the built SPA (frontend/dist) to an HF Space (Static SDK), the primary
# deployment target (specs §7). The repo ships no weights, so the deployed app loads them
# from the model repo pushed by `pipeline/publish.py --push-hf`: JOBFIT_MODEL_URL, by
# default main of HF_MODEL_REPO (default gw0/jobfit-model).
#
# Usage (`make push-hf-frontend` builds first and loads HF_TOKEN from .env.publish):
#   HF_TOKEN=hf_... [HF_SPACE_REPO=<org>/<space>] [HF_MODEL_REPO=<org>/<model>] \
#   frontend/scripts/push-hf-frontend.sh    (needs huggingface_hub: `make venv`)
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN to a real HF Hub API token with write access}"
HF_SPACE_REPO="${HF_SPACE_REPO:-gw0/jobfit-app}"
JOBFIT_MODEL_URL="${JOBFIT_MODEL_URL:-https://huggingface.co/${HF_MODEL_REPO:-gw0/jobfit-model}/resolve/main}"

FRONTEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "$FRONTEND_DIR/dist" ] || { echo "no frontend/dist: run \`make frontend-build\` first" >&2; exit 1; }

printf '{"modelUrl": "%s"}\n' "$JOBFIT_MODEL_URL" > "$FRONTEND_DIR/dist/config.json"

cp "$FRONTEND_DIR/hf-space-readme.md" "$FRONTEND_DIR/dist/README.md"

python3 - "$FRONTEND_DIR/dist" "$HF_SPACE_REPO" <<'PYEOF'
import sys

from huggingface_hub import HfApi

dist_dir, repo_id = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=dist_dir, ignore_patterns=["models/**"])
print(f"deployed -> https://huggingface.co/spaces/{repo_id}")
PYEOF
