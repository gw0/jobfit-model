#!/usr/bin/env bash
# Builds the SPA and deploys it to an HF Space (Static SDK), the primary deployment
# target (specs §7). The repo ships no weights, so the deployed app loads them from the
# model repo pushed by `pipeline/publish.py --push-hf`: JOBFIT_MODEL_URL, by default
# main of HF_MODEL_REPO (default gw0/jobfit-model).
#
# Usage:
#   HF_TOKEN=hf_... [HF_SPACE_REPO=<org>/<space>] [HF_MODEL_REPO=<org>/<model>] \
#   frontend/scripts/push-hf-frontend.sh    (needs huggingface_hub: `make venv`)
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN to a real HF Hub API token with write access}"
HF_SPACE_REPO="${HF_SPACE_REPO:-gw0/jobfit-app}"
JOBFIT_MODEL_URL="${JOBFIT_MODEL_URL:-https://huggingface.co/${HF_MODEL_REPO:-gw0/jobfit-model}/resolve/main}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend"

echo "building frontend (JOBFIT_MODEL_URL=$JOBFIT_MODEL_URL)..."
npm --prefix "$FRONTEND_DIR" install
npm --prefix "$FRONTEND_DIR" run build

printf '{"modelUrl": "%s"}\n' "$JOBFIT_MODEL_URL" > "$FRONTEND_DIR/dist/config.json"

cp "$FRONTEND_DIR/space-readme.md" "$FRONTEND_DIR/dist/README.md"

python3 - "$FRONTEND_DIR/dist" "$HF_SPACE_REPO" <<'PYEOF'
import sys

from huggingface_hub import HfApi

dist_dir, repo_id = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=dist_dir)
print(f"deployed -> https://huggingface.co/spaces/{repo_id}")
PYEOF
