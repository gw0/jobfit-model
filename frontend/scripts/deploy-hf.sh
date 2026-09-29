#!/usr/bin/env bash
# Builds the SPA and deploys it to an HF Space (Static SDK), the primary deployment
# target (specs §7). VITE_HF_MODEL_REPO must name the model repo pushed by
# `pipeline/publish.py --push-hf`: the repo ships no weights, so without it the
# deployed app has nothing to load.
#
# Usage:
#   HF_TOKEN=hf_... HF_SPACE_REPO=<user-or-org>/jobfit \
#   VITE_HF_MODEL_REPO=<user-or-org>/jobfit-smollm2-135m-instruct frontend/scripts/deploy-hf.sh
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN to a real HF Hub API token with write access}"
: "${HF_SPACE_REPO:?set HF_SPACE_REPO to <user-or-org>/<space-name>}"
if [ -z "${VITE_HF_MODEL_REPO:-}" ]; then
  echo "WARNING: VITE_HF_MODEL_REPO is unset -- the deployed Space will try to load" >&2
  echo "model weights from a local /models/ path that does not exist there." >&2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend"

echo "building frontend (VITE_HF_MODEL_REPO=${VITE_HF_MODEL_REPO:-<unset>})..."
export VITE_HF_MODEL_REPO
npm --prefix "$FRONTEND_DIR" install
npm --prefix "$FRONTEND_DIR" run build

cp "$FRONTEND_DIR/space-readme.md" "$FRONTEND_DIR/dist/README.md"

python3 -m pip show huggingface_hub >/dev/null 2>&1 || python3 -m pip install -q huggingface_hub

HF_TOKEN="$HF_TOKEN" python3 - "$FRONTEND_DIR/dist" "$HF_SPACE_REPO" <<'PYEOF'
import sys

from huggingface_hub import HfApi

dist_dir, repo_id = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=dist_dir)
print(f"deployed -> https://huggingface.co/spaces/{repo_id}")
PYEOF
