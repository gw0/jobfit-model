# AGENTS.md

## Guidelines

- Keep results simple, lean, and consistent — as if written from scratch.

## Commands

Commands (see README.md for details):
- `make venv && npm --prefix frontend install` — install deps
- `make test` — hermetic suite (pytest + frontend typecheck/build/vitest); this is what CI runs
- `make smoke` — live: network, Docker, and an authenticated `claude` CLI; don't run without asking first

## Hard constraints

- `make cluster-up`/`make cluster-down` must be run manually by the user, prefixed
  with `! DOCKER_HOST=` (e.g. `! DOCKER_HOST= make cluster-up`) — not run directly by
  the agent.
- `.env.local`, `.env.cluster` and `.env.publish` hold secrets: never read, print or
  commit them, and never put a token on a command line.
- `pipeline/jev.py` / `pipeline/jev_model.py` and `frontend/src/jev.mjs` must stay in
  exact parity. Mirror any change to one in the other, and check with
  `node frontend/scripts/verify-parity.mjs <model-export-dir>`.
- `data/generate_cvs.py` and `data/label_dataset.py` shell out to the `claude` CLI. A
  nested call from inside a Claude Code session does not inherit this session's auth,
  so running them from here will fail or hang.
- `datasets/` and `runs/` are symlinks to one committed scale (`datasets_smoke/`/`runs_smoke/`
  or `datasets_full/`/`runs_full/`). Don't overwrite or relabel either corpus in place —
  relabeling re-spends real LLM-judge calls.
- `pipeline/requirements.txt` pins `onnxruntime` to match the version bundled by
  `@huggingface/transformers` in `frontend/`; don't bump one without the other.
- `data/fetch_jobs/*.py` must never spoof a browser User-Agent — `make test` greps for
  "Mozilla" and fails the build if found.
- `specs/*.md` are the normative spec, not just background — check them (especially
  `specs/20260923-jev-model.md` for the Jev model interface) before assuming intended
  behavior.
