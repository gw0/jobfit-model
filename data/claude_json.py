"""`claude -p` structured-output plumbing for the dataset toolset (specs §11: the
Claude Code CLI stands in for a separately billed judge API).

Runs through `bash -i -c` so a user's `claude` shell alias is honoured. Needs an
authenticated `claude` in the calling shell; a nested call from inside another Claude
Code session does not inherit that session's auth.
"""

import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

TIMEOUT_S = 180
MAX_ATTEMPTS = 2


class UsageLimitError(Exception):
    """The account's usage limit is hit (HTTP 429): every further call fails the same
    way until it resets, so batch callers stop instead of skipping item after item.
    Deliberately not a RuntimeError, which callers treat as a per-item failure."""


def _debug(stdout, stderr, n=1200):
    """Tail (not head) of both streams: the useful diagnostic is near the end
    of stdout's JSON (the terminal event) or in stderr, and a head slice tends
    to just show the init event and cut off before reaching either."""
    return f"stdout: {stdout[-n:]!r}\nstderr: {stderr[-n:]!r}"


def run_claude(prompt, model=None):
    """Shell out to `claude -p`, return the assistant's final text.

    `--output-format json` prints a single JSON object -- the terminal
    "result" event -- whose "result" field holds the final text. `--system-prompt ''`
    is an explicit empty override: none of these calls are coding tasks, so Claude
    Code's default system prompt (environment info, tool descriptions, etc.) would
    only add input tokens; each prompt below already states its own output-format
    rules.
    """
    cmd = f"claude --output-format json -p {shlex.quote(prompt)} --system-prompt ''"
    if model:
        cmd += f" --model {shlex.quote(model)}"
    proc = subprocess.run(
        ["bash", "-i", "-c", cmd],
        capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    result = _result_event(proc)
    if result.get("api_error_status") == 429:
        raise UsageLimitError(f"claude usage limit hit: {result.get('result')}")
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}\n{_debug(proc.stdout, proc.stderr)}")
    if result.get("is_error"):
        raise RuntimeError(f"claude reported an error: {result.get('result')}")
    return result["result"]


def _json_slice(text):
    """The text from its first "{" to its last "}", or None if it has no such span."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else None


def _result_event(proc):
    """The terminal "result" event claude prints, even when it exits non-zero."""
    # `bash -i` sources ~/.bashrc, which can print banner/prompt text and ANSI
    # control sequences to stdout ahead of claude's own output -- anchor on
    # the JSON object's own "{"..."}" markers rather than assuming stdout is
    # pure JSON (an ANSI CSI sequence starts "\x1b[" but never "{", so this
    # needs no separate ANSI-stripping step).
    stdout, failed = proc.stdout, f"claude exited {proc.returncode}, " if proc.returncode else ""
    payload = _json_slice(stdout)
    if payload is None:
        raise RuntimeError(f"{failed}no JSON object found in claude output\n{_debug(stdout, proc.stderr)}")
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{failed}could not parse claude output as JSON: {exc}\n"
                           f"{_debug(stdout, proc.stderr)}") from exc
    if result.get("type") != "result":
        raise RuntimeError(f"{failed}no result event in claude output\n{_debug(stdout, proc.stderr)}")
    return result


def extract_json_object(text):
    """Parse `text` as JSON; fall back to the span from its first "{" to its last "}"
    if the model wrapped the object in prose or code fences.

    strict=False: model responses routinely contain literal (unescaped) newlines
    inside string fields rather than "\\n" -- standard real-world LLM JSON output,
    not malformed input worth rejecting.
    """
    text = text.strip()
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        payload = _json_slice(text)
        if payload is None:
            raise ValueError(f"could not find a JSON object in claude's response: {text[:200]!r}") from None
        return json.loads(payload, strict=False)


def ask_json(prompt, model=None):
    """Run `prompt` through claude and parse its reply as a JSON object, retrying
    transient failures (a bad call or malformed reply) up to MAX_ATTEMPTS times. A
    UsageLimitError is not transient and propagates at once."""
    last_exc = None
    for _ in range(MAX_ATTEMPTS):
        try:
            return extract_json_object(run_claude(prompt, model=model))
        except UsageLimitError:
            raise
        except Exception as exc:  # noqa: BLE001 -- retry loop, re-raised below
            last_exc = exc
    raise RuntimeError(f"claude call failed after {MAX_ATTEMPTS} attempt(s): {last_exc}") from last_exc


def run_parallel(what, items, fn, workers, on_result):
    """Calls `fn(item)` for each item on `workers` threads and `on_result(item, result)`
    on the calling thread as each finishes. A failed item is skipped (one bad reply
    shouldn't lose the batch); the usage limit stops at once, keeping what was already
    passed on. Returns how many items succeeded."""
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for n, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                result = future.result()
            except UsageLimitError as exc:
                pool.shutdown(cancel_futures=True)
                raise SystemExit(f"{exc} -- stopped after {done} {what}(s); re-run once it resets to resume")
            except Exception as exc:  # noqa: BLE001
                print(f"[{n}/{len(items)}] {what} {item} SKIPPED ({exc})")
                continue
            done += 1
            print(f"[{n}/{len(items)}] {what} {item} done")
            on_result(item, result)
    return done
