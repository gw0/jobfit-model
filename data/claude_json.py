"""`claude -p` structured-output plumbing for the dataset toolset (specs §11: the
Claude Code CLI stands in for a separately billed judge API).

Runs through `bash -i -c` so a user's `claude` shell alias is honoured. Needs an
authenticated `claude` in the calling shell; a nested call from inside another Claude
Code session does not inherit that session's auth.
"""

import json
import re
import shlex
import subprocess

DEFAULT_TIMEOUT_S = 180


class UsageLimitError(Exception):
    """The account's usage limit is hit (HTTP 429): every further call fails the same
    way until it resets, so batch callers stop instead of skipping item after item.
    Deliberately not a RuntimeError, which callers treat as a per-item failure."""


def _debug(stdout, stderr, n=1200):
    """Tail (not head) of both streams: the useful diagnostic is near the end
    of stdout's JSON (the terminal event) or in stderr, and a head slice tends
    to just show the init event and cut off before reaching either."""
    return f"stdout: {stdout[-n:]!r}\nstderr: {stderr[-n:]!r}"


def run_claude(prompt, model=None, timeout=DEFAULT_TIMEOUT_S):
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
        capture_output=True, text=True, timeout=timeout,
    )
    result = _result_event(proc)
    if result.get("api_error_status") == 429:
        raise UsageLimitError(f"claude usage limit hit: {result.get('result')}")
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}\n{_debug(proc.stdout, proc.stderr)}")
    if result.get("is_error"):
        raise RuntimeError(f"claude reported an error: {result.get('result')}")
    return result["result"]


def _result_event(proc):
    """The terminal "result" event claude prints, even when it exits non-zero."""
    # `bash -i` sources ~/.bashrc, which can print banner/prompt text and ANSI
    # control sequences to stdout ahead of claude's own output -- anchor on
    # the JSON object's own "{"..."}" markers rather than assuming stdout is
    # pure JSON (an ANSI CSI sequence starts "\x1b[" but never "{", so this
    # needs no separate ANSI-stripping step).
    stdout, failed = proc.stdout, f"claude exited {proc.returncode}, " if proc.returncode else ""
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"{failed}no JSON object found in claude output\n{_debug(stdout, proc.stderr)}")
    try:
        result = json.loads(stdout[start:end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{failed}could not parse claude output as JSON: {exc}\n"
                           f"{_debug(stdout, proc.stderr)}") from exc
    if result.get("type") != "result":
        raise RuntimeError(f"{failed}no result event in claude output\n{_debug(stdout, proc.stderr)}")
    return result


def extract_json_object(text):
    """Parse `text` as JSON; fall back to pulling the first balanced {...} block
    out of surrounding prose/code fences if the model didn't reply with pure JSON.

    strict=False: model responses routinely contain literal (unescaped) newlines
    inside string fields rather than "\\n" -- standard real-world LLM JSON output,
    not malformed input worth rejecting.
    """
    text = text.strip()
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1), strict=False)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1], strict=False)

    raise ValueError(f"could not find a JSON object in claude's response: {text[:200]!r}")


def ask_json(prompt, model=None, timeout=DEFAULT_TIMEOUT_S, max_attempts=2):
    """Run `prompt` through claude and parse its reply as a JSON object, retrying
    transient failures (a bad call or malformed reply) up to `max_attempts` times. A
    UsageLimitError is not transient and propagates at once."""
    last_exc = None
    for _ in range(max_attempts):
        try:
            raw = run_claude(prompt, model=model, timeout=timeout)
            return extract_json_object(raw)
        except UsageLimitError:
            raise
        except Exception as exc:  # noqa: BLE001 -- retry loop, re-raised below
            last_exc = exc
    raise RuntimeError(f"claude call failed after {max_attempts} attempt(s): {last_exc}")
