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


def _debug(stdout, stderr, n=1200):
    """Tail (not head) of both streams: the useful diagnostic is near the end
    of stdout's JSON (the terminal event) or in stderr, and a head slice tends
    to just show the init event and cut off before reaching either."""
    return f"stdout: {stdout[-n:]!r}\nstderr: {stderr[-n:]!r}"


def run_claude(prompt, model=None, timeout=DEFAULT_TIMEOUT_S):
    """Shell out to `claude -p`, return the assistant's final text.

    `--output-format json` prints a JSON array of session events; the last
    event (type "result") holds the final text in its "result" field.
    """
    cmd = f"claude --output-format json -p {shlex.quote(prompt)}"
    if model:
        cmd += f" --model {shlex.quote(model)}"
    proc = subprocess.run(
        ["bash", "-i", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}\n{_debug(proc.stdout, proc.stderr)}")

    # `bash -i` sources ~/.bashrc, which can print banner/prompt text and ANSI
    # control sequences to stdout ahead of claude's own output -- anchor on
    # the JSON array's own "[{"..."}]" markers rather than assuming stdout is
    # pure JSON (an ANSI CSI sequence starts "\x1b[" but never "[{", so this
    # needs no separate ANSI-stripping step).
    stdout = proc.stdout
    start = stdout.find("[{")
    end_marker = stdout.rfind("}]")
    if start == -1 or end_marker == -1 or end_marker < start:
        raise RuntimeError(f"no JSON array found in claude output\n{_debug(stdout, proc.stderr)}")
    try:
        events = json.loads(stdout[start:end_marker + 2])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not parse claude output as JSON: {exc}\n{_debug(stdout, proc.stderr)}") from exc

    result_events = [e for e in events if e.get("type") == "result"]
    if not result_events:
        raise RuntimeError(f"no result event in claude output\n{_debug(stdout, proc.stderr)}")

    result = result_events[-1]
    if result.get("is_error"):
        raise RuntimeError(f"claude reported an error: {result.get('result')}")
    return result["result"]


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
    transient failures (a bad call or malformed reply) up to `max_attempts` times."""
    last_exc = None
    for _ in range(max_attempts):
        try:
            raw = run_claude(prompt, model=model, timeout=timeout)
            return extract_json_object(raw)
        except Exception as exc:  # noqa: BLE001 -- retry loop, re-raised below
            last_exc = exc
    raise RuntimeError(f"claude call failed after {max_attempts} attempt(s): {last_exc}")
