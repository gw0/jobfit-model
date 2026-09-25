#!/usr/bin/env python3
"""Generates synthetic technical/SWE CVs via `claude -p` (specs §5).

No real CVs are ever collected: each CV is LLM-generated directly as Markdown, then
scanned with Presidio (pii_scrub.py) since generators do emit real-looking names.

Usage:
    ./data/generate_cvs.py --count 10 --out-dir datasets_smoke
    ./data/generate_cvs.py --count 300 --out-dir datasets_full --workers 8

Writes <out-dir>/cvs/<slug>.md. Needs an authenticated `claude` CLI in the calling shell.
"""

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import corpus
import pii_scrub
from claude_json import UsageLimitError, ask_json

# The CV at index i gets profile(i): a seeded draw over these axes, so a batch is
# varied (and a resumed batch continues it) without an extra LLM call. The role
# families follow the fetched job corpus; the last two are off-domain on purpose, so
# the pairs include clear mismatches.
ROLE_FAMILIES = [
    ("Machine Learning Engineer", "PyTorch, model training and MLOps tooling"),
    ("AI Research Engineer", "LLM evaluation, Python and large-scale experiments"),
    ("Security Engineer", "appsec, threat modeling and Go"),
    ("Site Reliability Engineer", "Kubernetes, Terraform and Go"),
    ("Data Engineer", "Spark, Airflow and dbt"),
    ("Backend Engineer", "Python/Django and PostgreSQL"),
    ("Frontend Engineer", "React and TypeScript"),
    ("Full-Stack Engineer", "Node.js, React and PostgreSQL"),
    ("Mobile Engineer", "Swift/SwiftUI and Kotlin"),
    ("Embedded/Robotics Engineer", "C/C++, ROS and RTOS"),
    ("Blockchain Engineer", "Rust, Solidity and distributed consensus"),
    ("QA/SDET", "Python test automation and Playwright"),
    ("Engineering Manager", "team leadership, delivery and hiring"),
    ("Product Designer", "Figma, user research and design systems"),
    ("Account Executive", "B2B SaaS sales and CRM pipelines"),
]
SENIORITIES = [("junior", 1), ("mid-level", 4), ("senior", 8), ("staff", 12), ("principal", 16)]
LOCATIONS = [
    "San Francisco, USA (on-site)", "New York, USA (hybrid)", "Austin, USA (remote)",
    "Toronto, Canada (remote)", "London, UK (hybrid)", "Berlin, Germany (remote)",
    "Warsaw, Poland (remote)", "Lagos, Nigeria (remote)", "Bangalore, India (on-site)",
    "Sao Paulo, Brazil (remote)", "Tokyo, Japan (hybrid)", "Sydney, Australia (hybrid)",
]
# Seeded initials for the invented name: left alone, the generator keeps returning the
# same few names (8 of the 10 smoke CVs are a "Marcus"). Rare initials are left out.
NAME_INITIALS = "ABCDEFGHIJKLMNOPRSTVWY"
WRITING_STYLES = [
    "polished and well structured",
    "terse, with sparse one-line bullets",
    "dense and somewhat disorganized, with inconsistent formatting",
]
DEFAULT_SEED = 42

PROMPT_TEMPLATE = """\
Generate ONE synthetic, entirely fictional CV.

Profile to write for:
- Role: {seniority} {role}
- Years of experience: ~{years}
- Primary stack/skills: {stack}
- Based in: {location}
- Writing style: {style}

Requirements:
- Invent a plausible but clearly fictional full name fitting the location, with a \
first name starting with "{first_initial}" and a surname starting with "{last_initial}", and \
fictional past employers (never a real company or real person).
- Use a fake contact email of the form firstname.lastname@example.com.
- Write the CV body as Markdown (headings + bullet lists), already in final parsed \
form -- no PDF/HTML artifacts. Roughly 300-600 words.
- Include: name/contact line, a short summary, a skills section, 1-4 work \
experience entries with bullet achievements, and education.

Respond with ONLY a single raw JSON object, no markdown code fences, no \
commentary before or after it, no tool use, with exactly these two keys:
{{"name": "<kebab-case slug of the fictional full name, e.g. jane-doe>", \
"markdown": "<the full CV as a markdown string>"}}
"""


def profile(index, seed=DEFAULT_SEED):
    """The prompt fields for the CV at `index`; the same for a given (index, seed)."""
    rng = random.Random(f"{seed}-{index}")
    role, stack = rng.choice(ROLE_FAMILIES)
    seniority, years = rng.choice(SENIORITIES)
    return {"role": role, "stack": stack, "seniority": seniority, "years": years,
            "location": rng.choice(LOCATIONS), "style": rng.choice(WRITING_STYLES),
            "first_initial": rng.choice(NAME_INITIALS), "last_initial": rng.choice(NAME_INITIALS)}


def generate_one(fields):
    try:
        data = ask_json(PROMPT_TEMPLATE.format(**fields))
        markdown = data["markdown"]
        if not markdown.strip():
            raise ValueError("empty markdown in response")
        return corpus.slugify(data["name"]) or "cv", markdown
    except UsageLimitError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to generate CV for profile {fields!r}: {exc}") from exc


def write_cv(out_dir, slug, markdown):
    """Writes cvs/<slug>.md, or <slug>-2.md, <slug>-3.md, ... if the name is taken."""
    cvs_dir = Path(out_dir) / "cvs"
    cvs_dir.mkdir(parents=True, exist_ok=True)
    path, n = cvs_dir / f"{slug}.md", 1
    while path.exists():
        n += 1
        path = cvs_dir / f"{slug}-{n}.md"
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, required=True, help="number of CVs to generate")
    parser.add_argument("--out-dir", type=Path, default=corpus.DEFAULT_DATASET_DIR)
    parser.add_argument("--workers", type=int, default=1, help="parallel `claude` calls")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-scrub", action="store_true", help="skip the Presidio PII scan")
    args = parser.parse_args()

    existing = corpus.list_cvs(args.out_dir)
    if len(existing) >= args.count:
        print(f"already have {len(existing)} CVs (target {args.count}); nothing to do")
        if len(existing) > args.count:
            print(f"WARNING: {len(existing)} existing CVs exceed the requested --count {args.count}")
        return

    # Files are written here on the main thread, so write_cv's name dedup can't race.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate_one, profile(i, args.seed)): i for i in range(len(existing), args.count)}
        for future in as_completed(futures):
            try:
                path = write_cv(args.out_dir, *future.result())
            except UsageLimitError as exc:
                pool.shutdown(cancel_futures=True)
                raise SystemExit(f"{exc} -- stopped; re-run once it resets to generate the rest")
            except RuntimeError as exc:
                print(f"[{futures[future] + 1}/{args.count}] SKIPPED ({exc})")
                continue
            print(f"[{futures[future] + 1}/{args.count}] wrote {path}")
            if not args.skip_scrub and not pii_scrub.scan_paths([path]):
                print(f"  WARNING: PII findings in {path.name} (see above)")

    # Non-zero, so `make cvs dataset` doesn't go on to split an incomplete corpus.
    written = len(corpus.list_cvs(args.out_dir))
    if written < args.count:
        raise SystemExit(f"only {written}/{args.count} CVs after skipped failures -- re-run to generate the rest")


if __name__ == "__main__":
    main()
