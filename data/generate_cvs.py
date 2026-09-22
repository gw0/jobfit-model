#!/usr/bin/env python3
"""Generates synthetic technical/SWE CVs via `claude -p` (specs §5).

No real CVs are ever collected: each CV is LLM-generated directly as Markdown, then
scanned with Presidio (pii_scrub.py) since generators do emit real-looking names.

Usage:
    ./data/generate_cvs.py --count 10 --out-dir datasets_smoke

Writes <out-dir>/cvs/<slug>.md. Needs an authenticated `claude` CLI in the calling shell.
"""

import argparse
from pathlib import Path

import corpus
import pii_scrub
from claude_json import ask_json

# Cycled through by index to keep a batch varied without an extra LLM call.
PROFILES = [
    ("Backend Engineer", "junior", 1, "Python/Django and PostgreSQL"),
    ("Frontend Engineer", "mid-level", 4, "React/TypeScript"),
    ("Site Reliability Engineer", "senior", 8, "Kubernetes, Terraform, and Go"),
    ("Data Engineer", "mid-level", 5, "Spark, Airflow, and dbt"),
    ("Mobile Engineer (iOS)", "senior", 7, "Swift and SwiftUI"),
    ("Machine Learning Engineer", "mid-level", 4, "PyTorch and MLOps tooling"),
    ("Security Engineer", "senior", 9, "appsec, threat modeling, and Go"),
    ("Full-Stack Engineer", "junior", 2, "Node.js and React"),
    ("Platform/DevOps Engineer", "senior", 10, "AWS, Terraform, and CI/CD"),
    ("QA/SDET", "mid-level", 5, "Python test automation and Selenium"),
    ("Embedded Systems Engineer", "senior", 8, "C/C++ and RTOS"),
    ("Staff Engineer", "staff", 12, "distributed systems in Go and Java"),
]

PROMPT_TEMPLATE = """\
Generate ONE synthetic, entirely fictional technical/software-engineering CV.

Profile to write for:
- Role: {seniority} {role}
- Years of experience: ~{years}
- Primary stack: {stack}

Requirements:
- Invent a plausible but clearly fictional full name and fictional past employers \
(never a real company or real person).
- Use a fake contact email of the form firstname.lastname@example.com.
- Write the CV body as clean Markdown (headings + bullet lists), already in \
final parsed form -- no PDF/HTML artifacts. Roughly 300-600 words.
- Include: name/contact line, a short summary, a skills section, 2-4 work \
experience entries with bullet achievements, and education.

Respond with ONLY a single raw JSON object, no markdown code fences, no \
commentary before or after it, with exactly these two keys:
{{"name": "<kebab-case slug of the fictional full name, e.g. jane-doe>", \
"markdown": "<the full CV as a markdown string>"}}
"""


def generate_one(profile):
    role, seniority, years, stack = profile
    prompt = PROMPT_TEMPLATE.format(role=role, seniority=seniority, years=years, stack=stack)
    try:
        data = ask_json(prompt)
        markdown = data["markdown"]
        if not markdown.strip():
            raise ValueError("empty markdown in response")
        return corpus.slugify(data["name"]) or "cv", markdown
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to generate CV for profile {profile!r}: {exc}") from exc


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
    parser.add_argument("--skip-scrub", action="store_true", help="skip the Presidio PII scan")
    args = parser.parse_args()

    for i in range(args.count):
        profile = PROFILES[i % len(PROFILES)]
        print(f"[{i + 1}/{args.count}] generating {profile[1]} {profile[0]}...")
        path = write_cv(args.out_dir, *generate_one(profile))
        print(f"  wrote {path}")
        if not args.skip_scrub and not pii_scrub.scan_paths([path]):
            print(f"  WARNING: PII findings in {path.name} (see above)")


if __name__ == "__main__":
    main()
