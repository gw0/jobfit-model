#!/usr/bin/env python3
"""Presidio PII scan over generated text (specs §5).

The corpus holds no real CVs, but generators occasionally emit real-looking contact
details. `pipeline/prepare.py` re-runs this scan read-only and fails on a hit.

Usage:
    ./data/pii_scrub.py <file-or-dir> [--threshold 0.5] [--spacy-model en_core_web_sm]

Exits 1 if any finding at/above --threshold is present in any scanned file.
"""

import argparse
import functools
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 0.5
DEFAULT_SPACY_MODEL = "en_core_web_sm"

# Every synthetic CV necessarily contains a name/employer/date; the generic
# PERSON/ORGANIZATION/LOCATION/DATE_TIME recognizers are too noisy to mean anything
# here, so the scan is scoped to types that are unambiguous PII if present at all.
SCAN_ENTITIES = [
    "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "IBAN_CODE",
    "IP_ADDRESS", "CRYPTO", "US_PASSPORT", "MEDICAL_LICENSE",
]

# generate_cvs.py asks for firstname.lastname@example.com contact emails.
FAKE_EMAIL_DOMAIN = "@example.com"


@functools.lru_cache(maxsize=None)
def _get_analyzer(spacy_model=DEFAULT_SPACY_MODEL):
    """Pinned to an explicit spaCy model: Presidio's default loads en_core_web_lg
    (~500MB), overkill for this safety net."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
    })
    return AnalyzerEngine(nlp_engine=provider.create_engine())


def scan_paths(paths, threshold=DEFAULT_THRESHOLD, spacy_model=DEFAULT_SPACY_MODEL):
    """Scans each file, or each .md under each directory, printing any findings at/above
    `threshold` (known-fake emails excepted). Returns True if everything is clean."""
    clean = True
    for path in paths:
        if Path(path).is_dir():
            clean &= scan_paths(sorted(Path(path).rglob("*.md")), threshold, spacy_model)
            continue
        text = Path(path).read_text(encoding="utf-8")
        findings = [
            r for r in _get_analyzer(spacy_model).analyze(text=text, language="en", entities=SCAN_ENTITIES)
            if r.score >= threshold
            and not (r.entity_type == "EMAIL_ADDRESS" and text[r.start:r.end].lower().endswith(FAKE_EMAIL_DOMAIN))
        ]
        if findings:
            clean = False
            print(f"{path}: {len(findings)} finding(s)")
            for f in findings:
                print(f"  {f.entity_type} score={f.score:.2f} text={text[f.start:f.end]!r}")
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="a file or a directory of .md files to scan")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    args = parser.parse_args()

    clean = scan_paths([args.path], threshold=args.threshold, spacy_model=args.spacy_model)
    print("clean" if clean else "PII found")
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
