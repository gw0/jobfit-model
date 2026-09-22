#!/usr/bin/env python3
"""Fetches job posts for every company in COMPANIES.

Usage: ./fetch_all.py [--include ...] [--exclude ...] [--location remote-us|remote-eu] [--out-dir ...]
"""

import sys

import fetch_ashby
import fetch_greenhouse
import fetch_lever
import fetch_workday
import jobboard

FETCHERS = {
    "greenhouse": fetch_greenhouse.fetch,
    "lever": fetch_lever.fetch,
    "ashby": fetch_ashby.fetch,
    "workday": fetch_workday.fetch,
}

# (name, provider, target): target is the org/board slug, or for workday the full
# tenant.host.myworkdayjobs.com/site careers URL.
COMPANIES = [
    # --- Greenhouse ---
    ("LawZero", "greenhouse", "lawzero"),
    ("UK AI Security Institute (AISI)", "greenhouse", "aisi"),
    ("Halcyon", "greenhouse", "halcyon"),
    ("Opaque Systems", "greenhouse", "opaquesystems"),
    ("Torq", "greenhouse", "torq"),
    ("Figure AI", "greenhouse", "figureai"),
    ("Xaira Therapeutics", "greenhouse", "xairatherapeutics"),
    ("Anthropic", "greenhouse", "anthropic"),
    ("Grafana Labs", "greenhouse", "grafanalabs"),
    ("Patronus AI", "greenhouse", "patronusaiinc"),
    ("AssemblyAI", "greenhouse", "assemblyai"),
    ("Endor Labs", "greenhouse", "endorlabs"),
    ("Dragos", "greenhouse", "dragos"),
    ("DataGrail", "greenhouse", "datagrail"),
    ("Osano", "greenhouse", "osano"),
    ("Transcend", "greenhouse", "transcendinc"),
    ("Huntress", "greenhouse", "huntress"),
    ("BitGo", "greenhouse", "bitgo"),
    ("Gensyn", "greenhouse", "gensyn"),
    ("Digital Asset (Canton Network)", "greenhouse", "digitalassetcorp"),
    ("Aptos", "greenhouse", "aptoslabs"),
    ("Aztec", "greenhouse", "aztec"),
    ("LayerZero", "greenhouse", "layerzerolabs"),
    ("Wormhole", "greenhouse", "wormholefoundation"),

    # --- Lever ---
    ("SuperAnnotate", "lever", "superannotate"),
    ("METR", "lever", "metr"),
    ("BenchSci", "lever", "benchsci"),
    ("Ketch", "lever", "Ketch"),
    ("Logz.io", "lever", "logz"),
    ("Anchorage Digital", "lever", "anchorage"),
    ("Celestia", "lever", "celestia"),
    ("Avive Solutions Inc", "lever", "AviveSolutions"),

    # --- Ashby ---
    ("Composio", "ashby", "composio"),
    ("E2B", "ashby", "e2b"),
    ("Letta", "ashby", "letta"),
    ("Mastra", "ashby", "Mastra"),
    ("Trigger.dev", "ashby", "triggerdev"),
    ("Vapi", "ashby", "vapi"),
    ("Cognition (Devin)", "ashby", "cognition"),
    ("Replit", "ashby", "replit"),
    ("Axelera AI", "ashby", "axelera"),
    ("Baseten", "ashby", "baseten"),
    ("Fireworks AI", "ashby", "fireworks"),
    ("Lambda", "ashby", "lambda"),
    ("Exa", "ashby", "exa"),
    ("Parallel Web Systems", "ashby", "parallel"),
    ("Pinecone", "ashby", "pinecone"),
    ("Abridge", "ashby", "Abridge"),
    ("Applied Intuition", "ashby", "applied"),
    ("Cohere", "ashby", "cohere"),
    ("insitro", "ashby", "insitro"),
    ("Harmonic", "ashby", "harmonic-ai"),
    ("Mistral AI", "ashby", "mistral.ai"),
    ("Runway", "ashby", "runway-ml"),
    ("Suno", "ashby", "suno"),
    ("Arthur AI", "ashby", "arthur-ai"),
    ("Traversal", "ashby", "traversal"),
    ("Deepgram", "ashby", "Deepgram"),
    ("Hume AI", "ashby", "hume-ai"),
    ("Contrast Security", "ashby", "contrast-security"),
    ("Ent", "ashby", "ent-security"),
    ("Socure", "ashby", "socure"),
    ("Optro (formerly AuditBoard)", "ashby", "optro"),
    ("Hyperliquid", "ashby", "Hyperliquid Labs"),
    ("Alchemy", "ashby", "alchemy"),
    ("Monad", "ashby", "monad.foundation"),
    ("Sui", "ashby", "Sui Foundation"),

    # --- Workday ---
    ("Boston Dynamics", "workday", "https://bostondynamics.wd1.myworkdayjobs.com/Boston_Dynamics"),
    ("CrowdStrike", "workday", "https://crowdstrike.wd5.myworkdayjobs.com/crowdstrikecareers"),
    ("SailPoint", "workday", "https://sailpoint.wd1.myworkdayjobs.com/SailPoint"),
    ("Darktrace", "workday", "https://darktrace.wd3.myworkdayjobs.com/DarktaceExternal"),
]


if __name__ == "__main__":
    args = jobboard.parse_cli()
    for name, provider, target in COMPANIES:
        print(f"--- {name} ({provider}) ---")
        try:
            posts = FETCHERS[provider](target, include=args.include, exclude=args.exclude)
        except Exception as exc:  # noqa: BLE001 -- one broken board shouldn't stop the rest
            print(f"error: {exc}", file=sys.stderr)
            continue
        for post in posts:
            post["company"] = name
        jobboard.save_posts(posts, location=args.location, out_dir=args.out_dir)
