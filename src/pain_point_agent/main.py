import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from pain_point_agent.client import make_client
from pain_point_agent.config import settings
from pain_point_agent.db import Database
from pain_point_agent.loop import run_consensus, run_research


def trace_path(niche: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", niche.lower()).strip("-")
    return Path("traces") / f"{slug}-{datetime.now():%Y%m%d-%H%M%S}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Find what people in a niche struggle with.")
    parser.add_argument("niche", help='e.g. "WhatsApp bots for small shops"')
    parser.add_argument("--trace", action="store_true", help="print every round and save the conversation to traces/")
    parser.add_argument("--runs", type=int, default=1, help="run several times and keep problems most runs agree on")
    parser.add_argument("--judge-votes", type=int, help="ask each judge this many times and take the majority")
    parser.add_argument("--max-age-days", type=int, help="reject evidence from posts older than this")
    parser.add_argument("--no-save", action="store_true", help="do not save the run to the SQLite database")
    args = parser.parse_args()
    if args.judge_votes:
        settings.judge_votes = args.judge_votes
    if args.max_age_days:
        settings.max_evidence_age_days = args.max_age_days

    client = make_client()
    db = None if args.no_save else Database(settings.db_path)
    try:
        if args.runs > 1:
            traces = [trace_path(f"{args.niche} run {number}") if args.trace else None for number in range(1, args.runs + 1)]
            research = run_consensus(args.niche, client, args.runs, traces, db)
        else:
            trace = trace_path(args.niche) if args.trace else None
            research = run_research(args.niche, client, trace_file=trace, db=db)
    finally:
        if db:
            db.close()
    print(research.report(args.niche))
    print(f"\nCost ≈ ${research.usage.get('cost_usd', 0.0):.4f}", file=sys.stderr)
    if research.stopped:
        sys.exit(2)
