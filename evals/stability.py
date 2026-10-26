import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from pain_point_agent.client import make_client
from pain_point_agent.config import settings
from pain_point_agent.loop import run_research
from pain_point_agent.metrics import thread_stability

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one niche several times and measure how stable the findings are.")
    parser.add_argument("niche")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    client = make_client()
    runs = []
    for number in range(1, args.runs + 1):
        research = run_research(args.niche, client)
        strong = research.strong_pains()
        runs.append(
            {
                "problems": list(strong),
                "threads": sorted({item["url"] for items in strong.values() for item in items}),
                "stopped": research.stopped,
                "cost_usd": research.usage.get("cost_usd", 0.0),
            }
        )
        print(f"run {number}: {len(strong)} problems, {len(runs[-1]['threads'])} threads, ${runs[-1]['cost_usd']:.4f}")

    stability = thread_stability([set(run["threads"]) for run in runs])
    result = {"niche": args.niche, "model": settings.model, "stability": stability, "runs": runs}
    RESULTS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", args.niche.lower()).strip("-")
    path = RESULTS / f"stability-{slug}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\nruns with findings: {stability['runs_with_findings']} of {stability['runs']}")
    print(f"threads found: {stability['threads']}")
    print(f"threads found by 2+ runs: {stability['threads_in_2_plus_runs']:.0%}")
    print(f"mean pairwise Jaccard: {stability['mean_pairwise_jaccard']:.2f}")
    print(f"total cost: ${sum(run['cost_usd'] for run in runs):.4f}")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
