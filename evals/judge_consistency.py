import argparse
import json
from datetime import datetime
from pathlib import Path

from pain_point_agent.client import make_client
from pain_point_agent.config import settings
from pain_point_agent.judge import judge_quote
from pain_point_agent.metrics import judge_consistency
from pain_point_agent.trace import Metered, Trace

CASES = Path(__file__).parent / "judge_cases.jsonl"
RESULTS = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the complaint judge the same cases several times.")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    cases = [json.loads(line) for line in CASES.read_text().splitlines()]
    trace = Trace()
    client = Metered(make_client(), trace)
    verdicts = [
        [judge_quote(case["quote"], case["text"], client).complaint for _ in range(args.repeats)] for case in cases
    ]
    result = judge_consistency(verdicts, [case["complaint"] for case in cases])
    flaky = [case["quote"] for case, repeats in zip(cases, verdicts) if len(set(repeats)) > 1]

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"judge-consistency-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"model": settings.model, **result, "flaky": flaky, "verdicts": verdicts}, indent=2))

    print(f"cases: {result['cases']}  repeats: {result['repeats']}")
    print(f"all repeats agree: {result['all_repeats_agree']:.0%}")
    print(f"majority correct: {result['majority_correct']:.0%}")
    for quote in flaky:
        print(f"  flaky: {quote[:100]}")
    print(f"cost: ${trace.cost():.4f}")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
