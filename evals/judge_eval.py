import json
import sys
from datetime import datetime
from pathlib import Path

from pain_point_agent.client import make_client
from pain_point_agent.config import settings
from pain_point_agent.judge import judge_quote
from pain_point_agent.trace import Metered, Trace

CASES = Path(__file__).parent / "judge_cases.jsonl"
RESULTS = Path(__file__).parent / "results"


def main() -> None:
    cases = [json.loads(line) for line in CASES.read_text().splitlines()]
    trace = Trace()
    client = Metered(make_client(), trace)
    wrong = []
    for case in cases:
        verdict = judge_quote(case["quote"], case["text"], client)
        if verdict.complaint != case["complaint"]:
            wrong.append((case, verdict))

    false_accepts = [case for case, _ in wrong if not case["complaint"]]
    false_rejects = [case for case, _ in wrong if case["complaint"]]
    path = save(cases, wrong, false_accepts, false_rejects, trace)

    print(f"model: {settings.model}  cases: {len(cases)}  correct: {len(cases) - len(wrong)}/{len(cases)}")
    print(f"accepted non-complaints: {len(false_accepts)}  rejected complaints: {len(false_rejects)}")
    for case, verdict in wrong:
        print(f"\n[label={case['complaint']}] {case['quote']}\n  judge: {verdict.reason}\n  {case['url']} #{case['post']}")
    print(f"cost: ${trace.cost():.4f}")
    print(f"saved: {path}")
    sys.exit(1 if wrong else 0)


def save(cases: list, wrong: list, false_accepts: list, false_rejects: list, trace: Trace) -> Path:
    result = {
        "model": settings.model,
        "cases": len(cases),
        "correct": len(cases) - len(wrong),
        "accepted_non_complaints": len(false_accepts),
        "rejected_complaints": len(false_rejects),
        "wrong": [
            {"quote": case["quote"], "label": case["complaint"], "reason": verdict.reason, "url": case["url"]}
            for case, verdict in wrong
        ],
        "cost_usd": trace.cost(),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"judge-eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return path


if __name__ == "__main__":
    main()
