import json
from pathlib import Path

from pain_point_agent.research import normalize

CASES = Path(__file__).parent.parent / "evals" / "judge_cases.jsonl"


def test_judge_cases_are_valid():
    cases = [json.loads(line) for line in CASES.read_text().splitlines()]

    assert len(cases) >= 20
    assert {case["complaint"] for case in cases} == {True, False}
    for case in cases:
        assert normalize(case["quote"]) in normalize(case["text"]), case["quote"]
