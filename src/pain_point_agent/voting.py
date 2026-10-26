from collections import Counter
from collections.abc import Callable
from typing import Any, TypeVar

from pain_point_agent.config import settings
from pain_point_agent.research import Evidence

Answer = TypeVar("Answer", bound=tuple)


def majority(judge: Callable[..., Answer], votes: int) -> Callable[..., Answer]:
    if votes <= 1:
        return judge

    def vote(*args: Any) -> Answer:
        answers = [judge(*args) for _ in range(votes)]
        yes = [answer for answer in answers if answer[0]]
        no = [answer for answer in answers if not answer[0]]
        return yes[0] if len(yes) > votes / 2 else no[0]

    return vote


def consensus_pains(runs: list[dict[str, list[Evidence]]]) -> dict[str, list[Evidence]]:
    needed = len(runs) // 2 + 1
    runs_per_thread: Counter[str] = Counter()
    for pains in runs:
        threads_in_run = {item["url"] for items in pains.values() for item in items}
        runs_per_thread.update(threads_in_run)
    confirmed = {url for url, count in runs_per_thread.items() if count >= needed}
    used: set[str] = set()
    result: dict[str, list[Evidence]] = {}
    for pains in runs:
        for pain, items in pains.items():
            kept: dict[str, Evidence] = {}
            for item in items:
                if item["url"] in confirmed and item["url"] not in used:
                    kept.setdefault(item["url"], item)
            if pain not in result and len(kept) >= settings.min_threads_per_pain:
                result[pain] = list(kept.values())
                used |= set(kept)
    return result
