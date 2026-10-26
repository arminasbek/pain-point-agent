from collections import Counter
from itertools import combinations
from statistics import mean


# Share of items two sets have in common: 1.0 identical, 0.0 nothing shared.
def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def thread_stability(runs: list[set[str]]) -> dict[str, float]:
    runs_per_thread = Counter(url for urls in runs for url in urls)
    repeated = sum(1 for count in runs_per_thread.values() if count >= 2)
    # Two runs that both found nothing are not agreement, so those pairs are skipped.
    pair_overlaps = [jaccard(a, b) for a, b in combinations(runs, 2) if a or b]
    return {
        "runs": len(runs),
        "runs_with_findings": sum(1 for urls in runs if urls),
        "threads": len(runs_per_thread),
        "threads_in_2_plus_runs": repeated / len(runs_per_thread) if runs_per_thread else 0.0,
        "mean_pairwise_jaccard": mean(pair_overlaps) if pair_overlaps else 0.0,
    }


def judge_consistency(verdicts: list[list[bool]], labels: list[bool]) -> dict[str, float]:
    majorities = [sum(repeats) * 2 > len(repeats) for repeats in verdicts]
    return {
        "cases": len(verdicts),
        "repeats": len(verdicts[0]) if verdicts else 0,
        "all_repeats_agree": sum(len(set(repeats)) == 1 for repeats in verdicts) / len(verdicts),
        "majority_correct": sum(vote == label for vote, label in zip(majorities, labels)) / len(verdicts),
    }
