import pytest

from pain_point_agent.metrics import judge_consistency, thread_stability

A, B, C = "https://x/a", "https://x/b", "https://x/c"


def test_identical_runs_are_fully_stable():
    result = thread_stability([{A, B}, {A, B}, {A, B}])

    assert result == {
        "runs": 3,
        "runs_with_findings": 3,
        "threads": 2,
        "threads_in_2_plus_runs": 1.0,
        "mean_pairwise_jaccard": 1.0,
    }


def test_disjoint_runs_have_zero_stability():
    result = thread_stability([{A}, {B}, {C}])

    assert result["threads_in_2_plus_runs"] == 0.0
    assert result["mean_pairwise_jaccard"] == 0.0


def test_partial_overlap():
    result = thread_stability([{A, B}, {A, C}, set()])

    assert result["threads"] == 3
    assert result["threads_in_2_plus_runs"] == pytest.approx(1 / 3)
    assert result["mean_pairwise_jaccard"] == pytest.approx((1 / 3 + 0 + 0) / 3)


def test_runs_that_both_found_nothing_are_not_counted_as_agreeing():
    result = thread_stability([set(), {A, B}, set(), set(), {A, B, C}])

    assert result["runs_with_findings"] == 2
    assert result["mean_pairwise_jaccard"] == pytest.approx((2 / 3) / 7)


def test_no_findings_at_all_is_not_stable():
    result = thread_stability([set(), set(), set()])

    assert result["runs_with_findings"] == 0
    assert result["mean_pairwise_jaccard"] == 0.0


def test_judge_consistency_counts_agreement_and_majority_accuracy():
    verdicts = [[True, True, True], [True, False, True], [False, False, False]]
    labels = [True, False, False]

    result = judge_consistency(verdicts, labels)

    assert result == {
        "cases": 3,
        "repeats": 3,
        "all_repeats_agree": pytest.approx(2 / 3),
        "majority_correct": pytest.approx(2 / 3),
    }
