from pain_point_agent.judge import Verdict
from pain_point_agent.voting import consensus_pains, majority

A, B, C, D = "https://x/a", "https://x/b", "https://x/c", "https://x/d"


def evidence(url, quote="quote text long enough"):
    return {"source": "n8n", "title": url, "url": url, "quote": quote, "date": ""}


def test_majority_vote_takes_the_most_common_verdict():
    answers = iter([Verdict(True, "yes 1"), Verdict(False, "no"), Verdict(True, "yes 2")])
    vote = majority(lambda quote, post: next(answers), 3)

    assert vote("q", "p") == Verdict(True, "yes 1")


def test_majority_vote_tie_rejects():
    answers = iter([Verdict(True, "yes"), Verdict(False, "no")])
    vote = majority(lambda quote, post: next(answers), 2)

    assert vote("q", "p") == Verdict(False, "no")


def test_single_vote_is_one_call():
    calls = []
    vote = majority(lambda quote, post: calls.append(1) or Verdict(True, "yes"), 1)

    vote("q", "p")

    assert calls == [1]


def test_consensus_keeps_problems_found_in_most_runs():
    runs = [
        {"Bot goes silent": [evidence(A), evidence(B)], "Pricing": [evidence(C), evidence(D)]},
        {"Bot stops replying": [evidence(A), evidence(C)]},
        {"Silent bot": [evidence(B), evidence(A)]},
    ]

    pains = consensus_pains(runs)

    assert list(pains) == ["Bot goes silent"]
    assert [item["url"] for item in pains["Bot goes silent"]] == [A, B]


def test_consensus_drops_problems_from_a_single_run():
    runs = [{"Pricing": [evidence(C), evidence(D)]}, {}, {}]

    assert consensus_pains(runs) == {}
