import json

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from pain_point_agent.db import Database
from pain_point_agent.judge import Fit, Verdict
from pain_point_agent.research import Research


def thread(item_id, text="every few days the bot just goes silent"):
    return {
        "id": item_id,
        "title": f"Thread {item_id}",
        "url": f"https://community.n8n.io/t/x/{item_id}",
        "posts_count": 1,
        "posts": [{"number": 1, "author": "a", "text": text, "created_at": "2026-10-01T10:00:00Z"}],
        "truncated": False,
    }


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "data" / "test.db")
    yield db
    db.close()


def rows(db, sql):
    with db.engine.connect() as connection:
        return [tuple(row) for row in connection.execute(text(sql))]


def research_with_evidence():
    research = Research(judge=lambda quote, post: Verdict(True, ""), fit=lambda pain, quote, post: Fit(True, ""))
    research.pains = {
        "Bot goes silent": [
            {"source": "n8n", "title": "T1", "url": "https://x/1", "quote": "bot goes silent", "date": "2026-10-01"},
            {"source": "n8n", "title": "T2", "url": "https://x/2", "quote": "it went silent", "date": ""},
        ],
        "Pricing": [{"source": "n8n", "title": "T3", "url": "https://x/3", "quote": "far too expensive", "date": ""}],
    }
    research.finished = True
    research.searches = 2
    research.note = "note"
    research.usage = {"rounds": 3, "cost_usd": 0.0123}
    return research


def test_creates_the_database_file_and_folder(tmp_path):
    path = tmp_path / "nested" / "pain.db"

    Database(path).close()

    assert path.exists()


def test_start_and_finish_run(db):
    run_id = db.start_run("WhatsApp bots")

    db.finish_run(run_id, research_with_evidence(), "# Pain report")

    run = rows(db, "SELECT niche, kind, finished, stopped, searches, cost_usd, report, usage FROM runs")[0]
    assert run[:7] == ("WhatsApp bots", "single", 1, "", 2, 0.0123, "# Pain report")
    assert json.loads(run[7])["rounds"] == 3


def test_evidence_is_saved_with_strength(db):
    run_id = db.start_run("WhatsApp bots")

    db.finish_run(run_id, research_with_evidence(), "")

    assert rows(db, "SELECT pain, strong, url, quote, post_date FROM evidence ORDER BY id") == [
        ("Bot goes silent", 1, "https://x/1", "bot goes silent", "2026-10-01"),
        ("Bot goes silent", 1, "https://x/2", "it went silent", ""),
        ("Pricing", 0, "https://x/3", "far too expensive", ""),
    ]


def test_search_call_is_saved_with_result_count(db):
    run_id = db.start_run("n")

    db.tool_call(run_id, "search", {"source": "hn", "query": "bot banned"}, json.dumps([{"id": 1}, {"id": 2}]), False)

    assert rows(db, "SELECT run_id, source, query, results FROM searches") == [(run_id, "hn", "bot banned", 2)]


def test_fetched_thread_and_posts_are_saved_once(db):
    first = db.start_run("n")
    second = db.start_run("n")

    db.tool_call(first, "fetch", {"source": "n8n", "id": 7}, json.dumps(thread(7, "old text")), False)
    db.tool_call(second, "fetch", {"source": "n8n", "id": 7}, json.dumps(thread(7, "new text")), False)

    assert rows(db, "SELECT source, item_id, title FROM threads") == [("n8n", 7, "Thread 7")]
    assert rows(db, "SELECT number, author, created_at, text FROM posts") == [
        (1, "a", "2026-10-01T10:00:00Z", "new text")
    ]
    assert rows(db, "SELECT run_id FROM run_threads ORDER BY run_id") == [(first,), (second,)]


def test_rejected_call_is_saved_with_its_reason(db):
    run_id = db.start_run("n")
    tool_input = {"pain": "Bot goes silent", "source": "n8n", "id": 1, "quote": "q"}

    db.tool_call(run_id, "record_pain", tool_input, "Quote is not a complaint: Advice.", True)

    tool, saved_input, reason = rows(db, "SELECT tool, input, reason FROM rejections")[0]
    assert (tool, json.loads(saved_input), reason) == ("record_pain", tool_input, "Quote is not a complaint: Advice.")


def test_failed_tool_call_is_saved_as_an_error_not_a_rejection(db):
    run_id = db.start_run("n")

    db.tool_call(run_id, "fetch", {"source": "appstore", "id": 1}, "ValueError: The review feed returned nothing", True, True)

    assert rows(db, "SELECT COUNT(*) FROM rejections") == [(0,)]
    stage, saved_input, error = rows(db, "SELECT stage, input, error FROM errors")[0]
    assert (stage, json.loads(saved_input), error) == (
        "fetch",
        {"source": "appstore", "id": 1},
        "ValueError: The review feed returned nothing",
    )


def test_finished_run_counts_its_errors(db):
    run_id = db.start_run("n")
    other = db.start_run("n")
    db.error(run_id, "theme", "ThemeError('cut off')")
    db.error(run_id, "fetch", "HTTPStatusError: 503", {"source": "hn", "id": 1})
    db.error(other, "model", "max_tokens")

    db.finish_run(run_id, research_with_evidence(), "")

    assert rows(db, f"SELECT errors FROM runs WHERE id = {run_id}") == [(2,)]


def test_accepted_record_pain_is_not_a_rejection(db):
    run_id = db.start_run("n")

    db.tool_call(run_id, "record_pain", {"pain": "p"}, json.dumps({"pain": "p", "threads": 1}), False)

    assert rows(db, "SELECT COUNT(*) FROM rejections") == [(0,)]


def test_consensus_runs_are_marked(db):
    run_id = db.start_run("n", kind="consensus")

    assert rows(db, f"SELECT kind FROM runs WHERE id = {run_id}") == [("consensus",)]


def test_child_run_points_to_its_parent(db):
    parent = db.start_run("n", kind="consensus")
    child = db.start_run("n", parent_id=parent)

    assert rows(db, f"SELECT parent_id FROM runs WHERE id = {child}") == [(parent,)]


def test_foreign_keys_are_enforced(db):
    with pytest.raises(IntegrityError):
        db.tool_call(999, "search", {"source": "hn", "query": "q"}, "[]", False)


def test_new_database_is_migrated_to_the_latest_version(db):
    assert rows(db, "SELECT version_num FROM alembic_version") == [("0003",)]


def test_opening_an_existing_database_keeps_its_data(tmp_path):
    path = tmp_path / "pain.db"
    first = Database(path)
    first.start_run("kept")
    first.close()

    second = Database(path)

    assert rows(second, "SELECT niche FROM runs") == [("kept",)]
    second.close()
