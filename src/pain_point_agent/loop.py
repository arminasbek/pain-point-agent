from collections.abc import Sequence
from functools import partial
from pathlib import Path

import anthropic
from anthropic.types import ContentBlock, Message, MessageParam, TextBlockParam, ToolResultBlockParam, ToolUseBlock

from pain_point_agent.config import settings
from pain_point_agent.db import Database
from pain_point_agent.judge import judge_fit, judge_quote
from pain_point_agent.prompts import load_prompt
from pain_point_agent.research import Research
from pain_point_agent.theme import ThemeError, find_theme
from pain_point_agent.tools import TOOLS, run_tool
from pain_point_agent.trace import Metered, Trace, save, show
from pain_point_agent.voting import consensus_pains, majority

SYSTEM_PROMPT = load_prompt("agent")


def run_research(
    niche: str,
    client: anthropic.Anthropic,
    max_steps: int | None = None,
    trace_file: Path | None = None,
    db: Database | None = None,
    parent_id: int | None = None,
) -> Research:
    max_steps = max_steps or settings.max_steps
    messages: list[MessageParam] = [{"role": "user", "content": f"Niche: {niche}"}]
    run_id = db.start_run(niche, parent_id=parent_id) if db else 0
    run = _RunContext(client, Trace(verbose=trace_file is not None), db, run_id)
    research = Research(
        judge=majority(partial(judge_quote, client=run.checks), settings.judge_votes),
        fit=majority(partial(judge_fit, client=run.checks, niche=niche), settings.judge_votes),
    )

    try:
        for step in range(max_steps):
            try:
                response = run.ask_agent(messages)
            except anthropic.APIError as error:
                run.stop(research, f"API error: {error}", repr(error))
                break
            run.trace.round(len(messages), response.usage)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                run.stop(research, f"model stopped with stop_reason={response.stop_reason}")
                break

            results = run.run_tools(response.content, research)
            if not research.finished:
                results.append({"type": "text", "text": research.status_for_model(max_steps - step - 1)})
            messages.append({"role": "user", "content": results})
            if research.finished:
                break
        else:
            run.stop(research, f"step limit of {max_steps} reached before finish", stage="budget")

        run.add_theme(niche, research)
        return research
    except BaseException as error:
        research.stopped = f"crashed: {error!r}"
        run.error("crash", repr(error))
        raise
    finally:
        research.usage = run.trace.summary()
        run.trace.totals()
        if db:
            db.finish_run(run_id, research, research.report(niche))
        if trace_file:
            save(messages, trace_file)


# Runs the agent N times from scratch and keeps only threads most runs found.
def run_consensus(
    niche: str,
    client: anthropic.Anthropic,
    runs: int,
    trace_files: Sequence[Path | None] | None = None,
    db: Database | None = None,
) -> Research:
    trace_files = trace_files or [None] * runs
    consensus_id = db.start_run(niche, kind="consensus") if db else None
    run = _RunContext(client, Trace(), db, consensus_id)
    combined = Research(
        judge=majority(partial(judge_quote, client=run.checks), settings.judge_votes),
        fit=majority(partial(judge_fit, client=run.checks, niche=niche), settings.judge_votes),
    )
    try:
        results = [
            run_research(niche, client, trace_file=trace_files[number], db=db, parent_id=consensus_id)
            for number in range(runs)
        ]
        combined.finished = all(result.finished for result in results)
        combined.stopped = "; ".join(
            f"run {number}: {result.stopped}" for number, result in enumerate(results, 1) if result.stopped
        )
        combined.pains = consensus_pains([result.strong_pains() for result in results])
        combined.searches = sum(result.searches for result in results)
        combined.fetched = {key: thread for result in results for key, thread in result.fetched.items()}
        needed = runs // 2 + 1
        combined.consensus = (
            f"Consensus of {runs} runs: a problem is kept when at least {settings.min_threads_per_pain} of its "
            f"threads were found by {needed} or more runs. "
            f"Problems per run: {', '.join(str(len(result.strong_pains())) for result in results)}."
        )
        combined.usage = {"cost_usd": sum(result.usage.get("cost_usd", 0.0) for result in results)}
        run.add_theme(niche, combined)
        combined.usage["cost_usd"] += run.trace.cost()
        return combined
    except BaseException as error:
        combined.stopped = f"crashed: {error!r}"
        run.error("crash", repr(error))
        raise
    finally:
        if db and consensus_id is not None:
            db.finish_run(consensus_id, combined, combined.report(niche))


class _RunContext:
    def __init__(self, client: anthropic.Anthropic, trace: Trace, db: Database | None, run_id: int | None) -> None:
        self.client = client
        self.trace = trace
        self.checks = Metered(client, trace)
        self.db = db
        self.run_id = run_id

    def error(self, stage: str, detail: str) -> None:
        if self.db and self.run_id is not None:
            self.db.error(self.run_id, stage, detail)

    def stop(self, research: Research, reason: str, detail: str | None = None, stage: str = "model") -> None:
        research.stopped = reason
        self.error(stage, detail or reason)

    def ask_agent(self, messages: list[MessageParam]) -> Message:
        return self.client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "any"},
            cache_control={"type": "ephemeral"},
            messages=messages,
        )

    def run_tools(self, blocks: list[ContentBlock], research: Research) -> list[ToolResultBlockParam | TextBlockParam]:
        results: list[ToolResultBlockParam | TextBlockParam] = []
        for block in blocks:
            if not isinstance(block, ToolUseBlock):
                continue
            self.trace.call(block.name, block.input)
            content, is_error, failed = run_tool(block.name, block.input, research)
            self.trace.result(content, is_error)
            if self.db and self.run_id is not None:
                self.db.tool_call(self.run_id, block.name, block.input, content, is_error, failed)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error})
        return results

    def add_theme(self, niche: str, research: Research) -> None:
        strong = research.strong_pains()
        if not strong:
            return
        try:
            research.theme = find_theme(niche, strong, self.checks)
        except (ThemeError, anthropic.APIError) as error:
            show(f"warning: theme step failed: {error}")
            self.error("theme", repr(error))
