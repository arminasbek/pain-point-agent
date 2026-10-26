import json
from collections.abc import Callable
from typing import Any, NamedTuple, cast

from anthropic.types import ToolParam

from pain_point_agent.config import settings
from pain_point_agent.research import Research, ResearchError

ENABLED_SOURCES = {name: source for name, source in settings.sources.items() if source.enabled}

SOURCE_PROPERTY = {
    "type": "string",
    "enum": list(ENABLED_SOURCES),
    "description": "Where to look. " + "; ".join(f"{name}: {source.about}" for name, source in ENABLED_SOURCES.items()),
}

TOOLS: list[ToolParam] = [
    {
        "name": "search",
        "strict": True,
        "description": f"Search one source. At most {settings.max_searches} searches per run and {settings.max_searches_per_source} per source, so spread them over the sources that fit the niche. Returns up to 10 threads with id, title, posts_count, created_at, url and a matched text snippet (blurb).",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": SOURCE_PROPERTY,
                "query": {"type": "string", "description": "Search terms, e.g. 'whatsapp bot stops replying'"},
            },
            "required": ["source", "query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch",
        "strict": True,
        "description": "Read one thread by id (from search results of the same source). Returns the posts as plain text with post number and author. Long threads are cut off; truncated=true when that happens.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": SOURCE_PROPERTY,
                "id": {"type": "integer", "description": "Thread id from a search result"},
            },
            "required": ["source", "id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "record_pain",
        "strict": True,
        "description": "Record one piece of evidence for a problem. The thread must be fetched first and the quote must appear word for word in one of its posts, otherwise it is rejected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pain": {
                    "type": "string",
                    "description": "Short name of the problem. Use the exact same text to add more threads to an existing problem.",
                },
                "source": SOURCE_PROPERTY,
                "id": {"type": "integer", "description": "Id of a thread you fetched"},
                "quote": {
                    "type": "string",
                    "description": f"Copied word for word from one post, at least {settings.min_quote_chars} characters",
                },
            },
            "required": ["pain", "source", "id", "quote"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "strict": True,
        "description": f"End the research. Rejected while fewer than {settings.min_pains} problems have {settings.min_threads_per_pain}+ threads and searches are left. The report is built from recorded evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "Anything the reader should know, e.g. why fewer problems were found. Do not restate counts, the report computes them from recorded evidence. Can be empty.",
                },
            },
            "required": ["note"],
            "additionalProperties": False,
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "search": Research.search,
    "fetch": Research.fetch,
    "record_pain": Research.record_pain,
    "finish": Research.finish,
}

JSON_TYPES: dict[str, type] = {"string": str, "integer": int}


class ToolResult(NamedTuple):
    content: str
    is_error: bool
    failed: bool = False


def validate(name: str, tool_input: dict[str, Any]) -> str | None:
    tool = next((tool for tool in TOOLS if tool["name"] == name), None)
    if tool is None:
        return f"Unknown tool: {name}"
    schema = cast(dict[str, Any], tool["input_schema"])
    properties = schema["properties"]
    extra = set(tool_input) - set(properties)
    if extra:
        return f"Unexpected arguments: {sorted(extra)}"
    missing = set(schema["required"]) - set(tool_input)
    if missing:
        return f"Missing arguments: {sorted(missing)}"
    for key, value in tool_input.items():
        spec = properties[key]
        if isinstance(value, bool) or not isinstance(value, JSON_TYPES[spec["type"]]):
            return f"Argument '{key}' must be {spec['type']}"
        if "enum" in spec and value not in spec["enum"]:
            return f"Argument '{key}' must be one of {spec['enum']}"
    return None


def run_tool(name: str, tool_input: dict[str, Any], research: Research) -> ToolResult:
    error = validate(name, tool_input)
    if error:
        return ToolResult(error, True)
    try:
        return ToolResult(json.dumps(TOOL_FUNCTIONS[name](research, **tool_input), ensure_ascii=False), False)
    except ResearchError as error:
        return ToolResult(str(error), True)
    except Exception as error:
        return ToolResult(f"{type(error).__name__}: {error}", True, failed=True)
