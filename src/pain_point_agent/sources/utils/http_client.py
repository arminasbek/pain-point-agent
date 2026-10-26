import time
from typing import Any

import httpx

from pain_point_agent.config import settings
from pain_point_agent.sources.models import SourceError


def get(url: str, params: dict[str, str] | None = None, headers: dict[str, str] | None = None) -> Any:
    for attempt in range(settings.max_retries + 1):
        last = attempt == settings.max_retries
        try:
            response = httpx.get(
                url,
                params=params,
                headers={"User-Agent": settings.user_agent, **(headers or {})},
                timeout=settings.http_timeout,
            )
        except httpx.TransportError:
            if last:
                raise
            time.sleep(_retry_delay(attempt))
            continue
        if response.status_code not in settings.retry_statuses or last:
            response.raise_for_status()
            try:
                return response.json()
            except ValueError as error:
                raise SourceError(f"{url} did not return JSON. The source may be failing.") from error
        time.sleep(_retry_delay(attempt, response))
    raise SourceError(f"{url}: no request attempted, check max_retries.")


def _retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    retry_after = response.headers.get("Retry-After", "") if response is not None else ""
    delay = float(retry_after) if retry_after.isdigit() else 2.0**attempt
    return min(delay, settings.max_retry_delay)
