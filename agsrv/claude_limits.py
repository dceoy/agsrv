"""Normalize Claude's local subscription usage output."""

from __future__ import annotations

import re
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_LABELS = {
    "Current session": "five_hour",
    "Current week (all models)": "seven_day",
    "Current week (Opus)": "seven_day_opus",
    "Current week (Sonnet)": "seven_day_sonnet",
}
_LINE = re.compile(r"^(.+?):\s*(\d+)% used(?:\s*·\s*resets\s*(.*))?\s*$")
_ZONE = re.compile(r"\s*\(([^()]*)\)\s*$")


def _reset(value: str, now: datetime) -> str | None:
    zone_match = _ZONE.search(value)
    zone = UTC
    if zone_match:
        with suppress(ZoneInfoNotFoundError, ValueError):
            zone = ZoneInfo(zone_match.group(1))
        value = value[: zone_match.start()]
    local_now = now.astimezone(zone)
    try:
        result = datetime.strptime(  # timezone is attached below
            f"{value.strip()} {local_now.year}", "%b %d, %I:%M%p %Y"
        ).replace(tzinfo=zone)
        if result < local_now - timedelta(hours=12):
            result = result.replace(year=result.year + 1)
    except ValueError:
        return None
    return result.isoformat()


def parse_usage(raw: str, now: datetime | None = None) -> dict[str, Any]:
    """Parse recognized usage windows while preserving the full text.

    Args:
        raw: Claude's local command output.
        now: Optional timezone-aware clock for deterministic parsing.

    Returns:
        Normalized windows, availability, original text and fetch time.
    """
    now = now or datetime.now(UTC)
    windows: dict[str, Any] = dict.fromkeys(_LABELS.values())
    for line in raw.splitlines():
        match = _LINE.match(line.strip())
        if match and match.group(1) in _LABELS:
            windows[_LABELS[match.group(1)]] = {
                "used_percent": int(match.group(2)),
                "resets_at": _reset(match.group(3) or "", now),
            }
    return {
        "available": any(window is not None for window in windows.values()),
        **windows,
        "raw": raw,
        "fetched_at": now.astimezone(UTC).isoformat(),
    }
