from __future__ import annotations

from typing import Any, Dict

DAY_MIN = 24 * 60

def to_min(hhmm: str) -> int:
    # Convert 'HH:MM' to minutes from midnight.
    hhmm = hhmm.strip()
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def to_hhmm(m: int) -> str:
    m = int(m)
    m = max(0, min(DAY_MIN, m))
    h = m // 60
    mm = m % 60
    return f"{h:02d}:{mm:02d}"

def deep_get(d: Dict[str, Any], path: str, default: Any=None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
def min_to_hhmm(minutes: int) -> str:
    """Convert minutes-from-midnight to HH:MM string."""
    if minutes is None:
        return ""
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"