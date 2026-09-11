"""Small, process-local API security controls."""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque

from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        if limit < 1 or window_seconds < 1:
            raise ValueError("Rate-limit values must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            timestamps = self._requests.setdefault(key, deque())
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                return False
            timestamps.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()


def require_api_key(request: Request, expected_key: str | None) -> None:
    """Require X-API-Key only when API_KEY is configured."""
    if expected_key is None:
        return
    supplied = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(supplied, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
