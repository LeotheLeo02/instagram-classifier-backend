"""Async token bucket rate limiter utilities."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable, Optional, TypeVar


T = TypeVar("T")


class RateLimiter:
    """Token-bucket rate limiter with temporary penalty support."""

    def __init__(self, qps: float, burst: int = 1, name: str = ""):
        self.qps = max(qps, 0.001)
        self.burst = max(burst, 1)
        self._tokens = float(self.burst)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()
        self._multiplier = 1.0
        self.name = name

    @property
    def multiplier(self) -> float:
        return self._multiplier

    async def acquire(self, tokens: float = 1.0) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                refill_rate = self.qps / self._multiplier
                self._tokens = min(self.burst, self._tokens + (now - self._updated) * refill_rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                needed = tokens - self._tokens
                wait = needed / refill_rate if refill_rate > 0 else 1.0
            await asyncio.sleep(wait)

    def penalize(self, factor: float, duration: float) -> None:
        factor = max(1.0, factor)

        async def _penalty() -> None:
            previous = self._multiplier
            self._multiplier = max(self._multiplier, factor)
            try:
                await asyncio.sleep(duration)
            finally:
                if self._multiplier <= factor:
                    self._multiplier = previous

        asyncio.create_task(_penalty())


async def with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    limiter: RateLimiter,
    what: str,
    max_attempts: int = 5,
    base_delay: float = 1.5,
    hard_penalty_factor: float = 4.0,
    hard_penalty_seconds: float = 180.0,
    soft_penalty_factor: float = 2.0,
    soft_penalty_seconds: float = 120.0,
    on_429: Optional[Callable[[int], None]] = None,
    on_5xx: Optional[Callable[[int], None]] = None,
) -> T:
    for attempt in range(max_attempts):
        await limiter.acquire()
        resp = await call()
        status = getattr(resp, "status", None)

        if status in (200, 204, 304) or status is None:
            return resp

        if status == 429:
            if on_429:
                on_429(attempt)
            retry_after = None
            try:
                headers = resp.headers()
                retry_after_header = headers.get("Retry-After") or headers.get("retry-after")
                if retry_after_header is not None:
                    retry_after = float(retry_after_header)
            except Exception:
                retry_after = None

            delay = retry_after if retry_after is not None else base_delay * (2 ** attempt)
            limiter.penalize(factor=hard_penalty_factor, duration=min(hard_penalty_seconds, delay * 2))
            await asyncio.sleep(delay + random.random() * 0.5)
            continue

        if isinstance(status, int) and status >= 500:
            if on_5xx:
                on_5xx(attempt)
            limiter.penalize(
                factor=soft_penalty_factor,
                duration=min(soft_penalty_seconds, base_delay * (2 ** attempt)),
            )
            await asyncio.sleep(base_delay * (2 ** attempt) + random.random() * 0.5)
            continue

        try:
            body = await resp.text()
            snippet = (body or "")[:300].replace("\n", " ")
        except Exception:
            snippet = "<unreadable>"
        raise RuntimeError(f"{what} returned {status}: {snippet}")

    raise RuntimeError(f"{what}: exceeded {max_attempts} attempts")


