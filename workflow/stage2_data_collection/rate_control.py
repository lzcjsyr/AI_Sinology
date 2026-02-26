from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import monotonic
from typing import Awaitable, Callable


@dataclass(frozen=True)
class RateLimits:
    rpm: int
    tpm: int

    def normalized(self) -> "RateLimits":
        return RateLimits(rpm=max(1, int(self.rpm)), tpm=max(1, int(self.tpm)))


@dataclass(frozen=True)
class AcquireReservation:
    estimated_tokens: int
    wait_seconds: float
    throttled: bool


class DualRateLimiter:
    """Async dual limiter for requests/minute and tokens/minute."""

    def __init__(
        self,
        *,
        name: str,
        limits: RateLimits,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        safe = limits.normalized()
        self.name = name
        self.limits = safe
        self._clock = clock or monotonic
        self._sleep = sleep or asyncio.sleep
        self._lock = asyncio.Lock()

        self._req_capacity = float(safe.rpm)
        self._tok_capacity = float(safe.tpm)
        self._req_refill_per_second = self._req_capacity / 60.0
        self._tok_refill_per_second = self._tok_capacity / 60.0

        self._req_tokens = self._req_capacity
        self._tok_tokens = self._tok_capacity
        self._last_refill = self._clock()

    def _refill_locked(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_refill)
        if elapsed <= 0:
            return
        self._req_tokens = min(
            self._req_capacity,
            self._req_tokens + elapsed * self._req_refill_per_second,
        )
        self._tok_tokens = min(
            self._tok_capacity,
            self._tok_tokens + elapsed * self._tok_refill_per_second,
        )
        self._last_refill = now

    async def acquire(self, estimated_tokens: int) -> AcquireReservation:
        # A single request may be estimated above minute-capacity. Clamp it to
        # avoid an impossible wait loop while still charging the fullest window.
        reserved_tokens = max(1, min(int(estimated_tokens), int(self._tok_capacity)))
        total_wait = 0.0
        throttled = False

        while True:
            async with self._lock:
                now = self._clock()
                self._refill_locked(now)

                req_short = max(0.0, 1.0 - self._req_tokens)
                tok_short = max(0.0, float(reserved_tokens) - self._tok_tokens)
                if req_short <= 0 and tok_short <= 0:
                    self._req_tokens -= 1.0
                    self._tok_tokens -= float(reserved_tokens)
                    return AcquireReservation(
                        estimated_tokens=reserved_tokens,
                        wait_seconds=total_wait,
                        throttled=throttled,
                    )

                wait_req = req_short / self._req_refill_per_second if req_short > 0 else 0.0
                wait_tok = tok_short / self._tok_refill_per_second if tok_short > 0 else 0.0
                sleep_for = max(wait_req, wait_tok, 0.001)

            throttled = True
            total_wait += sleep_for
            await self._sleep(sleep_for)

    async def commit(self, reservation: AcquireReservation, actual_tokens: int | None) -> None:
        if actual_tokens is None:
            return
        actual = max(1, int(actual_tokens))
        delta = int(reservation.estimated_tokens) - actual
        if delta == 0:
            return

        async with self._lock:
            now = self._clock()
            self._refill_locked(now)
            self._tok_tokens += float(delta)
            if self._tok_tokens > self._tok_capacity:
                self._tok_tokens = self._tok_capacity
            min_floor = -self._tok_capacity
            if self._tok_tokens < min_floor:
                self._tok_tokens = min_floor


def build_lowest_shared_limits(
    *,
    llm1_limits: RateLimits,
    llm2_limits: RateLimits,
    headroom: float,
) -> RateLimits:
    safe_headroom = max(0.01, min(1.0, float(headroom)))
    rpm = max(1, int(min(llm1_limits.rpm, llm2_limits.rpm) * safe_headroom))
    tpm = max(1, int(min(llm1_limits.tpm, llm2_limits.tpm) * safe_headroom))
    return RateLimits(rpm=rpm, tpm=tpm)
