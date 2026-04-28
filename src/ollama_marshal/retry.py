"""Transient-failure retry helper for Ollama HTTP calls.

Wraps the scheduler's `forward_request` so that brief Ollama hiccups
(connect refused while the daemon recycles, transient 502/503 from a
overloaded backend) get absorbed before the failure reaches the client.

Design constraints:

- **Streaming responses are NEVER retried.** Once chunks have started
  flowing, marshal can't safely re-issue the request without re-emitting
  partial output to the client. This matches openai-python, litellm,
  and anthropic-sdk-python behavior.
- **ReadTimeout is NOT retried by default for non-idempotent endpoints.**
  Ollama may have already started generating; retrying could double-bill
  the model run or, worse, re-execute a tool call. Configurable via
  `retry.read_timeouts: false` (the default).
- **Only ConnectError, ConnectTimeout, and 502/503/504 statuses retry.**
  These are unambiguously "Ollama hadn't started processing yet."
- **Per-request override**: clients can set `X-Marshal-Retry-Max` to
  override or disable retries on individual requests.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

import httpx
import structlog

logger = structlog.get_logger()


# HTTP status codes that mean "Ollama wasn't ready" (transient).
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({502, 503, 504})

# Exception types that are unambiguously pre-processing (safe to retry).
SAFE_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
)

# Exception types that MAY have caused server-side work to start. Only
# retried when caller explicitly opts in (e.g. for embeddings, which are
# idempotent).
UNSAFE_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)


T = TypeVar("T")


class _RetryableStatusError(Exception):
    """Internal sentinel raised when a non-streaming response has a retryable status.

    Lets us reuse the same retry loop for HTTP-status retries and
    exception-based retries.
    """

    def __init__(self, status_code: int) -> None:
        super().__init__(f"retryable status {status_code}")
        self.status_code = status_code


async def call_with_retry(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int,
    base_delay_s: float,
    max_delay_s: float,
    retry_read_timeouts: bool,
    request_id: str,
    **kwargs: Any,
) -> tuple[T, int]:
    """Invoke `func(*args, **kwargs)` with retry on transient failures.

    Streaming responses (callable returns an `AsyncIterator`) are returned
    immediately on first success and NEVER retried mid-flight.

    Non-streaming responses with a retryable status code (502/503/504)
    are retried up to `max_attempts` total tries.

    Args:
        func: The async callable to invoke (e.g. forward_request).
        *args: Positional args forwarded to `func`.
        max_attempts: Total number of attempts including the first
            (so 1 = no retry, 2 = one retry).
        base_delay_s: First-retry backoff in seconds. Doubles on each
            subsequent attempt.
        max_delay_s: Cap on a single backoff sleep.
        retry_read_timeouts: When True, ReadTimeout is also retryable.
            Default False (caller decides per call site — embeddings can
            opt in safely).
        request_id: Opaque ID for log correlation. Lets a reader trace
            "request X failed twice then succeeded."
        **kwargs: Keyword args forwarded to `func`.

    Returns:
        `(result, attempts_used)` — `result` is whatever `func`
        returned on the first successful attempt; `attempts_used` is the
        attempt number that succeeded (1 = first try, no retry used;
        2 = succeeded on first retry; etc.). Lets the caller record
        retry metrics cleanly.

    Raises:
        The last exception or HTTP-status error after attempts are
        exhausted.
    """
    if max_attempts < 1:
        msg = f"max_attempts must be >= 1, got {max_attempts}"
        raise ValueError(msg)

    retryable_excs: tuple[type[Exception], ...] = SAFE_RETRY_EXCEPTIONS
    if retry_read_timeouts:
        retryable_excs = SAFE_RETRY_EXCEPTIONS + UNSAFE_RETRY_EXCEPTIONS

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = await func(*args, **kwargs)
        except retryable_excs as exc:
            last_exc = exc
            if attempt == max_attempts:
                logger.warning(
                    "retry.exhausted",
                    request_id=request_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error_type=type(exc).__name__,
                )
                raise
            delay = _backoff_delay(attempt, base_delay_s, max_delay_s)
            logger.info(
                "retry.attempt_failed",
                request_id=request_id,
                attempt=attempt,
                max_attempts=max_attempts,
                error_type=type(exc).__name__,
                next_delay_s=round(delay, 3),
            )
            await asyncio.sleep(delay)
            continue

        # Streaming results are async iterators — return immediately,
        # never retry.
        if isinstance(result, AsyncIterator):
            return result, attempt

        # Non-streaming: inspect status code.
        if (
            isinstance(result, httpx.Response)
            and result.status_code in RETRYABLE_STATUS_CODES
        ):
            if attempt == max_attempts:
                logger.warning(
                    "retry.exhausted_on_status",
                    request_id=request_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status_code=result.status_code,
                )
                # Caller still gets the failed response, not an exception
                # — they can decide whether to surface it or transform.
                return result, attempt
            delay = _backoff_delay(attempt, base_delay_s, max_delay_s)
            logger.info(
                "retry.status_failed",
                request_id=request_id,
                attempt=attempt,
                max_attempts=max_attempts,
                status_code=result.status_code,
                next_delay_s=round(delay, 3),
            )
            await asyncio.sleep(delay)
            continue

        # Success.
        if attempt > 1:
            logger.info(
                "retry.succeeded",
                request_id=request_id,
                attempts_used=attempt,
            )
        return result, attempt

    # Unreachable: loop body always returns or raises in the final
    # iteration. Keep mypy happy.
    if last_exc is not None:
        raise last_exc
    msg = "retry loop exited without result or exception"
    raise RuntimeError(msg)


def _backoff_delay(attempt: int, base_delay_s: float, max_delay_s: float) -> float:
    """Exponential backoff with full jitter.

    Full jitter (uniform random in [0, exp_delay]) reduces thundering
    herd on simultaneous-failure recovery (e.g., if Ollama briefly
    flapped, every queued request would otherwise retry at the same
    millisecond).
    """
    exp_delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
    # Backoff jitter — security context is irrelevant here.
    return random.uniform(0, exp_delay)  # noqa: S311
