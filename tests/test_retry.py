from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ollama_marshal.retry import (
    RETRYABLE_STATUS_CODES,
    SAFE_RETRY_EXCEPTIONS,
    UNSAFE_RETRY_EXCEPTIONS,
    _backoff_delay,
    call_with_retry,
)


# ---------------------------------------------------------------------------
# call_with_retry — exception paths
# ---------------------------------------------------------------------------


class TestCallWithRetrySafeExceptions:
    """ConnectError / ConnectTimeout — Ollama wasn't ready, safe to retry."""

    async def test_succeeds_on_first_attempt(self):
        func = AsyncMock(return_value=_ok_response())
        result, attempts = await call_with_retry(
            func,
            max_attempts=3,
            base_delay_s=0.001,
            max_delay_s=0.01,
            retry_read_timeouts=False,
            request_id="t",
        )
        assert attempts == 1
        assert func.call_count == 1
        assert result.status_code == 200

    async def test_retries_on_connect_error_and_succeeds(self):
        func = AsyncMock(
            side_effect=[
                httpx.ConnectError("refused"),
                httpx.ConnectError("refused"),
                _ok_response(),
            ]
        )
        result, attempts = await call_with_retry(
            func,
            max_attempts=3,
            base_delay_s=0.001,
            max_delay_s=0.01,
            retry_read_timeouts=False,
            request_id="t",
        )
        assert attempts == 3
        assert func.call_count == 3
        assert result.status_code == 200

    async def test_retries_on_connect_timeout(self):
        func = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("timeout"),
                _ok_response(),
            ]
        )
        result, attempts = await call_with_retry(
            func,
            max_attempts=3,
            base_delay_s=0.001,
            max_delay_s=0.01,
            retry_read_timeouts=False,
            request_id="t",
        )
        assert attempts == 2
        assert result.status_code == 200

    async def test_exhausts_and_raises_last_exception(self):
        last_exc = httpx.ConnectError("permanent")
        func = AsyncMock(
            side_effect=[
                httpx.ConnectError("transient"),
                last_exc,
            ]
        )
        with pytest.raises(httpx.ConnectError):
            await call_with_retry(
                func,
                max_attempts=2,
                base_delay_s=0.001,
                max_delay_s=0.01,
                retry_read_timeouts=False,
                request_id="t",
            )
        assert func.call_count == 2


class TestCallWithRetryReadTimeouts:
    """ReadTimeout: retried only when caller opts in (idempotent endpoints)."""

    async def test_read_timeout_not_retried_by_default(self):
        # Default: ReadTimeout reraises immediately, no retry.
        func = AsyncMock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(httpx.ReadTimeout):
            await call_with_retry(
                func,
                max_attempts=3,
                base_delay_s=0.001,
                max_delay_s=0.01,
                retry_read_timeouts=False,
                request_id="t",
            )
        assert func.call_count == 1

    async def test_read_timeout_retried_when_enabled(self):
        func = AsyncMock(
            side_effect=[httpx.ReadTimeout("slow"), _ok_response()],
        )
        result, attempts = await call_with_retry(
            func,
            max_attempts=3,
            base_delay_s=0.001,
            max_delay_s=0.01,
            retry_read_timeouts=True,
            request_id="t",
        )
        assert attempts == 2
        assert result.status_code == 200


class TestCallWithRetryNonRetryableExceptions:
    async def test_value_error_reraises_immediately(self):
        # Non-network errors must NOT be retried — bug in our code.
        func = AsyncMock(side_effect=ValueError("bug"))
        with pytest.raises(ValueError):
            await call_with_retry(
                func,
                max_attempts=3,
                base_delay_s=0.001,
                max_delay_s=0.01,
                retry_read_timeouts=False,
                request_id="t",
            )
        assert func.call_count == 1


# ---------------------------------------------------------------------------
# call_with_retry — HTTP status paths
# ---------------------------------------------------------------------------


class TestCallWithRetryStatusCodes:
    @pytest.mark.parametrize("code", sorted(RETRYABLE_STATUS_CODES))
    async def test_retries_on_retryable_status(self, code):
        func = AsyncMock(
            side_effect=[
                _resp(code),
                _ok_response(),
            ]
        )
        result, attempts = await call_with_retry(
            func,
            max_attempts=3,
            base_delay_s=0.001,
            max_delay_s=0.01,
            retry_read_timeouts=False,
            request_id="t",
        )
        assert attempts == 2
        assert result.status_code == 200

    async def test_does_not_retry_on_4xx(self):
        # 400 / 404 are client errors — not transient.
        func = AsyncMock(return_value=_resp(404))
        result, attempts = await call_with_retry(
            func,
            max_attempts=3,
            base_delay_s=0.001,
            max_delay_s=0.01,
            retry_read_timeouts=False,
            request_id="t",
        )
        assert attempts == 1
        assert result.status_code == 404
        assert func.call_count == 1

    async def test_does_not_retry_on_500(self):
        # 500 is server error but NOT one of our retryable codes —
        # could be a real bug in Ollama, not a transient hiccup.
        func = AsyncMock(return_value=_resp(500))
        result, attempts = await call_with_retry(
            func,
            max_attempts=3,
            base_delay_s=0.001,
            max_delay_s=0.01,
            retry_read_timeouts=False,
            request_id="t",
        )
        assert attempts == 1
        assert result.status_code == 500

    async def test_returns_failed_response_after_exhausting_retries(self):
        # All retries returned 503 → caller still gets the response,
        # not an exception (lets them surface the underlying status).
        func = AsyncMock(return_value=_resp(503))
        result, attempts = await call_with_retry(
            func,
            max_attempts=2,
            base_delay_s=0.001,
            max_delay_s=0.01,
            retry_read_timeouts=False,
            request_id="t",
        )
        assert attempts == 2
        assert result.status_code == 503
        assert func.call_count == 2


# ---------------------------------------------------------------------------
# call_with_retry — streaming path
# ---------------------------------------------------------------------------


class TestCallWithRetryStreaming:
    """Streaming results are NEVER retried — we'd corrupt the byte stream."""

    async def test_streaming_result_returned_on_first_success(self):
        async def fake_stream() -> AsyncIterator[bytes]:
            yield b"chunk-1"

        func = AsyncMock(return_value=fake_stream())
        result, attempts = await call_with_retry(
            func,
            max_attempts=3,
            base_delay_s=0.001,
            max_delay_s=0.01,
            retry_read_timeouts=True,
            request_id="t",
        )
        assert attempts == 1
        # Sanity: it really is an async iterator.
        chunks = [c async for c in result]
        assert chunks == [b"chunk-1"]


# ---------------------------------------------------------------------------
# call_with_retry — argument validation + edge cases
# ---------------------------------------------------------------------------


class TestCallWithRetryArgs:
    async def test_max_attempts_zero_is_invalid(self):
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            await call_with_retry(
                AsyncMock(),
                max_attempts=0,
                base_delay_s=0.001,
                max_delay_s=0.01,
                retry_read_timeouts=False,
                request_id="t",
            )

    async def test_max_attempts_one_is_no_retry(self):
        # max_attempts=1 means a single call, exception passes through.
        func = AsyncMock(side_effect=httpx.ConnectError("bad"))
        with pytest.raises(httpx.ConnectError):
            await call_with_retry(
                func,
                max_attempts=1,
                base_delay_s=0.001,
                max_delay_s=0.01,
                retry_read_timeouts=False,
                request_id="t",
            )
        assert func.call_count == 1

    async def test_kwargs_passed_through(self):
        func = AsyncMock(return_value=_ok_response())
        await call_with_retry(
            func,
            max_attempts=2,
            base_delay_s=0.001,
            max_delay_s=0.01,
            retry_read_timeouts=False,
            request_id="t",
            ollama_host="http://x",
            endpoint="/api/chat",
        )
        func.assert_called_once_with(
            ollama_host="http://x",
            endpoint="/api/chat",
        )


# ---------------------------------------------------------------------------
# _backoff_delay
# ---------------------------------------------------------------------------


class TestBackoffDelay:
    def test_is_bounded_by_base_on_first_attempt(self):
        for _ in range(20):
            d = _backoff_delay(1, base_delay_s=1.0, max_delay_s=10.0)
            assert 0.0 <= d <= 1.0

    def test_doubles_until_max_delay(self):
        # Full jitter is uniform [0, exp_delay], so the upper bound
        # doubles each attempt up to max_delay_s.
        for _ in range(20):
            d2 = _backoff_delay(2, base_delay_s=1.0, max_delay_s=10.0)
            assert d2 <= 2.0

    def test_capped_at_max_delay(self):
        for _ in range(20):
            d = _backoff_delay(20, base_delay_s=1.0, max_delay_s=5.0)
            assert d <= 5.0


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestRetryConstants:
    def test_retryable_status_codes_include_502_503_504(self):
        # Documented behavior — pin the set so a refactor can't quietly
        # add 500 or 429 (those have different semantics).
        assert RETRYABLE_STATUS_CODES == frozenset({502, 503, 504})

    def test_safe_excs_only_connection_errors(self):
        # Anything in SAFE means "Ollama hadn't received the request yet."
        assert httpx.ConnectError in SAFE_RETRY_EXCEPTIONS
        assert httpx.ConnectTimeout in SAFE_RETRY_EXCEPTIONS
        # ReadTimeout is NOT safe — server may have started processing.
        assert httpx.ReadTimeout not in SAFE_RETRY_EXCEPTIONS

    def test_unsafe_excs_include_read_timeout(self):
        assert httpx.ReadTimeout in UNSAFE_RETRY_EXCEPTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(status: int) -> Any:
    """Return a MagicMock that quacks like httpx.Response for status checks."""
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    return r


def _ok_response() -> Any:
    return _resp(200)
