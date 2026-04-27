from __future__ import annotations

import asyncio
import time

import pytest

from ollama_marshal.queue import ModelQueues, RequestEnvelope

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_envelope():
    def _factory(
        model="llama3:latest",
        program_id="test-app",
        endpoint="/api/chat",
        stream=False,
        arrived_at=None,
    ):
        kwargs = {
            "model": model,
            "program_id": program_id,
            "request_body": {"model": model, "messages": []},
            "endpoint": endpoint,
            "stream": stream,
        }
        if arrived_at is not None:
            kwargs["arrived_at"] = arrived_at
        return RequestEnvelope(**kwargs)

    return _factory


@pytest.fixture
def queues():
    return ModelQueues()


# ---------------------------------------------------------------------------
# RequestEnvelope
# ---------------------------------------------------------------------------


class TestRequestEnvelopeCreation:
    def test_default_fields(self, make_envelope):
        env = make_envelope()
        assert env.model == "llama3:latest"
        assert env.program_id == "test-app"
        assert env.endpoint == "/api/chat"
        assert env.stream is False
        assert env.skip_count == 0
        assert env.response is None
        assert env.error is None
        assert not env.done_event.is_set()

    def test_arrived_at_is_monotonic(self, make_envelope):
        before = time.monotonic()
        env = make_envelope()
        after = time.monotonic()
        assert before <= env.arrived_at <= after

    def test_custom_arrived_at(self, make_envelope):
        env = make_envelope(arrived_at=100.0)
        assert env.arrived_at == 100.0

    def test_stream_flag(self, make_envelope):
        env = make_envelope(stream=True)
        assert env.stream is True


class TestRequestEnvelopeSkipCounting:
    def test_increment_skip(self, make_envelope):
        env = make_envelope()
        assert env.skip_count == 0
        env.increment_skip()
        assert env.skip_count == 1
        env.increment_skip()
        assert env.skip_count == 2

    def test_is_unskippable_below_threshold(self, make_envelope):
        env = make_envelope()
        env.skip_count = 2
        assert env.is_unskippable(3) is False

    def test_is_unskippable_at_threshold(self, make_envelope):
        env = make_envelope()
        env.skip_count = 3
        assert env.is_unskippable(3) is True

    def test_is_unskippable_above_threshold(self, make_envelope):
        env = make_envelope()
        env.skip_count = 5
        assert env.is_unskippable(3) is True

    def test_is_unskippable_zero_skips(self, make_envelope):
        env = make_envelope()
        assert env.is_unskippable(0) is True  # 0 >= 0


class TestRequestEnvelopeComplete:
    def test_complete_sets_response_and_event(self, make_envelope):
        env = make_envelope()
        response = {"message": "hello"}
        env.complete(response)
        assert env.response == response
        assert env.done_event.is_set()

    def test_complete_with_none_response(self, make_envelope):
        env = make_envelope()
        env.complete(None)
        assert env.response is None
        assert env.done_event.is_set()


class TestRequestEnvelopeFail:
    def test_fail_sets_error_and_event(self, make_envelope):
        env = make_envelope()
        error = RuntimeError("something broke")
        env.fail(error)
        assert env.error is error
        assert env.done_event.is_set()


class TestRequestEnvelopeWaitTime:
    def test_wait_time_is_positive(self, make_envelope):
        env = make_envelope()
        assert env.wait_time >= 0

    def test_wait_time_with_known_arrived_at(self, make_envelope):
        past = time.monotonic() - 5.0
        env = make_envelope(arrived_at=past)
        assert env.wait_time >= 4.9  # allow small drift


# ---------------------------------------------------------------------------
# ModelQueues - enqueue / dequeue
# ---------------------------------------------------------------------------


class TestModelQueuesEnqueueDequeue:
    async def test_enqueue_and_dequeue_single(self, queues, make_envelope):
        env = make_envelope(model="llama3:latest")
        await queues.enqueue(env)
        result = await queues.dequeue("llama3:latest")
        assert result is env

    async def test_dequeue_empty_returns_none(self, queues):
        result = await queues.dequeue("nonexistent-model")
        assert result is None

    async def test_dequeue_after_drain_returns_none(self, queues, make_envelope):
        env = make_envelope()
        await queues.enqueue(env)
        await queues.dequeue("llama3:latest")
        result = await queues.dequeue("llama3:latest")
        assert result is None

    async def test_fifo_order(self, queues, make_envelope):
        e1 = make_envelope(model="llama3:latest", program_id="first")
        e2 = make_envelope(model="llama3:latest", program_id="second")
        e3 = make_envelope(model="llama3:latest", program_id="third")
        await queues.enqueue(e1)
        await queues.enqueue(e2)
        await queues.enqueue(e3)
        assert await queues.dequeue("llama3:latest") is e1
        assert await queues.dequeue("llama3:latest") is e2
        assert await queues.dequeue("llama3:latest") is e3

    async def test_enqueue_multiple_models(self, queues, make_envelope):
        e_llama = make_envelope(model="llama3:latest")
        e_mistral = make_envelope(model="mistral:latest")
        await queues.enqueue(e_llama)
        await queues.enqueue(e_mistral)
        assert await queues.dequeue("llama3:latest") is e_llama
        assert await queues.dequeue("mistral:latest") is e_mistral


# ---------------------------------------------------------------------------
# ModelQueues - dequeue_batch
# ---------------------------------------------------------------------------


class TestModelQueuesDequeueBatch:
    async def test_dequeue_batch_all(self, queues, make_envelope):
        envs = [make_envelope(model="llama3:latest") for _ in range(5)]
        for e in envs:
            await queues.enqueue(e)
        batch = await queues.dequeue_batch("llama3:latest")
        assert len(batch) == 5
        assert batch == envs

    async def test_dequeue_batch_with_max_count(self, queues, make_envelope):
        envs = [make_envelope(model="llama3:latest") for _ in range(5)]
        for e in envs:
            await queues.enqueue(e)
        batch = await queues.dequeue_batch("llama3:latest", max_count=3)
        assert len(batch) == 3
        assert batch == envs[:3]
        # Remaining 2 should still be in the queue
        remaining = await queues.dequeue_batch("llama3:latest")
        assert len(remaining) == 2

    async def test_dequeue_batch_max_count_exceeds_queue(self, queues, make_envelope):
        envs = [make_envelope(model="llama3:latest") for _ in range(2)]
        for e in envs:
            await queues.enqueue(e)
        batch = await queues.dequeue_batch("llama3:latest", max_count=10)
        assert len(batch) == 2

    async def test_dequeue_batch_empty_queue(self, queues):
        batch = await queues.dequeue_batch("nonexistent-model")
        assert batch == []

    async def test_dequeue_batch_none_max_count_takes_all(self, queues, make_envelope):
        envs = [make_envelope(model="llama3:latest") for _ in range(3)]
        for e in envs:
            await queues.enqueue(e)
        batch = await queues.dequeue_batch("llama3:latest", max_count=None)
        assert len(batch) == 3
        assert await queues.pending_count("llama3:latest") == 0


# ---------------------------------------------------------------------------
# ModelQueues - peek_models
# ---------------------------------------------------------------------------


class TestModelQueuesPeekModels:
    async def test_peek_empty(self, queues):
        models = await queues.peek_models()
        assert models == []

    async def test_peek_with_pending(self, queues, make_envelope):
        await queues.enqueue(make_envelope(model="llama3:latest"))
        await queues.enqueue(make_envelope(model="mistral:latest"))
        models = await queues.peek_models()
        assert set(models) == {"llama3:latest", "mistral:latest"}

    async def test_peek_excludes_drained_models(self, queues, make_envelope):
        await queues.enqueue(make_envelope(model="llama3:latest"))
        await queues.enqueue(make_envelope(model="mistral:latest"))
        await queues.dequeue("llama3:latest")
        models = await queues.peek_models()
        assert models == ["mistral:latest"]


# ---------------------------------------------------------------------------
# ModelQueues - pending counts
# ---------------------------------------------------------------------------


class TestModelQueuesPendingCounts:
    async def test_pending_count_empty(self, queues):
        assert await queues.pending_count("llama3:latest") == 0

    async def test_pending_count_after_enqueue(self, queues, make_envelope):
        await queues.enqueue(make_envelope(model="llama3:latest"))
        await queues.enqueue(make_envelope(model="llama3:latest"))
        assert await queues.pending_count("llama3:latest") == 2

    async def test_pending_count_after_dequeue(self, queues, make_envelope):
        await queues.enqueue(make_envelope(model="llama3:latest"))
        await queues.enqueue(make_envelope(model="llama3:latest"))
        await queues.dequeue("llama3:latest")
        assert await queues.pending_count("llama3:latest") == 1

    async def test_total_pending_empty(self, queues):
        assert await queues.total_pending() == 0

    async def test_total_pending_multiple_models(self, queues, make_envelope):
        await queues.enqueue(make_envelope(model="llama3:latest"))
        await queues.enqueue(make_envelope(model="llama3:latest"))
        await queues.enqueue(make_envelope(model="mistral:latest"))
        assert await queues.total_pending() == 3

    async def test_pending_by_model_empty(self, queues):
        result = await queues.pending_by_model()
        assert result == {}

    async def test_pending_by_model(self, queues, make_envelope):
        await queues.enqueue(make_envelope(model="llama3:latest"))
        await queues.enqueue(make_envelope(model="llama3:latest"))
        await queues.enqueue(make_envelope(model="mistral:latest"))
        result = await queues.pending_by_model()
        assert result == {"llama3:latest": 2, "mistral:latest": 1}

    async def test_pending_by_model_excludes_empty(self, queues, make_envelope):
        await queues.enqueue(make_envelope(model="llama3:latest"))
        await queues.dequeue("llama3:latest")
        result = await queues.pending_by_model()
        assert "llama3:latest" not in result


# ---------------------------------------------------------------------------
# ModelQueues - get_all_sorted_by_arrival
# ---------------------------------------------------------------------------


class TestModelQueuesGetAllSorted:
    async def test_empty(self, queues):
        result = await queues.get_all_sorted_by_arrival()
        assert result == []

    async def test_sorted_fifo_across_models(self, queues, make_envelope):
        e1 = make_envelope(model="llama3:latest", arrived_at=100.0)
        e2 = make_envelope(model="mistral:latest", arrived_at=101.0)
        e3 = make_envelope(model="llama3:latest", arrived_at=102.0)
        await queues.enqueue(e1)
        await queues.enqueue(e2)
        await queues.enqueue(e3)
        result = await queues.get_all_sorted_by_arrival()
        assert result == [e1, e2, e3]

    async def test_does_not_remove_from_queues(self, queues, make_envelope):
        env = make_envelope(model="llama3:latest")
        await queues.enqueue(env)
        await queues.get_all_sorted_by_arrival()
        assert await queues.pending_count("llama3:latest") == 1


# ---------------------------------------------------------------------------
# ModelQueues - get_unskippable / increment_skips_for_model
# ---------------------------------------------------------------------------


class TestModelQueuesSkipping:
    async def test_no_unskippable_initially(self, queues, make_envelope):
        await queues.enqueue(make_envelope(model="llama3:latest"))
        result = await queues.get_unskippable(max_skips=3)
        assert result == []

    async def test_unskippable_after_reaching_threshold(self, queues, make_envelope):
        env = make_envelope(model="llama3:latest", arrived_at=100.0)
        await queues.enqueue(env)
        for _ in range(3):
            await queues.increment_skips_for_model("llama3:latest")
        result = await queues.get_unskippable(max_skips=3)
        assert len(result) == 1
        assert result[0] is env

    async def test_increment_skips_only_affects_target_model(
        self, queues, make_envelope
    ):
        e_llama = make_envelope(model="llama3:latest")
        e_mistral = make_envelope(model="mistral:latest")
        await queues.enqueue(e_llama)
        await queues.enqueue(e_mistral)
        await queues.increment_skips_for_model("llama3:latest")
        assert e_llama.skip_count == 1
        assert e_mistral.skip_count == 0

    async def test_increment_skips_nonexistent_model(self, queues):
        # Should not raise
        await queues.increment_skips_for_model("no-such-model")

    async def test_unskippable_sorted_by_arrival(self, queues, make_envelope):
        e1 = make_envelope(model="llama3:latest", arrived_at=200.0)
        e2 = make_envelope(model="mistral:latest", arrived_at=100.0)
        await queues.enqueue(e1)
        await queues.enqueue(e2)
        # Skip both models enough times
        for _ in range(3):
            await queues.increment_skips_for_model("llama3:latest")
            await queues.increment_skips_for_model("mistral:latest")
        result = await queues.get_unskippable(max_skips=3)
        # e2 arrived first (100.0) so should come first
        assert result[0] is e2
        assert result[1] is e1

    async def test_unskippable_does_not_remove_from_queues(self, queues, make_envelope):
        env = make_envelope(model="llama3:latest")
        await queues.enqueue(env)
        for _ in range(5):
            await queues.increment_skips_for_model("llama3:latest")
        await queues.get_unskippable(max_skips=3)
        assert await queues.pending_count("llama3:latest") == 1

    async def test_multiple_envelopes_same_model_skipped(self, queues, make_envelope):
        e1 = make_envelope(model="llama3:latest", arrived_at=100.0)
        e2 = make_envelope(model="llama3:latest", arrived_at=101.0)
        await queues.enqueue(e1)
        await queues.enqueue(e2)
        await queues.increment_skips_for_model("llama3:latest")
        assert e1.skip_count == 1
        assert e2.skip_count == 1


# ---------------------------------------------------------------------------
# ModelQueues - async done_event integration
# ---------------------------------------------------------------------------


class TestModelQueuesAsyncIntegration:
    async def test_done_event_is_awaitable(self, make_envelope):
        env = make_envelope()

        async def complete_later():
            await asyncio.sleep(0.01)
            env.complete({"result": "ok"})

        asyncio.create_task(complete_later())
        await asyncio.wait_for(env.done_event.wait(), timeout=1.0)
        assert env.response == {"result": "ok"}

    async def test_fail_event_is_awaitable(self, make_envelope):
        env = make_envelope()

        async def fail_later():
            await asyncio.sleep(0.01)
            env.fail(ValueError("oops"))

        asyncio.create_task(fail_later())
        await asyncio.wait_for(env.done_event.wait(), timeout=1.0)
        assert isinstance(env.error, ValueError)
