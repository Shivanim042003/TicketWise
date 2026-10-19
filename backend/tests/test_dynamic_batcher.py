import asyncio

import pytest

from backend.app.dynamic_batcher import DynamicBatcher


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_batcher_returns_one_result_per_item(batch_size):
    calls = []

    def batch_fn(items):
        calls.append(list(items))
        return [item.upper() for item in items]

    async def scenario():
        batcher = DynamicBatcher(
            batch_fn,
            max_batch_size=batch_size,
            max_wait_ms=10,
        )

        try:
            results = await asyncio.gather(
                *(batcher.submit(f"query-{i}") for i in range(6))
            )

            assert results == [
                f"QUERY-{i}" for i in range(6)
            ]

        finally:
            await batcher.close()

    asyncio.run(scenario())

    assert sum(len(call) for call in calls) == 6
    assert all(
        len(call) <= batch_size
        for call in calls
    )


def test_batcher_combines_concurrent_queries():
    calls = []

    def batch_fn(items):
        calls.append(list(items))
        return list(items)

    async def scenario():
        batcher = DynamicBatcher(
            batch_fn,
            max_batch_size=8,
            max_wait_ms=50,
        )

        try:
            results = await asyncio.gather(
                batcher.submit("alpha"),
                batcher.submit("beta"),
                batcher.submit("gamma"),
            )

            assert results == ["alpha", "beta", "gamma"]

        finally:
            await batcher.close()

    asyncio.run(scenario())

    assert any(len(call) > 1 for call in calls)


def test_batcher_propagates_inference_errors():
    def batch_fn(items):
        raise ValueError("inference failed")

    async def scenario():
        batcher = DynamicBatcher(
            batch_fn,
            max_batch_size=4,
            max_wait_ms=20,
        )

        try:
            results = await asyncio.gather(
                batcher.submit("alpha"),
                batcher.submit("beta"),
                return_exceptions=True,
            )

            assert all(
                isinstance(result, ValueError)
                for result in results
            )

        finally:
            await batcher.close()

    asyncio.run(scenario())


def test_batcher_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        DynamicBatcher(
            lambda items: items,
            max_batch_size=0,
        )

    with pytest.raises(ValueError):
        DynamicBatcher(
            lambda items: items,
            max_wait_ms=-1,
        )


def test_batcher_rejects_wrong_result_count():
    async def scenario():
        batcher = DynamicBatcher(
            lambda items: [],
            max_batch_size=4,
            max_wait_ms=10,
        )

        try:
            with pytest.raises(
                RuntimeError,
                match="different number of results",
            ):
                await batcher.submit("alpha")

        finally:
            await batcher.close()

    asyncio.run(scenario())


def test_batcher_closes_after_processing_pending_requests():
    calls = []

    def batch_fn(items):
        calls.append(list(items))
        return list(items)

    async def scenario():
        batcher = DynamicBatcher(
            batch_fn,
            max_batch_size=8,
            max_wait_ms=100,
        )

        tasks = [
            asyncio.create_task(batcher.submit("alpha")),
            asyncio.create_task(batcher.submit("beta")),
        ]

        results = await asyncio.gather(*tasks)
        await batcher.close()

        assert results == ["alpha", "beta"]
        assert calls

        with pytest.raises(RuntimeError, match="closed"):
            await batcher.submit("gamma")

    asyncio.run(scenario())


def test_batcher_rejects_invalid_queue_size():
    with pytest.raises(ValueError, match="max_queue_size"):
        DynamicBatcher(
            lambda items: items,
            max_queue_size=0,
        )


def test_batcher_close_is_idempotent():
    async def scenario():
        batcher = DynamicBatcher(
            lambda items: list(items),
            max_wait_ms=1,
        )

        await batcher.start()
        await batcher.close()
        await batcher.close()

        with pytest.raises(RuntimeError, match="closed"):
            await batcher.submit("after-close")

    asyncio.run(scenario())


def test_batcher_processes_all_concurrent_requests_before_close():
    calls = []

    def batch_fn(items):
        calls.append(list(items))
        return [f"done:{item}" for item in items]

    async def scenario():
        batcher = DynamicBatcher(
            batch_fn,
            max_batch_size=3,
            max_wait_ms=10,
            max_queue_size=20,
        )

        tasks = [
            asyncio.create_task(batcher.submit(f"q{i}"))
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)
        await batcher.close()

        assert results == [f"done:q{i}" for i in range(10)]
        assert sum(len(batch) for batch in calls) == 10
        assert all(len(batch) <= 3 for batch in calls)

    asyncio.run(scenario())
