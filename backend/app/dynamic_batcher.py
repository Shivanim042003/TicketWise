
import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")
_STOP = object()


@dataclass(frozen=True)
class BatchRequest:
    """A semantic-search request with its own result limit."""

    query: str
    top_k: int = 5


class DynamicBatcher:
    """Collect concurrent requests and process them in bounded batches."""

    def __init__(
        self,
        batch_fn: Callable[[list[Any]], Sequence[T]],
        max_batch_size: int = 8,
        max_wait_ms: float = 5.0,
        max_queue_size: int = 256,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")

        if max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative")

        if max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")

        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000

        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: asyncio.Task | None = None
        self._closed = False
        self._state_lock = asyncio.Lock()

        # Runtime metrics for evaluating dynamic batching.
        self.total_batches = 0
        self.total_requests_processed = 0
        self.max_observed_batch_size = 0

    @property
    def metrics(self) -> dict:
        """Return runtime batching metrics."""
        return {
            "total_batches": self.total_batches,
            "total_requests_processed": self.total_requests_processed,
            "average_batch_size": (
                round(
                    self.total_requests_processed / self.total_batches,
                    2,
                )
                if self.total_batches
                else 0.0
            ),
            "max_observed_batch_size": self.max_observed_batch_size,
        }

    async def start(self) -> None:
        """Start the background worker once."""
        async with self._state_lock:
            if self._closed:
                raise RuntimeError("Batcher is closed")

            if self._worker_task is None:
                self._worker_task = asyncio.create_task(self._worker())

    async def submit(self, item: Any) -> T:
        """Submit an item and await its corresponding result."""
        async with self._state_lock:
            if self._closed:
                raise RuntimeError("Batcher is closed")

            if self._worker_task is None:
                self._worker_task = asyncio.create_task(self._worker())

            future = asyncio.get_running_loop().create_future()
            await self._queue.put((item, future))

        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _worker(self) -> None:
        """Collect requests, process batches, and deliver results."""
        while True:
            first = await self._queue.get()

            if first is _STOP:
                self._queue.task_done()
                return

            batch = [first]
            deadline = (
                asyncio.get_running_loop().time()
                + self.max_wait_seconds
            )

            while len(batch) < self.max_batch_size:
                # Take anything already queued without waiting, so that
                # max_wait_ms=0 still batches requests that arrived while
                # the previous batch was being processed.
                try:
                    entry = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - asyncio.get_running_loop().time()

                    if remaining <= 0:
                        break

                    try:
                        entry = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=remaining,
                        )
                    except asyncio.TimeoutError:
                        break

                if entry is _STOP:
                    # Preserve the stop signal for the next worker iteration.
                    self._queue.task_done()
                    await self._queue.put(_STOP)
                    break

                batch.append(entry)

            try:
                items = [item for item, _ in batch]

                # Record each batch sent to inference.
                batch_size = len(items)
                self.total_batches += 1
                self.total_requests_processed += batch_size
                self.max_observed_batch_size = max(
                    self.max_observed_batch_size,
                    batch_size,
                )

                results = await asyncio.to_thread(
                    self.batch_fn,
                    items,
                )

                if len(results) != len(batch):
                    raise RuntimeError(
                        "Batch function returned a different number "
                        "of results than inputs"
                    )

                for (_, future), result in zip(batch, results):
                    if not future.done():
                        future.set_result(result)

            except Exception as exc:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)

            finally:
                for _ in batch:
                    self._queue.task_done()

    async def close(self) -> None:
        """Reject new requests, drain queued work, and stop the worker."""
        async with self._state_lock:
            if self._closed:
                return

            self._closed = True
            worker_task = self._worker_task

        if worker_task is None:
            return

        await self._queue.join()
        await self._queue.put(_STOP)
        await worker_task

        async with self._state_lock:
            self._worker_task = None
