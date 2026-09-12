import multiprocessing
import os
import signal
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor


def _init_worker(initializer: Callable[[], None] | None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is handled by the main process, which stops the workers
    if initializer is not None:
        initializer()


def parallel_map[T, R](
    fn: Callable[[T], R],
    items: Sequence[T],
    batch_size: int = 1024,
    chunksize: int = 1,
    max_workers: int | None = None,
    initializer: Callable[[], None] | None = None,
) -> Iterator[R]:
    """Apply `fn` to `items` in worker processes, yielding the results in the order of `items`.

    Items are submitted `batch_size` at a time, so memory use stays bounded when the consumer
    is slower than the workers. `fn` and `initializer` (run once in each worker) must be
    module-level functions. Uses one worker per CPU unless `max_workers` is given. On Ctrl-C the
    workers are terminated immediately.
    """
    ctx = multiprocessing.get_context("spawn")  # polars and mediapipe are not fork-safe
    pool = ProcessPoolExecutor(
        max_workers=max_workers or os.cpu_count(), mp_context=ctx, initializer=_init_worker, initargs=(initializer,)
    )

    def stop(signum, frame):
        for process in pool._processes.values():  # the executor has no public way to kill running workers
            process.terminate()
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, stop)
    try:
        for i in range(0, len(items), batch_size):
            yield from pool.map(fn, items[i : i + batch_size], chunksize=chunksize)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        pool.shutdown(cancel_futures=True)
