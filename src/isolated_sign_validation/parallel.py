import multiprocessing
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor


def parallel_map[T, R](fn: Callable[[T], R], items: Sequence[T], batch_size: int = 1024, chunksize: int = 1) -> Iterator[R]:
    """Apply `fn` to `items` in worker processes, yielding the results in the order of `items`.

    Items are submitted `batch_size` at a time, so memory use stays bounded when the consumer
    is slower than the workers. `fn` must be a module-level function.
    """
    ctx = multiprocessing.get_context("spawn")  # polars and mediapipe are not fork-safe
    with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx) as pool:
        for i in range(0, len(items), batch_size):
            yield from pool.map(fn, items[i : i + batch_size], chunksize=chunksize)
