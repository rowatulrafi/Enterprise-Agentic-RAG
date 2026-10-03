from contextlib import contextmanager
from threading import Lock
from time import perf_counter


_trace = {}
_lock = Lock()


def reset_latency_trace():
    with _lock:
        _trace.clear()


def get_latency_trace():
    with _lock:
        return dict(_trace)


@contextmanager
def measure_stage(stage_name: str):

    start = perf_counter()

    try:
        yield

    finally:
        elapsed_ms = (
            perf_counter() - start
        ) * 1000

        with _lock:
            _trace[stage_name] = (
                _trace.get(stage_name, 0.0)
                + elapsed_ms
            )