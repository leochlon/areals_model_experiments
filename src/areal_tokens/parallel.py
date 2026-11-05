"""Worker pools for per-file jobs.

Every worker would otherwise start a BLAS thread pool the size of the machine;
with several workers those pools busy-wait against each other and a parallel
audit or conversion ends up slower than a sequential one. Workers are spawned
fresh with the thread variables set to one, so the parent's own libraries are
untouched.

A worker can also die outright: the kernel kills it for memory, or a native
mesh library crashes on a corrupt file. A plain multiprocessing pool then waits
forever for the lost job. Here the pool breaks instead, the jobs that were in
flight are rerun one at a time in a fresh process, and a job that still takes
its process down is reported as failed while the rest of the map completes.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from multiprocessing import get_context

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

WORKER_DIED = "worker process died on this record (killed for memory, or a crash in a native library)"


@contextmanager
def single_thread_environment():
    previous = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update({name: "1" for name in THREAD_VARIABLES})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_alone(function, job, context, on_crash):
    with ProcessPoolExecutor(1, mp_context=context) as pool:
        try:
            return pool.submit(function, job).result()
        except BrokenProcessPool:
            if on_crash is None:
                raise RuntimeError(WORKER_DIED) from None
            return on_crash(job)


def map_jobs(function, jobs: list, workers: int, *, ordered: bool = True, on_crash=None) -> list:
    """Apply a module-level function to every job, in worker processes when workers > 1.

    Results always come back in job order (``ordered`` is kept for callers of
    the earlier interface). ``on_crash(job)`` builds the result for a job whose
    process died even when run alone; without it such a job raises.
    """
    del ordered
    workers = max(1, min(int(workers), len(jobs)))
    if workers == 1 or not jobs:
        return [function(job) for job in jobs]
    context = get_context("spawn")
    results = [None] * len(jobs)
    finished = [False] * len(jobs)
    pending = list(range(len(jobs)))
    with single_thread_environment():
        while pending:
            broken = False
            with ProcessPoolExecutor(workers, mp_context=context) as pool:
                futures = {pool.submit(function, jobs[index]): index for index in pending}
                try:
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()
                        finished[futures[future]] = True
                except BrokenProcessPool:
                    broken = True
                    for future, index in futures.items():
                        if future.done() and not future.cancelled() and future.exception() is None:
                            results[index], finished[index] = future.result(), True
            pending = [index for index in pending if not finished[index]]
            if broken:
                # Jobs are dispatched in submission order, at most 2 * workers + 1
                # at a time, so the one that took a worker down is among the
                # earliest unfinished jobs. Each of those runs alone.
                suspects, pending = pending[: 2 * workers + 1], pending[2 * workers + 1 :]
                for index in suspects:
                    results[index] = _run_alone(function, jobs[index], context, on_crash)
                    finished[index] = True
    return results
