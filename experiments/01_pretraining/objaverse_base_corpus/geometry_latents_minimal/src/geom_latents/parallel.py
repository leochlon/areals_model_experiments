"""Worker pools for per-file jobs.

Every worker would otherwise start a BLAS thread pool the size of the machine;
with several workers those pools busy-wait against each other and a parallel
audit or conversion ends up slower than a sequential one. Workers are spawned
fresh with the thread variables set to one, so the parent's own libraries are
untouched.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from multiprocessing import get_context

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


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


def map_jobs(function, jobs: list, workers: int, *, ordered: bool = True) -> list:
    """Apply a module-level function to every job, in worker processes when workers > 1.

    Results keep the job order when ordered; otherwise they arrive as finished.
    """
    workers = max(1, min(int(workers), len(jobs)))
    if workers == 1 or not jobs:
        return [function(job) for job in jobs]
    chunksize = max(1, min(64, len(jobs) // (4 * workers)))
    with single_thread_environment(), get_context("spawn").Pool(workers) as pool:
        mapper = pool.imap if ordered else pool.imap_unordered
        return list(mapper(function, jobs, chunksize=chunksize))

