# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Shared fixtures.

The tests run on one process or under an MPI launcher
(``mpiexec -n 4 python -m pytest``); every rank runs every test, in the same
order, and checks the tiles it owns. One context serves the whole session.
"""

import os

# Task bodies call NumPy from PaRSEC worker threads: keep its BLAS to one
# thread per call so the ranks do not oversubscribe the cores.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pytest

import py_parsec
from py_parsec import dtd


@pytest.fixture(scope="session")
def ctx():
    cores = int(os.environ.get("PY_PARSEC_TEST_CORES", "2"))
    with py_parsec.Context(nb_cores=cores) as context:
        yield context


@pytest.fixture
def tp(ctx):
    """A DTD taskpool enqueued in the started session context."""
    ctx.start()
    taskpool = dtd.Taskpool()
    ctx.add_taskpool(taskpool)
    yield taskpool
    taskpool.free()


def scatter(matrix, full):
    """Copy the global array ``full`` into the local tiles of ``matrix``."""
    for m, n, tile in matrix.local_tiles():
        block = full[m * matrix.mb:(m + 1) * matrix.mb, n * matrix.nb:(n + 1) * matrix.nb]
        tile[:block.shape[0], :block.shape[1]] = block


def assert_matches(matrix, full, **tol):
    """Check the local tiles of ``matrix`` against the global array ``full``."""
    for m, n, tile in matrix.local_tiles():
        block = full[m * matrix.mb:(m + 1) * matrix.mb, n * matrix.nb:(n + 1) * matrix.nb]
        np.testing.assert_allclose(tile[:block.shape[0], :block.shape[1]], block,
                                   err_msg=f"tile ({m}, {n}) of {matrix.name}", **tol)
