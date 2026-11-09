# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.

import numpy as np
import pytest

import py_parsec


def test_block_cyclic_shape(ctx):
    A = py_parsec.BlockCyclic(ctx, 50, 33, 16, 8, name="A")
    assert (A.m, A.n, A.mb, A.nb) == (50, 33, 16, 8)
    assert (A.mt, A.nt) == (4, 5)
    assert A.dtype == np.float64
    assert A.rank == ctx.rank and A.nodes == ctx.nb_nodes
    owners = {A.rank_of(m, n) for m in range(A.mt) for n in range(A.nt)}
    assert owners <= set(range(ctx.nb_nodes))
    local = list(A.local_tiles())
    assert len(local) == A.nb_local_tiles
    for m, n, tile in local:
        assert A.is_local(m, n)
        assert tile.shape == (16, 8) and tile.flags.f_contiguous and tile.flags.writeable
        assert not tile.any()  # storage starts zeroed
    A.release()
    with pytest.raises(RuntimeError, match="released"):
        A.m


def test_tile_views_share_storage(ctx):
    A = py_parsec.BlockCyclic(ctx, 32, 32, 8, 8, dtype=np.float32)
    for m, n, tile in A.local_tiles():
        tile[:] = m * 10 + n
        tile[1, 2] = -1
    for m, n, _ in A.local_tiles():
        again = A.tile(m, n)
        assert again.dtype == np.float32
        assert again[0, 0] == m * 10 + n and again[1, 2] == -1
        # column-major storage: element (1, 2) is the 17th of the tile
        assert again.ravel(order="F")[2 * 8 + 1] == -1


def test_tile_errors(ctx):
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    with pytest.raises(IndexError):
        A.tile(2, 0)
    for m in range(A.mt):
        for n in range(A.nt):
            if not A.is_local(m, n):
                with pytest.raises(ValueError, match="owned by rank"):
                    A.tile(m, n)


def test_block_cyclic_arguments(ctx):
    with pytest.raises(TypeError, match="dtype"):
        py_parsec.BlockCyclic(ctx, 8, 8, 4, 4, dtype=np.int64)
    with pytest.raises(ValueError):
        py_parsec.BlockCyclic(ctx, 8, 8, 0, 4)
    with pytest.raises(ValueError, match="does not divide"):
        py_parsec.BlockCyclic(ctx, 8, 8, 4, 4, P=ctx.nb_nodes + 1)
    one_row = py_parsec.BlockCyclic(ctx, 8, 8 * ctx.nb_nodes, 8, 8, P=1)
    assert sorted(one_row.rank_of(0, n) for n in range(one_row.nt)) == list(range(ctx.nb_nodes))
