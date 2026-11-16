#!/usr/bin/env python3
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""DTD and PTG taskpools working on the same data in the same context.

A DTD taskpool fills a block-cyclic matrix, the PTG redistribute taskpool of
the PaRSEC library copies a submatrix of it into a matrix with another tile
size and process grid, and a second DTD taskpool updates the copy. Run with
any number of processes, e.g. ``mpiexec -n 4 python redistribute.py``.
"""

import numpy as np

import py_parsec
from py_parsec import dtd
from py_parsec.matrix import redistribute_new

M, N = 300, 200          # source matrix
SUB = (120, 90)          # submatrix copied ...
FROM, TO = (30, 20), (7, 11)   # ... from this offset in the source to this one in the target


def value(i, j):
    return i * 1000.0 + j


def main():
    with py_parsec.Context() as ctx:
        src = py_parsec.BlockCyclic(ctx, M, N, 32, 32, name="src")
        dst = py_parsec.BlockCyclic(ctx, 200, 150, 25, 40, P=1, name="dst")
        ctx.start()

        # 1. DTD: initialize the source.
        fill = dtd.Taskpool()
        ctx.add_taskpool(fill)
        init = fill.task_class("init", dtd.Data(dtd.OUTPUT | dtd.AFFINITY, src), dtd.Value("i"), dtd.Value("i"))

        def init_body(tile, m, n):
            rows = np.arange(m * tile.shape[0], (m + 1) * tile.shape[0])[:, None]
            cols = np.arange(n * tile.shape[1], (n + 1) * tile.shape[1])[None, :]
            tile[:] = value(rows, cols)

        init.add_chore(py_parsec.DEV_CPU, init_body)
        for m in range(src.mt):
            for n in range(src.nt):
                fill.insert_task(init, dtd.tile_of(src, m, n), m, n)
        fill.data_flush_all(src)

        # 2. PTG: copy the submatrix; enqueued like any other taskpool.
        copy = redistribute_new(src, dst, *SUB, *FROM, *TO)
        ctx.add_taskpool(copy)
        ctx.wait()

        # 3. DTD again, on the target.
        ctx.start()
        update = dtd.Taskpool()
        ctx.add_taskpool(update)
        negate = update.task_class("negate", dtd.Data(dtd.INOUT | dtd.AFFINITY, dst))
        negate.add_chore(py_parsec.DEV_CPU, lambda tile: np.negative(tile, out=tile))
        for m in range(dst.mt):
            for n in range(dst.nt):
                update.insert_task(negate, dtd.tile_of(dst, m, n))
        update.data_flush_all(dst)
        ctx.wait()

        # Check the tiles this rank owns.
        expected = np.zeros((dst.m, dst.n))
        rows = np.arange(FROM[0], FROM[0] + SUB[0])[:, None]
        cols = np.arange(FROM[1], FROM[1] + SUB[1])[None, :]
        expected[TO[0]:TO[0] + SUB[0], TO[1]:TO[1] + SUB[1]] = -value(rows, cols)
        ok = True
        for m, n, tile in dst.local_tiles():
            block = expected[m * dst.mb:(m + 1) * dst.mb, n * dst.nb:(n + 1) * dst.nb]
            ok &= np.array_equal(tile[:block.shape[0], :block.shape[1]], block)
        print(f"rank {ctx.rank}/{ctx.nb_nodes}: {dst.nb_local_tiles} tiles {'PASSED' if ok else 'FAILED'}")
        for tp in (fill, copy, update):
            tp.free()
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
