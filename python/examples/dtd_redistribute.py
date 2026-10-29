#!/usr/bin/env python3
"""
Minimal example for the DTD redistribute Python API.

This mirrors tests/collections/redistribute/testing_redistribute.c (in the PaRSEC repo root)
but focuses only on invoking parsec_redistribute_dtd from Python.
"""

import numpy as np
from mpi4py import MPI

import py_parsec.dtd as dtd


def choose_pq(size: int):
    p = int(size ** 0.5)
    while p > 1 and size % p != 0:
        p -= 1
    q = size // p
    return p, q


def main():
    if not MPI.Is_initialized():
        MPI.Init()
        mpi_initialized_here = True
    else:
        mpi_initialized_here = False

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    P, Q = choose_pq(size)

    # Source/target parameters (similar defaults to testing_redistribute.c)
    M = N = 4
    MB = NB = 4
    size_row = M
    size_col = N
    disi_Y = disj_Y = 0
    disi_T = disj_T = 0

    ctx = dtd.ParsecDTDContext()
    tp = dtd.ParsecDTDTaskpool()
    ctx.add_taskpool(tp)

    src = dtd.ParsecMatrixBlockCyclic()
    dst = dtd.ParsecMatrixBlockCyclic()
    src.init("dcY", rank, MB, NB, M, N, P, Q)
    dst.init("dcT", rank, MB, NB, M, N, P, Q)

    # Initialize local buffers (single-rank friendly)
    src_buf = src.local_buffer()
    dst_buf = dst.local_buffer()
    src_buf[:] = np.arange(src_buf.size, dtype=np.float64)
    dst_buf[:] = 0.0

    comm.Barrier()

    # DTD redistribute
    dtd.parsec_redistribute_dtd(
        ctx, src, dst,
        size_row, size_col,
        disi_Y, disj_Y,
        disi_T, disj_T,
    )

    comm.Barrier()

    # Correctness check (simple case: same sizes/displacements)
    local_ok = np.allclose(dst_buf, src_buf)
    ok = comm.allreduce(local_ok, op=MPI.LAND)

    if rank == 0:
        print("Redistribute DTD complete.")
        print("dst buffer (first 16):", dst_buf[:16])
        if ok:
            print("Correctness check: PASSED")
        else:
            print("Correctness check: FAILED")

    # Cleanup
    try:
        src.destroy()
        dst.destroy()
    except Exception:
        pass
    try:
        tp.free()
    except Exception:
        pass
    ctx.fini()

    if mpi_initialized_here and MPI.Is_initialized():
        MPI.Finalize()


if __name__ == "__main__":
    main()
