#!/usr/bin/env python3
"""
PaRSEC merge_sort workflow - Direct core API, NO DTD!

Mirrors tests/apps/merge_sort/main.c (in the PaRSEC repo root):
1. MPI_Init (if available)
2. parsec_init
3. create_and_distribute_data
4. merge_sort_new + context_add_taskpool/start/wait
5. parsec_fini
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from py_parsec.merge_sort_core import (
    ParsecMergeSortContext,
    ParsecMergeSortMatrix,
    ParsecMergeSortTaskpool,
)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def run_merge_sort_official(nt: int = 1234, nb: int = 5, cores: int = -1, typesize: int = 4):
    """
    Official merge_sort workflow using core PaRSEC API.
    Args match tests/apps/merge_sort/main.c (in the PaRSEC repo root).
    Returns dict with timing and basic info.
    """
    if MPI is not None:
        if not MPI.Is_initialized():
            MPI.Init_thread(required=MPI.THREAD_SERIALIZED)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world = comm.Get_size()
    else:
        rank = 0
        world = 1

    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    parsec_init_start = time.time()
    parsec = ParsecMergeSortContext(nb_cores=cores)
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    parsec_init_time = time.time() - parsec_init_start

    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    init_data_start = time.time()
    dcA = ParsecMergeSortMatrix(rank, world, nb, nt, typesize=typesize, key="A")
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    init_data_time = time.time() - init_data_start

    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    exec_start = time.time()
    msort = ParsecMergeSortTaskpool(dcA, nb, nt)
    parsec.add_taskpool(msort)
    parsec.start()
    parsec.wait()
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    exec_time = time.time() - exec_start

    msort.free()
    parsec.fini()

    return {
        "rank": rank,
        "world": world,
        "nt": nt,
        "nb": nb,
        "parsec_init_time": parsec_init_time,
        "init_data_time": init_data_time,
        "exec_time": exec_time,
    }


def main():
    import argparse

    p = argparse.ArgumentParser(description="PaRSEC merge_sort (core API, no DTD)")
    p.add_argument("--nt", type=int, default=1234, help="number of tiles")
    p.add_argument("--nb", type=int, default=5, help="tile size")
    p.add_argument("--cores", type=int, default=-1, help="PaRSEC cores (-1=auto)")
    p.add_argument("--typesize", type=int, default=4, help="bytes per element")
    args = p.parse_args()

    info = run_merge_sort_official(nt=args.nt, nb=args.nb, cores=args.cores, typesize=args.typesize)
    if info["rank"] == 0:
        print(
            f"nt={info['nt']}\tnb={info['nb']}\t"
            f"parsec_init_time={info['parsec_init_time']:.9f}s\t"
            f"init_data_time={info['init_data_time']:.9f}s\t"
            f"exec_time={info['exec_time']:.9f}s"
        )


if __name__ == "__main__":
    main()

