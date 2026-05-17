#!/usr/bin/env python3
"""
Official PaRSEC stencil-1D workflow - Direct core API, NO DTD!

Exactly mirrors testing_stencil_1D.c:
1. parsec_init
2. parsec_matrix_block_cyclic_init (with ghost columns NB+2*R)
3. parsec_apply (initialize tiles)
4. parsec_stencil_1D (run kernel with SYNC_TIME timing)
5. parsec_fini
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from py_parsec.stencil_core import (
    ParsecCoreContext,
    ParsecMatrix,
    PARSEC_MATRIX_FULL,
    PARSEC_MATRIX_DOUBLE,
    PARSEC_MATRIX_TILE
)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def run_stencil_official(M: int, N: int, MB: int, NB: int, iter: int, R: int,
                         P: int = 1, KP: int = 1, KQ: int = 1, cores: int = -1):
    """
    Official stencil workflow using core PaRSEC API.
    
    Args match testing_stencil_1D.c exactly.
    Returns dict with matrix info and performance metrics.
    """
    # Initialize MPI if available
    if MPI is not None:
        if not MPI.Is_initialized():
            MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nodes = comm.Get_size()
    else:
        rank = 0
        nodes = 1
    
    if P <= 0 or nodes % P != 0:
        raise ValueError(f"Invalid process grid: P={P}, nodes={nodes}")
    Q = nodes // P
    
    # Number of column tiles (for ghost columns calculation)
    NNB = (N + NB - 1) // NB
    
    # Step 1: Initialize PaRSEC (like official: parsec_init)
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    parsec_init_start = __import__('time').time()
    parsec = ParsecCoreContext(nb_cores=cores)
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    parsec_init_time = __import__('time').time() - parsec_init_start
    if rank == 0:
        print(f"ParsecCore init_time={parsec_init_time:.9f}s", file=__import__('sys').stderr)
    
    # Step 2: Initialize matrix with ghost columns (like official: parsec_matrix_block_cyclic_init)
    # Official: parsec_matrix_block_cyclic_init(&dcA, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
    #           rank, MB, NB+2*R, M, N+2*R*NNB, 0, 0, M, N+2*R*NNB, P, nodes/P, KP, KQ, 0, 0);
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    init_data_start = __import__('time').time()
    dcA = ParsecMatrix()
    dcA.init(
        key="dcA",
        myrank=rank,
        mb=MB,
        nb=NB + 2*R,  # Ghost columns
        lm=M,
        ln=N + 2*R*NNB,  # Total columns including ghosts
        P=P, Q=Q,
        kp=KP, kq=KQ,
        mtype=PARSEC_MATRIX_DOUBLE,
        storage=PARSEC_MATRIX_TILE
    )
    
    # Step 3: Initialize tiles using parsec_apply (like official)
    # Official: parsec_apply(parsec, PARSEC_MATRIX_FULL, (parsec_tiled_matrix_t*)&dcA, stencil_1D_init_ops, &R);
    parsec.apply(dcA.as_capsule(), PARSEC_MATRIX_FULL, R)
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    init_data_time = __import__('time').time() - init_data_start
    if rank == 0:
        print(f"Data init_time={init_data_time:.9f}s", file=__import__('sys').stderr)
    
    # Step 4: Run stencil kernel with generic SYNC_TIME timing
    # Official: parsec_stencil_1D(parsec, (parsec_tiled_matrix_t*)&dcA, iter, R);
    # FLOPS = iter * (2*(2*R+1)) * N*MB (similar to testing_stencil_1D.c)
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    exec_start = __import__('time').time()
    parsec.stencil_1D(dcA.as_capsule(), iter, R)
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()
    exec_time = __import__('time').time() - exec_start
    
    # Calculate performance metrics
    # FLOPS_STENCIL_1D(n) = iter * (2*(2*R+1)) * n
    # where n = N * MB (columns * rows per tile)
    flops = iter * (2 * (2*R + 1)) * N * MB
    gflops = (flops / 1e9) / exec_time if exec_time > 0 else 0.0
    
    # Step 5: Finalize PaRSEC (like official: parsec_fini)
    parsec.fini()
    
    return {
        "rank": rank,
        "nodes": nodes,
        "mt": dcA.mt,
        "nt": dcA.nt,
        "mb": dcA.mb,
        "nb": dcA.nb,
        "m": dcA.m,
        "n": dcA.n,
        "parsec_init_time": parsec_init_time,
        "init_data_time": init_data_time,
        "exec_time": exec_time,
        "gflops": gflops,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Official PaRSEC stencil-1D (core API, no DTD)")
    p.add_argument("--M", type=int, default=8, help="global rows")
    p.add_argument("--N", type=int, default=12, help="global cols (without ghosts)")
    p.add_argument("--MB", type=int, default=4, help="tile rows")
    p.add_argument("--NB", type=int, default=4, help="tile cols")
    p.add_argument("--iter", type=int, default=3, help="iterations")
    p.add_argument("--R", type=int, default=1, help="stencil radius")
    p.add_argument("--P", type=int, default=1, help="process grid rows")
    p.add_argument("--KP", type=int, default=1, help="K-cyclicity rows")
    p.add_argument("--KQ", type=int, default=1, help="K-cyclicity cols")
    p.add_argument("--cores", type=int, default=-1, help="PaRSEC cores (-1=auto)")
    args = p.parse_args()
    
    info = run_stencil_official(args.M, args.N, args.MB, args.NB, args.iter, args.R,
                                P=args.P, KP=args.KP, KQ=args.KQ, cores=args.cores)
    
    if info["rank"] == 0:
        # Single line output for easy plotting with parameter names
        Q = info['nodes'] // args.P if args.P > 0 else 1
        print(f"M={args.M}\tN={args.N}\tMB={args.MB}\tNB={args.NB}\titer={args.iter}\tR={args.R}\tP={args.P}\tQ={Q}\tparsec_init_time={info['parsec_init_time']:.9f}s\tinit_data_time={info['init_data_time']:.9f}s\texec_time={info['exec_time']:.9f}s\tgflops={info['gflops']:.6f}")


if __name__ == "__main__":
    main()
