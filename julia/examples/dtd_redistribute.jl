#!/usr/bin/env julia

using MPI

include(joinpath(@__DIR__, "..", "src", "dtd_simple.jl"))
using .DTDSimple

function choose_pq(size::Int)
    p = Int(floor(sqrt(size)))
    while p > 1 && size % p != 0
        p -= 1
    end
    return p, div(size, p)
end

function main()
    prepare_mpi!()
    mpi_initialized_here = false
    if !MPI.Initialized()
        MPI.Init()
        mpi_initialized_here = true
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nodes = MPI.Comm_size(comm)
    P, Q = choose_pq(nodes)

    M = 4
    N = 4
    MB = 4
    NB = 4

    ctx = ParsecDTDContext()
    tp = ParsecDTDTaskpool()
    add_taskpool(ctx, tp)  # initializes global DTD tile mempool

    src = ParsecMatrixBlockCyclic()
    dst = ParsecMatrixBlockCyclic()
    init(src, "dcY", rank, MB, NB, M, N, P, Q)
    init(dst, "dcT", rank, MB, NB, M, N, P, Q)
    dtd_data_collection_init(src)
    dtd_data_collection_init(dst)

    src_buf = local_buffer(src)
    dst_buf = local_buffer(dst)
    src_buf .= collect(0.0:(length(src_buf)-1))
    dst_buf .= 0.0

    MPI.Barrier(comm)
    parsec_redistribute_dtd(ctx, src, dst, M, N, 0, 0, 0, 0)
    MPI.Barrier(comm)

    local_ok = all(isapprox.(dst_buf, src_buf))
    ok = MPI.Allreduce(local_ok, MPI.LAND, comm)
    if rank == 0
        println("Redistribute DTD complete.")
        println("Correctness check: ", ok ? "PASSED" : "FAILED")
    end

    destroy(src)
    destroy(dst)
    free(tp)
    fini(ctx)

    if mpi_initialized_here && MPI.Initialized()
        MPI.Finalize()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

