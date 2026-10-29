#!/usr/bin/env julia
"""
Official PaRSEC stencil-1D workflow - Direct core API, NO DTD!

Exactly mirrors testing_stencil_1D.c:
1. parsec_init
2. parsec_matrix_block_cyclic_init (with ghost columns NB+2*R)
3. parsec_apply (initialize tiles)
4. parsec_stencil_1D (run kernel with SYNC_TIME timing)
5. parsec_fini
"""

using MPI
using Printf

# Add src directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Load PaRSEC4Julia modules
include(joinpath(@__DIR__, "..", "src", "stencil_core.jl"))

using .StencilCore

function run_stencil_official(M::Int, N::Int, MB::Int, NB::Int, iter::Int, R::Int;
                              P::Int = 1, KP::Int = 1, KQ::Int = 1, cores::Int = -1)
    """
    Official stencil workflow using core PaRSEC API.
    Matches testing_stencil_1D.c exactly.
    
    Returns dict with matrix info and performance metrics.
    """
    # Initialize MPI if available
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nodes = MPI.Comm_size(comm)
    else
        rank = 0
        nodes = 1
    end
    
    if P <= 0 || nodes % P != 0
        error("Invalid process grid: P=$P, nodes=$nodes")
    end
    Q = div(nodes, P)
    
    # Number of column tiles (for ghost columns calculation)
    NNB = div(N + NB - 1, NB)
    
    # Step 1: Initialize PaRSEC (like official: parsec_init)
    if MPI.Initialized()
        MPI.Barrier(MPI.COMM_WORLD)
    end
    parsec_init_start = time()
    parsec = parsec_init(cores)
    if MPI.Initialized()
        MPI.Barrier(MPI.COMM_WORLD)
    end
    parsec_init_time = time() - parsec_init_start
    if rank == 0
        println(stderr, "ParsecCore init_time=$(parsec_init_time)")
    end
    
    # Step 2: Initialize matrix with ghost columns (like official: parsec_matrix_block_cyclic_init)
    # Official: parsec_matrix_block_cyclic_init(&dcA, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
    #           rank, MB, NB+2*R, M, N+2*R*NNB, 0, 0, M, N+2*R*NNB, P, nodes/P, KP, KQ, 0, 0);
    if MPI.Initialized()
        MPI.Barrier(MPI.COMM_WORLD)
    end
    init_data_start = time()
    dcA = ParsecMatrix()
    parsec_matrix_init!(dcA, rank, MB, NB + 2*R, M, N + 2*R*NNB, P, Q;
                       kp=KP, kq=KQ,
                       mtype=PARSEC_MATRIX_DOUBLE,
                       storage=PARSEC_MATRIX_TILE)
    
    # Step 3: Initialize tiles using parsec_apply (like official)
    # Official: parsec_apply(parsec, PARSEC_MATRIX_FULL, (parsec_tiled_matrix_t*)&dcA, stencil_1D_init_ops, &R);
    parsec_apply(parsec, PARSEC_MATRIX_FULL, dcA, R)
    if MPI.Initialized()
        MPI.Barrier(MPI.COMM_WORLD)
    end
    init_data_time = time() - init_data_start
    if rank == 0
        println(stderr, "Data init_time=$(init_data_time)")
    end
    
    # Step 4: Run stencil kernel with generic SYNC_TIME timing
    # Official: parsec_stencil_1D(parsec, (parsec_tiled_matrix_t*)&dcA, iter, R);
    # FLOPS = iter * (2*(2*R+1)) * N*MB (similar to testing_stencil_1D.c)
    if MPI.Initialized()
        MPI.Barrier(MPI.COMM_WORLD)
    end
    exec_start = time()
    parsec_stencil_1D(parsec, dcA, iter, R)
    if MPI.Initialized()
        MPI.Barrier(MPI.COMM_WORLD)
    end
    exec_time = time() - exec_start
    
    # Calculate performance metrics
    # FLOPS_STENCIL_1D(n) = iter * (2*(2*R+1)) * n
    # where n = N * MB (columns * rows per tile)
    flops = iter * (2 * (2*R + 1)) * N * MB
    gflops = exec_time > 0 ? (flops / 1e9) / exec_time : 0.0
    
    # Step 5: Finalize PaRSEC (like official: parsec_fini)
    parsec_fini(parsec)
    
    return Dict(
        "rank" => rank,
        "nodes" => nodes,
        "mt" => dcA.mt,
        "nt" => dcA.nt,
        "mb" => dcA.mb,
        "nb" => dcA.nb,
        "m" => dcA.m,
        "n" => dcA.n,
        "parsec_init_time" => parsec_init_time,
        "init_data_time" => init_data_time,
        "exec_time" => exec_time,
        "gflops" => gflops,
    )
end

function main()
    """Main function - equivalent to main() in testing_stencil_1D.c"""
    
    # Parse command line arguments
    ap = parse_args(ARGS)
    
    M = ap["M"]
    N = ap["N"]
    MB = ap["MB"]
    NB = ap["NB"]
    iter = ap["iter"]
    R = ap["R"]
    P = ap["P"]
    KP = ap["KP"]
    KQ = ap["KQ"]
    cores = ap["cores"]
    
    println("PaRSEC4Julia 1D Stencil - Real PaRSEC Functions (Official Workflow)")
    println("=" ^ 70)
    println("Parameters: M=$M, N=$N, MB=$MB, NB=$NB, P=$P")
    println("Iterations=$iter, Radius=$R")
    
    # Initialize MPI
    mpi_initialized_by_us = false
    if !MPI.Initialized()
        provided = MPI.Init_thread(MPI.THREAD_SERIALIZED)
        mpi_initialized_by_us = true
        if provided < MPI.THREAD_SERIALIZED
            println(stderr, "WARNING: MPI thread support < THREAD_SERIALIZED; PaRSEC may hang")
        end
    end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nodes = MPI.Comm_size(comm)
    
    if rank == 0
        println("MPI: rank=$rank, nodes=$nodes")
    end
    
    # Validate parameters
    if M < 1 || N < 1 || MB < 1 || NB < 1 || P < 1 || KP < 1 || KQ < 1 || iter < 1 || R < 1
        if rank == 0
            println("Error: Wrong value is passed!")
            println("M=$M N=$N MB=$MB NB=$NB P=$P KP=$KP KQ=$KQ iter=$iter R=$R")
        end
        exit(1)
    end
    
    # Number of column tiles and buffers
    NNB = div(N + NB - 1, NB)  # Number of column tiles
    MMB = div(M + MB - 1, MB)  # Number of row tiles
    
    if MMB < 2
        if rank == 0
            println("Error: At least two buffers needed, got $MMB (ceil(M/MB) with M=$M, MB=$MB)")
        end
        exit(1)
    end
    
    # Calculate FLOPS (same formula as C code)
    # FLOPS_STENCIL_1D(n) = iter * (2*(2*R+1)) * n
    flops = iter * (2 * (2*R + 1)) * N * MB
    
    if rank == 0
        println("FLOPS: $flops")
        println("Number of buffers: MMB=$MMB, NNB=$NNB")
        println()
    end
    
    # Run the official stencil workflow
    info = run_stencil_official(M, N, MB, NB, iter, R; P=P, KP=KP, KQ=KQ, cores=cores)
    
    if rank == 0
        # Single line output for easy plotting with parameter names (like Python version)
        Q = div(info["nodes"], P)
        @printf("M=%d\tN=%d\tMB=%d\tNB=%d\titer=%d\tR=%d\tP=%d\tQ=%d\tparsec_init_time=%.9f\tinit_data_time=%.9f\texec_time=%.9f\tgflops=%.6f\n",
                M, N, MB, NB, iter, R, P, Q,
                info["parsec_init_time"],
                info["init_data_time"],
                info["exec_time"],
                info["gflops"])
    end
    
    # Finalize MPI only if this script initialized it
    if mpi_initialized_by_us && MPI.Initialized() && !MPI.Finalized()
        MPI.Finalize()
    end
end

function parse_args(args::Vector{String})::Dict{String, Any}
    """Parse command line arguments"""
    defaults = Dict(
        "M" => 8,
        "N" => 12,
        "MB" => 4,
        "NB" => 4,
        "iter" => 3,
        "R" => 1,
        "P" => 1,
        "KP" => 1,
        "KQ" => 1,
        "cores" => -1,
    )
    
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            if i < length(args)
                i += 1
                val = args[i]
                if haskey(defaults, key)
                    if isa(defaults[key], Int)
                        defaults[key] = parse(Int, val)
                    else
                        defaults[key] = val
                    end
                end
            end
        end
        i += 1
    end
    
    defaults
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

