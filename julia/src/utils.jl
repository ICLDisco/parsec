"""
Utility functions for PaRSEC4Julia
"""

"""
    rank_neighbor(descA::ParsecTiledMatrix, m::Int, n::Int, m_max::Int, n_max::Int)

Get the rank of a neighbor tile.

# Arguments
- `descA`: Tiled matrix descriptor
- `m`: Row index
- `n`: Column index
- `m_max`: Maximum row index
- `n_max`: Maximum column index

# Returns
- `Int`: Rank of the neighbor (-999 if out of bounds)
"""
function rank_neighbor(descA::ParsecTiledMatrix, m::Int, n::Int, m_max::Int, n_max::Int)
    if (m >= 0) && (n >= 0) && (m <= m_max) && (n <= n_max)
        # Simplified rank calculation - in a real implementation this would
        # use the actual distribution logic
        return 0
    end
    return -999
end

"""
    move_submatrix(m::Int, n::Int, S::Matrix{Float64}, S_i::Int, S_j::Int, S_lda::Int,
                  D::Matrix{Float64}, D_i::Int, D_j::Int, D_lda::Int)

Copy submatrix from source to destination.

# Arguments
- `m`: Row size
- `n`: Column size
- `S`: Source matrix
- `S_i`, `S_j`: Source starting position
- `S_lda`: Source leading dimension
- `D`: Destination matrix
- `D_i`, `D_j`: Destination starting position
- `D_lda`: Destination leading dimension
"""
function move_submatrix(m::Int, n::Int, S::Matrix{Float64}, S_i::Int, S_j::Int, S_lda::Int,
                       D::Matrix{Float64}, D_i::Int, D_j::Int, D_lda::Int)
    for j in 1:n
        for i in 1:m
            D[D_j + j, D_i + i] = S[S_j + j, S_i + i]
        end
    end
end

"""
    sync_time_start()

Start timing measurement.

# Returns
- `Float64`: Start time
"""
function sync_time_start()
    return time()
end

"""
    sync_time_print(rank::Int, message::String, start_time::Float64)

Print timing information.

# Arguments
- `rank`: Process rank
- `message`: Message to print
- `start_time`: Start time from sync_time_start()
"""
function sync_time_print(rank::Int, message::String, start_time::Float64)
    elapsed = time() - start_time
    if rank == 0
        @printf("%s: %.6f seconds\n", message, elapsed)
    end
    return elapsed
end

"""
    validate_parameters(M::Int, N::Int, MB::Int, NB::Int, P::Int, 
                       KP::Int, KQ::Int, cores::Int, iter::Int, R::Int)

Validate PaRSEC parameters.

# Arguments
- `M`, `N`: Matrix dimensions
- `MB`, `NB`: Tile dimensions
- `P`: Process grid rows
- `KP`, `KQ`: K-cyclicity parameters
- `cores`: Number of cores
- `iter`: Number of iterations
- `R`: Stencil radius

# Returns
- `Bool`: True if parameters are valid

# Throws
- `ArgumentError`: If parameters are invalid
"""
function validate_parameters(M::Int, N::Int, MB::Int, NB::Int, P::Int,
                           KP::Int, KQ::Int, cores::Int, iter::Int, R::Int)
    
    if M < 1 || N < 1 || MB < 1 || NB < 1 || P < 1 || KP < 1 || KQ < 1 || iter < 1 || R < 1
        throw(ArgumentError("Invalid parameters: M=$M, N=$N, MB=$MB, NB=$NB, P=$P, KP=$KP, KQ=$KQ, cores=$cores, iter=$iter, R=$R"))
    end
    
    # Check minimum number of buffers
    MMB = div(M + MB - 1, MB)
    if MMB < 2
        throw(ArgumentError("At least two buffers needed, got $MMB (M=$M, MB=$MB)"))
    end
    
    return true
end

"""
    calculate_performance(flops::Float64, execution_time::Float64)

Calculate performance metrics.

# Arguments
- `flops`: FLOPS count
- `execution_time`: Execution time in seconds

# Returns
- `NamedTuple`: Performance metrics (gflops, efficiency)
"""
function calculate_performance(flops::Float64, execution_time::Float64)
    gflops = execution_time > 0 ? (flops / 1e9) / execution_time : 0.0
    efficiency = gflops > 0 ? gflops / 100.0 : 0.0  # Assuming 100 GFLOPS is 100% efficiency
    
    return (gflops = gflops, efficiency = efficiency)
end

"""
    print_parameters(M::Int, N::Int, MB::Int, NB::Int, P::Int, Q::Int,
                    KP::Int, KQ::Int, cores::Int, iter::Int, R::Int)

Print PaRSEC parameters in a formatted way.

# Arguments
- `M`, `N`: Matrix dimensions
- `MB`, `NB`: Tile dimensions
- `P`, `Q`: Process grid dimensions
- `KP`, `KQ`: K-cyclicity parameters
- `cores`: Number of cores
- `iter`: Number of iterations
- `R`: Stencil radius
"""
function print_parameters(M::Int, N::Int, MB::Int, NB::Int, P::Int, Q::Int,
                         KP::Int, KQ::Int, cores::Int, iter::Int, R::Int)
    
    println("PaRSEC Parameters:")
    println("  Matrix: $(M)x$(N)")
    println("  Tiles: $(MB)x$(NB)")
    println("  Process Grid: $(P)x$(Q)")
    println("  K-cyclicity: $(KP)x$(KQ)")
    println("  Cores: $cores")
    println("  Iterations: $iter")
    println("  Radius: $R")
end

"""
    get_parsec_context()

Get or create a global PaRSEC context (for compatibility with Python examples).

# Returns
- `ParsecContext`: Global PaRSEC context
"""
const _global_parsec_context = Ref{Union{ParsecContext, Nothing}}(nothing)

function get_parsec_context()
    if _global_parsec_context[] === nothing
        _global_parsec_context[] = parsec_init(1)
    end
    return _global_parsec_context[]
end

"""
    cleanup_global_context()

Clean up the global PaRSEC context.
"""
function cleanup_global_context()
    if _global_parsec_context[] !== nothing
        parsec_fini(_global_parsec_context[])
        _global_parsec_context[] = nothing
    end
end
