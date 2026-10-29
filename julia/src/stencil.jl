"""
PaRSEC stencil computation functions
"""

# Include the C wrapper
include("parsec_c_wrapper.jl")

using Printf

"""
    parsec_stencil_1D(matrix::ParsecMatrixBlockCyclic, iterations::Int, radius::Int)

Run 1D stencil computation on a block-cyclic matrix using real PaRSEC task system.

# Arguments
- `matrix`: Block-cyclic matrix to compute on
- `iterations`: Number of stencil iterations
- `radius`: Stencil radius

# Example
```julia
parsec_stencil_1D(matrix, 10, 1)
```
"""
function parsec_stencil_1D(matrix::ParsecMatrixBlockCyclic, iterations::Int, radius::Int)
    println("Running stencil_1D: $iterations iterations, radius $radius using real PaRSEC task system")
    
    # Get the global PaRSEC context
    ctx = get_parsec_context()
    
    if ctx.c_ptr == C_NULL
        error("PaRSEC context not initialized")
    end
    
    # Initialize stencil data using PaRSEC task system
    println("Initializing stencil data using PaRSEC apply...")
    parsec_apply(ctx, PARSEC_MATRIX_FULL, matrix, stencil_1D_init_ops, radius)
    
    # Run stencil iterations using PaRSEC task system
    println("Running stencil iterations using PaRSEC apply...")
    for iteration in 1:iterations
        # Use PaRSEC apply with the core stencil kernel
        parsec_apply(ctx, PARSEC_MATRIX_FULL, matrix, CORE_stencil_1D, radius)
    end
    
    println("✓ Stencil computation completed using real PaRSEC task system: $iterations iterations executed")
end

"""
    _initialize_stencil_weights(radius::Int)

Initialize stencil weights for 1D stencil computation.

# Arguments
- `radius`: Stencil radius

# Returns
- `Vector{Float64}`: Weight array
"""
function _initialize_stencil_weights(radius::Int)
    weight_1D = zeros(Float64, 2 * radius + 1)
    
    for jj in 1:radius
        weight_1D[jj + radius + 1] = 1.0 / (2.0 * jj * radius)
        weight_1D[-jj + radius + 1] = -1.0 / (2.0 * jj * radius)
    end
    weight_1D[radius + 1] = 1.0
    
    return weight_1D
end

"""
    _apply_stencil_iteration(matrix::ParsecMatrixBlockCyclic, radius::Int, weights::Vector{Float64})

Apply one iteration of stencil computation.

# Arguments
- `matrix`: Block-cyclic matrix
- `radius`: Stencil radius
- `weights`: Stencil weights
"""
function _apply_stencil_iteration(matrix::ParsecMatrixBlockCyclic, radius::Int, weights::Vector{Float64})
    for tile_idx in 1:matrix.nb_local_tiles
        tile_data = _get_tile(matrix, tile_idx)
        _CORE_stencil_1D(tile_data, weights, radius)
        _set_tile(matrix, tile_idx, tile_data)
    end
end

"""
    _CORE_stencil_1D(tile_data::Matrix{Float64}, weights::Vector{Float64}, radius::Int)

Core 1D stencil kernel computation.

# Arguments
- `tile_data`: Tile data (modified in-place)
- `weights`: Stencil weights
- `radius`: Stencil radius
"""
function _CORE_stencil_1D(tile_data::Matrix{Float64}, weights::Vector{Float64}, radius::Int)
    mb, nb = size(tile_data)
    
    # Create output array
    output_tile = copy(tile_data)
    
    for j in (radius+1):(nb-radius)
        for i in 1:mb
            # Apply stencil: weighted sum of neighbors
            output_tile[i, j] = 0.0
            for jj in -radius:radius
                if jj == 0
                    weight = 1.0
                else
                    weight = 1.0 / (2.0 * abs(jj) * radius)
                    if jj < 0
                        weight = -weight
                    end
                end
                output_tile[i, j] += weight * tile_data[i, j + jj]
            end
        end
    end
    
    # Copy result back
    tile_data[:] = output_tile[:]
end

"""
    parsec_stencil_init_1D(ctx::ParsecContext, matrix::ParsecMatrixBlockCyclic, radius::Int)

Initialize 1D stencil data using real PaRSEC.

# Arguments
- `ctx`: PaRSEC context
- `matrix`: Block-cyclic matrix
- `radius`: Stencil radius

# Example
```julia
parsec_stencil_init_1D(ctx, matrix, 1)
```
"""
function parsec_stencil_init_1D(ctx::ParsecContext, matrix::ParsecMatrixBlockCyclic, radius::Int)
    println("Initializing 1D stencil data with radius $radius using real PaRSEC")
    
    if ctx.c_ptr == C_NULL
        error("PaRSEC context not initialized")
    end
    
    # Call the real PaRSEC stencil init function
    result = parsec_stencil_init_1D_c(
        ctx.c_ptr,
        pointer_from_objref(matrix),
        Cint(radius)
    )
    
    if result != 0
        error("Failed to initialize stencil data - parsec_stencil_init_1D returned $result")
    end
    
    println("✓ 1D stencil data initialized using real PaRSEC")
end

"""
    stencil_1D_init_ops(es::ParsecExecutionStream, descA::ParsecTiledMatrix, 
                        A::Matrix{Float64}, uplo::ParsecMatrixUplo, 
                        m::Int, n::Int, args)

Stencil 1D initialization operator for PaRSEC task system using real PaRSEC.

# Arguments
- `es`: Execution stream
- `descA`: Tiled matrix descriptor
- `A`: Matrix data
- `uplo`: Matrix uplo type
- `m`: Tile row index
- `n`: Tile column index
- `args`: Arguments (radius)

# Returns
- `Int`: Status code (0 for success)
"""
function stencil_1D_init_ops(es::ParsecExecutionStream, descA::ParsecTiledMatrix,
                            A::Matrix{Float64}, uplo::ParsecMatrixUplo,
                            m::Int, n::Int, args)
    
    if isa(descA, ParsecMatrixBlockCyclic)
        R = args
        
        # Call the real PaRSEC stencil init ops function
        result = stencil_1D_init_ops_c(
            pointer_from_objref(es),
            pointer_from_objref(descA),
            pointer(A),
            Cint(uplo.value),
            Cint(m),
            Cint(n),
            pointer([Cint(R)])
        )
        
        if result != 0
            error("Failed to run stencil init ops - stencil_1D_init_ops returned $result")
        end
    end
    
    return 0  # Success
end

"""
    CORE_stencil_1D(OUT::Matrix{Float64}, IN::Matrix{Float64}, 
                   weights::Vector{Float64}, mb::Int, nb::Int, 
                   lda::Int, R::Int)

Core stencil 1D kernel function using real PaRSEC.

# Arguments
- `OUT`: Output matrix
- `IN`: Input matrix
- `weights`: Stencil weights
- `mb`: Row tile size
- `nb`: Column tile size
- `lda`: Leading dimension
- `R`: Stencil radius
"""
function CORE_stencil_1D(OUT::Matrix{Float64}, IN::Matrix{Float64},
                        weights::Vector{Float64}, mb::Int, nb::Int,
                        lda::Int, R::Int)
    
    # Call the real PaRSEC CORE stencil function
    CORE_stencil_1D_c(
        pointer(OUT),
        pointer(IN),
        pointer(weights),
        Cint(mb),
        Cint(nb),
        Cint(lda),
        Cint(R)
    )
end

"""
    calculate_stencil_flops(N::Int, MB::Int, iterations::Int, radius::Int)

Calculate FLOPS for stencil computation.

# Arguments
- `N`: Column dimension
- `MB`: Row tile size
- `iterations`: Number of iterations
- `radius`: Stencil radius

# Returns
- `Float64`: FLOPS count
"""
function calculate_stencil_flops(N::Int, MB::Int, iterations::Int, radius::Int)
    return Float64(iterations) * (2 * (2 * radius + 1)) * Float64(N * MB)
end

"""
    print_matrix(A::Matrix{Float64}, mb::Int, nb::Int, 
                 disi::Int, disj::Int, lda::Int)

Print a matrix for debugging.

# Arguments
- `A`: Matrix to print
- `mb`: Row size to print
- `nb`: Column size to print
- `disi`: Row displacement
- `disj`: Column displacement
- `lda`: Leading dimension
"""
function print_matrix(A::Matrix{Float64}, mb::Int, nb::Int,
                     disi::Int, disj::Int, lda::Int)
    for i in 1:mb
        for j in 1:nb
            @printf("%.6f ", A[disi+i, disj+j])
        end
        println()
    end
    println()
end
