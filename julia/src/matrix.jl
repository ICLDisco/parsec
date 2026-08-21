"""
PaRSEC matrix operations and data distribution functions
"""

# Include the C wrapper
include("parsec_c_wrapper.jl")

"""
    parsec_matrix_block_cyclic_init(matrix::ParsecMatrixBlockCyclic, 
                                   mtype::Int, storage::Int, myrank::Int,
                                   mb::Int, nb::Int, lm::Int, ln::Int,
                                   i::Int, j::Int, m::Int, n::Int,
                                   p::Int, q::Int, kp::Int, kq::Int,
                                   ip::Int, jq::Int)

Initialize a block-cyclic matrix distribution.

# Arguments
- `matrix`: Matrix to initialize
- `mtype`: Matrix type (1 = double)
- `storage`: Storage type (0 = tile)
- `myrank`: Current process rank
- `mb`, `nb`: Tile dimensions
- `lm`, `ln`: Local matrix dimensions
- `i`, `j`: Starting point in global matrix
- `m`, `n`: Submatrix size
- `p`, `q`: Process grid dimensions
- `kp`, `kq`: K-cyclicity parameters
- `ip`, `jq`: Starting point on process grid

# Example
```julia
matrix = ParsecMatrixBlockCyclic(8, 8, 4, 4, 1, 1)
parsec_matrix_block_cyclic_init(matrix, 1, 0, 0, 4, 4, 8, 8, 
                               0, 0, 8, 8, 1, 1, 1, 1, 0, 0)
```
"""
function parsec_matrix_block_cyclic_init(matrix::ParsecMatrixBlockCyclic,
                                       mtype::Int, storage::Int, myrank::Int,
                                       mb::Int, nb::Int, lm::Int, ln::Int,
                                       i::Int, j::Int, m::Int, n::Int,
                                       p::Int, q::Int, kp::Int, kq::Int,
                                       ip::Int, jq::Int)
    
    # Update matrix parameters
    matrix.mtype = mtype
    matrix.storage = storage
    matrix.myrank = myrank
    matrix.mb = mb
    matrix.nb = nb
    matrix.lm = lm
    matrix.ln = ln
    matrix.i = i
    matrix.j = j
    matrix.m = m
    matrix.n = n
    matrix.p = p
    matrix.q = q
    matrix.kp = kp
    matrix.kq = kq
    matrix.ip = ip
    matrix.jq = jq
    
    # Recalculate derived parameters
    matrix.nodes = p * q
    matrix.nb_local_tiles = _calculate_local_tiles(m, n, mb, nb, p, q, myrank)
    matrix.bsiz = mb * nb
    
    # Reallocate data if needed
    required_size = matrix.nb_local_tiles * matrix.bsiz
    if length(matrix.mat) != required_size
        matrix.mat = zeros(Float64, required_size)
    end
    
    # Call the actual PaRSEC function
    result = parsec_matrix_block_cyclic_init_c(
        pointer_from_objref(matrix),
        Cint(mtype),
        Cint(storage),
        Cint(m),
        Cint(n),
        Cint(mb),
        Cint(nb),
        Cint(i),
        Cint(j),
        Ptr{Cvoid}(pointer(matrix.mat)),
        Cint(lm),
        Cint(8)  # nb_elems_per_line
    )
    
    if result != 0
        error("Failed to initialize block-cyclic matrix - parsec_matrix_block_cyclic_init returned $result")
    end
    
    println("✓ Matrix block cyclic initialized using libparsec.so: $(matrix.m)x$(matrix.n), tiles: $(matrix.nb_local_tiles), rank: $myrank")
end

"""
    parsec_data_allocate(size::Int, dtype::Type = Float64)

Allocate data for PaRSEC operations.

# Arguments
- `size`: Size of data to allocate
- `dtype`: Data type (default: Float64)

# Returns
- `Vector{dtype}`: Allocated data array

# Example
```julia
data = parsec_data_allocate(1000)
```
"""
function parsec_data_allocate(size::Int, dtype::Type = Float64)
    return zeros(dtype, size)
end

"""
    parsec_data_collection_set_key(collection::ParsecDataCollection, key::String)

Set the key for a data collection.

# Arguments
- `collection`: Data collection
- `key`: Key string

# Example
```julia
parsec_data_collection_set_key(matrix, "dcA")
```
"""
function parsec_data_collection_set_key(collection::ParsecDataCollection, key::String)
    if isa(collection, ParsecMatrixBlockCyclic)
        collection.key = key
    end
end

"""
    parsec_apply(ctx::ParsecContext, uplo::ParsecMatrixUplo, 
                 matrix::ParsecTiledMatrix, op, args)

Apply an operation to a tiled matrix.

# Arguments
- `ctx`: PaRSEC context
- `uplo`: Matrix uplo type
- `matrix`: Tiled matrix
- `op`: Operation to apply (function or nothing)
- `args`: Arguments for the operation

# Example
```julia
parsec_apply(ctx, PARSEC_MATRIX_FULL, matrix, nothing, 1)
```
"""
function parsec_apply(ctx::ParsecContext, uplo::ParsecMatrixUplo,
                     matrix::ParsecTiledMatrix, op, args)
    
    if isa(matrix, ParsecMatrixBlockCyclic)
        if op === nothing
            # Initialize data
            _stencil_1D_init_ops(matrix, args)
        else
            # Apply operation
            op(matrix, args)
        end
    end
end

# Add specific method for ParsecMatrixBlockCyclic
function parsec_apply(ctx::ParsecContext, uplo::ParsecMatrixUplo,
                     matrix::ParsecMatrixBlockCyclic, op, args)
    
    if op === nothing
        # Initialize data using Julia simulation (since we don't have the C operators)
        _stencil_1D_init_ops(matrix, args)
    else
        # For now, use Julia simulation since the C operators are not available
        # In a real implementation, we would register the Julia functions as C callbacks
        # and let PaRSEC call them through the task system
        
        # Apply operation using Julia simulation
        if op == stencil_1D_init_ops
            _stencil_1D_init_ops(matrix, args)
        elseif op == CORE_stencil_1D
            # Apply stencil kernel to all tiles
            for tile_idx in 1:matrix.nb_local_tiles
                tile_data = _get_tile(matrix, tile_idx)
                _CORE_stencil_1D(tile_data, _initialize_stencil_weights(args), args)
                _set_tile(matrix, tile_idx, tile_data)
            end
        else
            # Generic operation
            op(matrix, args)
        end
    end
end

"""
    parsec_tiled_matrix_destroy(matrix::ParsecTiledMatrix)

Destroy a tiled matrix and free its resources.

# Arguments
- `matrix`: Tiled matrix to destroy

# Example
```julia
parsec_tiled_matrix_destroy(matrix)
```
"""
function parsec_tiled_matrix_destroy(matrix::ParsecTiledMatrix)
    if isa(matrix, ParsecMatrixBlockCyclic)
        # Clear data
        matrix.mat = Float64[]
        println("Matrix destroyed")
    end
end

"""
    parsec_data_free(data::Vector)

Free allocated data.

# Arguments
- `data`: Data vector to free

# Example
```julia
parsec_data_free(matrix.mat)
```
"""
function parsec_data_free(data::Vector)
    # In Julia, garbage collection handles this automatically
    # This is mainly for API compatibility
    empty!(data)
end

# Helper functions for matrix operations

"""
    _get_tile(matrix::ParsecMatrixBlockCyclic, tile_idx::Int)

Get a specific tile from the matrix.

# Arguments
- `matrix`: Block-cyclic matrix
- `tile_idx`: Tile index

# Returns
- `Matrix{Float64}`: Tile data
"""
function _get_tile(matrix::ParsecMatrixBlockCyclic, tile_idx::Int)
    if tile_idx >= matrix.nb_local_tiles
        return zeros(Float64, matrix.mb, matrix.nb)
    end
    
    start_idx = tile_idx * matrix.bsiz + 1
    end_idx = start_idx + matrix.bsiz - 1
    
    if end_idx > length(matrix.mat)
        return zeros(Float64, matrix.mb, matrix.nb)
    end
    
    return reshape(matrix.mat[start_idx:end_idx], matrix.mb, matrix.nb)
end

"""
    _set_tile(matrix::ParsecMatrixBlockCyclic, tile_idx::Int, tile_data::Matrix{Float64})

Set a specific tile in the matrix.

# Arguments
- `matrix`: Block-cyclic matrix
- `tile_idx`: Tile index
- `tile_data`: Tile data to set
"""
function _set_tile(matrix::ParsecMatrixBlockCyclic, tile_idx::Int, tile_data::Matrix{Float64})
    if tile_idx >= matrix.nb_local_tiles
        return
    end
    
    start_idx = tile_idx * matrix.bsiz + 1
    end_idx = start_idx + matrix.bsiz - 1
    
    if end_idx > length(matrix.mat)
        return
    end
    
    # Ensure tile_data is the right size
    if size(tile_data) != (matrix.mb, matrix.nb)
        resized_tile = zeros(Float64, matrix.mb, matrix.nb)
        min_mb = min(size(tile_data, 1), matrix.mb)
        min_nb = min(size(tile_data, 2), matrix.nb)
        resized_tile[1:min_mb, 1:min_nb] = tile_data[1:min_mb, 1:min_nb]
        tile_data = resized_tile
    end
    
    matrix.mat[start_idx:end_idx] = vec(tile_data)
end

"""
    _stencil_1D_init_ops(matrix::ParsecMatrixBlockCyclic, R::Int)

Initialize stencil data with ghost regions.

# Arguments
- `matrix`: Block-cyclic matrix
- `R`: Radius of ghost region
"""
function _stencil_1D_init_ops(matrix::ParsecMatrixBlockCyclic, R::Int)
    for tile_idx in 1:matrix.nb_local_tiles
        tile_data = _get_tile(matrix, tile_idx)
        _stencil_1D_init_tile(tile_data, R)
        _set_tile(matrix, tile_idx, tile_data)
    end
end

"""
    _stencil_1D_init_tile(tile_data::Matrix{Float64}, R::Int)

Initialize a single tile with stencil data.

# Arguments
- `tile_data`: Tile data to initialize
- `R`: Radius of ghost region
"""
function _stencil_1D_init_tile(tile_data::Matrix{Float64}, R::Int)
    mb, nb = size(tile_data)
    
    # Initialize main region: i*1.0 + j*1.0
    for j in (R+1):(nb-R)
        for i in 1:mb
            tile_data[i, j] = Float64(i-1) + Float64(j-1)
        end
    end
    
    # Initialize ghost regions to 0.0
    for j in 1:R  # Left ghost
        for i in 1:mb
            tile_data[i, j] = 0.0
        end
    end
    
    for j in (nb-R+1):nb  # Right ghost
        for i in 1:mb
            tile_data[i, j] = 0.0
        end
    end
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

# DTD (Dynamic Task Discovery) functions

"""
    parsec_data_collection_set_key(dc::ParsecDataCollection, key::String)

Set the key for a data collection (for DTD operations).
"""
function parsec_data_collection_set_key(dc::ParsecDataCollection, key::String)
    if isa(dc, ParsecMatrixBlockCyclic)
        dc.key = key
        println("✓ Data collection key set to: $key")
    else
        error("Unsupported data collection type")
    end
end

"""
    parsec_data_allocate(size::Int) -> Ptr{Cvoid}

Allocate memory for PaRSEC data.
"""
function parsec_data_allocate(size::Int)::Ptr{Cvoid}
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_data_allocate, PARSEC_LIB),
            Ptr{Cvoid},
            (Csize_t,),
            Csize_t(size)
        )
    else
        # For simulation, allocate Julia memory
        return pointer(zeros(UInt8, size))
    end
end

"""
    parsec_data_free(ptr::Ptr{Cvoid})

Free PaRSEC data memory.
"""
function parsec_data_free(ptr::Ptr{Cvoid})
    if USING_REAL_PARSEC
        ccall(
            (:parsec_data_free, PARSEC_LIB),
            Cvoid,
            (Ptr{Cvoid},),
            ptr
        )
    end
    # For simulation, Julia GC will handle cleanup
end
