"""
Core type definitions for PaRSEC4Julia
"""

"""
    ParsecContext

Represents a PaRSEC execution context that manages the runtime system.
"""
mutable struct ParsecContext
    nb_cores::Int
    nb_vp::Int
    virtual_processes::Vector{Any}
    initialized::Bool
    c_ptr::Ptr{Cvoid}  # C pointer to parsec_context_t
    
    function ParsecContext(; nb_cores::Int = -1)
        new(nb_cores, 1, [], false, C_NULL)
    end
end

"""
    ParsecMatrixBlockCyclic

Represents a matrix with block-cyclic distribution for parallel computation.
"""
mutable struct ParsecMatrixBlockCyclic
    # Matrix dimensions
    m::Int                    # Global row dimension
    n::Int                    # Global column dimension
    mb::Int                   # Row tile size
    nb::Int                   # Column tile size
    lm::Int                   # Local row dimension
    ln::Int                   # Local column dimension
    i::Int                    # Starting row index
    j::Int                    # Starting column index
    
    # Process grid
    p::Int                    # Process grid rows
    q::Int                    # Process grid columns
    myrank::Int               # Current process rank
    
    # Cyclic distribution parameters
    kp::Int                   # K-cyclicity rows
    kq::Int                   # K-cyclicity columns
    ip::Int                   # Starting row in process grid
    jq::Int                   # Starting column in process grid
    
    # Matrix properties
    mtype::Int                # Matrix type (1 = double)
    storage::Int              # Storage type (0 = tile)
    
    # Data storage
    mat::Vector{Float64}      # Local matrix data
    key::String               # Data collection key
    
    # Derived properties
    nodes::Int                # Total number of processes
    nb_local_tiles::Int       # Number of local tiles
    bsiz::Int                 # Block size (mb * nb)
    
    function ParsecMatrixBlockCyclic(m::Int, n::Int, mb::Int, nb::Int, 
                                   p::Int, q::Int; 
                                   kp::Int = 1, kq::Int = 1, 
                                   ip::Int = 0, jq::Int = 0,
                                   mtype::Int = 1, storage::Int = 0)
        
        # For simplicity, use rank 0 (single process)
        myrank = 0
        
        # Calculate derived parameters
        nodes = p * q
        nb_local_tiles = _calculate_local_tiles(m, n, mb, nb, p, q, myrank)
        bsiz = mb * nb
        lm = m
        ln = n
        
        # Allocate data
        mat = zeros(Float64, nb_local_tiles * bsiz)
        
        new(m, n, mb, nb, lm, ln, 0, 0, p, q, myrank, kp, kq, ip, jq, 
            mtype, storage, mat, "dcA", nodes, nb_local_tiles, bsiz)
    end
end

"""
    StencilWeights

Stores weights for stencil operations.
"""
mutable struct StencilWeights
    weights::Vector{Float64}
    radius::Int
    
    function StencilWeights(radius::Int)
        weights = zeros(Float64, 2 * radius + 1)
        new(weights, radius)
    end
end

"""
    ParsecExecutionStream

Represents an execution stream for task execution.
"""
mutable struct ParsecExecutionStream
    id::Int
    context::ParsecContext
    
    function ParsecExecutionStream(id::Int, context::ParsecContext)
        new(id, context)
    end
end

"""
    ParsecTiledMatrix

Base type for tiled matrices.
"""
abstract type ParsecTiledMatrix end

# Make ParsecMatrixBlockCyclic a subtype of ParsecTiledMatrix
ParsecMatrixBlockCyclic <: ParsecTiledMatrix

"""
    ParsecDataCollection

Base type for data collections.
"""
abstract type ParsecDataCollection end

# Make ParsecMatrixBlockCyclic a subtype of ParsecDataCollection
ParsecMatrixBlockCyclic <: ParsecDataCollection

"""
    ParsecMatrixUplo

Matrix uplo type enumeration.
"""
@enum ParsecMatrixUplo begin
    PARSEC_MATRIX_FULL = 0
    PARSEC_MATRIX_UPPER = 1
    PARSEC_MATRIX_LOWER = 2
end

"""
    ParsecMatrixType

Matrix type enumeration.
"""
@enum ParsecMatrixType begin
    PARSEC_MATRIX_DOUBLE = 1
    PARSEC_MATRIX_FLOAT = 2
    PARSEC_MATRIX_COMPLEX_DOUBLE = 3
    PARSEC_MATRIX_COMPLEX_FLOAT = 4
end

"""
    ParsecStorageType

Storage type enumeration.
"""
@enum ParsecStorageType begin
    PARSEC_MATRIX_TILE = 0
    PARSEC_MATRIX_BLOCK = 1
end

"""
    ParsecDeviceType

Device type enumeration for DTD tasks.
"""
@enum ParsecDeviceType begin
    PARSEC_DEV_CPU = 1
    PARSEC_DEV_CUDA = 4
    PARSEC_DEV_HIP = 8
end

# DTD (Dynamic Task Discovery) constants
const TILE_FULL = -1

# DTD task flags - using actual PaRSEC values
const PARSEC_INPUT = 0x100000
const PARSEC_OUTPUT = 0x200000
const PARSEC_INOUT = 0x300000
const PARSEC_AFFINITY = 1<<16
const PARSEC_PUSHOUT = 8

# DTD argument types - using actual PaRSEC values
const PARSEC_VALUE = 0x600000
const PARSEC_REF = 0x700000
const PASSED_BY_REF = 2
const PARSEC_DTD_ARG_END = 0
const PARSEC_DTD_EMPTY_FLAG = 0

# Helper function to calculate local tiles
function _calculate_local_tiles(m::Int, n::Int, mb::Int, nb::Int, 
                               p::Int, q::Int, myrank::Int)
    """Calculate number of local tiles for this process"""
    tiles_per_row = div(m + mb - 1, mb)
    tiles_per_col = div(n + nb - 1, nb)
    return tiles_per_row * tiles_per_col
end
