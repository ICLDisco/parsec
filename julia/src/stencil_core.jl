"""
Direct PaRSEC core API bindings for stencil - NO DTD!
Exactly mirrors the official testing_stencil_1D.c workflow.
"""

module StencilCore

using MPI

export ParsecCoreContext,
    ParsecMatrix,
    PARSEC_MATRIX_FULL,
    PARSEC_MATRIX_DOUBLE,
    PARSEC_MATRIX_TILE,
    parsec_init,
    parsec_fini,
    parsec_apply,
    parsec_stencil_1D,
    parsec_redistribute,
    parsec_matrix_init!

# Constants for matrix types
const PARSEC_MATRIX_FULL = 123
const PARSEC_MATRIX_DOUBLE = 3
const PARSEC_MATRIX_TILE = 1

# ============================================================================
# Type Definitions
# ============================================================================

"""
    ParsecCoreContext

Wrapper for parsec_context_t - manages the PaRSEC runtime context.
"""
mutable struct ParsecCoreContext
    c_ptr::Ptr{Cvoid}  # Pointer to parsec_context_t
    initialized::Bool
    
    function ParsecCoreContext(nb_cores::Int = -1)
        ctx = new(C_NULL, false)
        _parsec_init(ctx, nb_cores)
        ctx
    end
end

"""
    ParsecMatrix

Wrapper for parsec_matrix_block_cyclic_t - manages matrix data and distribution.
"""
mutable struct ParsecMatrix
    c_ptr::Ptr{Cvoid}  # Pointer to parsec_matrix_block_cyclic_t
    mat_ptr::Ptr{Cvoid}  # Pointer to matrix data buffer
    key::String
    m::Int             # Global row dimension
    n::Int             # Global column dimension
    mb::Int            # Row tile size
    nb::Int            # Column tile size
    mt::Int            # Number of row tiles
    nt::Int            # Number of column tiles
    initialized::Bool
    
    function ParsecMatrix()
        new(C_NULL, C_NULL, "", 0, 0, 0, 0, 0, 0, false)
    end
end

# ============================================================================
# C Function Bindings
# ============================================================================

function _parsec_init(ctx::ParsecCoreContext, nb_cores::Int)
    """Initialize PaRSEC context by calling parsec_init()"""
    try
        # Create reference parameters for argc/argv
        argc_ref = Ref{Cint}(0)
        argv_ref = Ref{Ptr{Cstring}}(C_NULL)
        
        # Call parsec_init(nb_cores, &argc, &argv)
        c_ptr = @ccall "libparsec".parsec_init(
            Cint(nb_cores)::Cint,
            argc_ref::Ref{Cint},
            argv_ref::Ref{Ptr{Cstring}}
        )::Ptr{Cvoid}
        
        if c_ptr == C_NULL
            error("parsec_init failed - PaRSEC context is NULL")
        end
        
        ctx.c_ptr = c_ptr
        ctx.initialized = true
    catch e
        error("Failed to initialize PaRSEC context: $e")
    end
end

"""
    parsec_init(nb_cores::Int = -1) -> ParsecCoreContext

Initialize PaRSEC runtime context.
- nb_cores: number of cores to use (-1 = auto-detect all available cores)

Returns a ParsecCoreContext that must be finalized with parsec_fini().
"""
function parsec_init(nb_cores::Int = -1)
    ParsecCoreContext(nb_cores)
end

"""
    parsec_fini(ctx::ParsecCoreContext)

Finalize and clean up PaRSEC context.
"""
function parsec_fini(ctx::ParsecCoreContext)
    if ctx.initialized && ctx.c_ptr != C_NULL
        try
            c_ptr_ref = Ref(ctx.c_ptr)
            @ccall "libparsec".parsec_fini(
                c_ptr_ref::Ref{Ptr{Cvoid}}
            )::Cint
            ctx.c_ptr = C_NULL
            ctx.initialized = false
        catch e
            @warn "Error finalizing PaRSEC context: $e"
        end
    end
end

"""
    parsec_matrix_init!(mat::ParsecMatrix, myrank::Int, mb::Int, nb::Int,
                       lm::Int, ln::Int, P::Int, Q::Int;
                       kp::Int = 1, kq::Int = 1, 
                       mtype::Int = PARSEC_MATRIX_DOUBLE,
                       storage::Int = PARSEC_MATRIX_TILE)

Initialize a block-cyclic distributed matrix.
"""
function parsec_matrix_init!(mat::ParsecMatrix, myrank::Int, mb::Int, nb::Int,
                            lm::Int, ln::Int, P::Int, Q::Int;
                            kp::Int = 1, kq::Int = 1,
                            mtype::Int = PARSEC_MATRIX_DOUBLE,
                            storage::Int = PARSEC_MATRIX_TILE)
    try
    # Allocate descriptor from C to ensure correct struct size/zeroing
    mat.c_ptr = @ccall "libstencil_jl".jl_block_cyclic_alloc()::Ptr{Cvoid}
        
        if mat.c_ptr == C_NULL
            error("Failed to allocate matrix descriptor")
        end
        
        # Call parsec_matrix_block_cyclic_init
        # parsec_matrix_block_cyclic_init(parsec_matrix_block_cyclic_t *mat,
        #                                  int mtype, int storage, int rank,
        #                                  int mb, int nb,
        #                                  int lm, int ln,
        #                                  int i, int j,
        #                                  int m, int n,
        #                                  int P, int Q,
        #                                  int kp, int kq,
        #                                  int ip, int jq)
        @ccall "libparsec".parsec_matrix_block_cyclic_init(
            mat.c_ptr::Ptr{Cvoid},
            Cint(mtype)::Cint,
            Cint(storage)::Cint,
            Cint(myrank)::Cint,
            Cint(mb)::Cint,
            Cint(nb)::Cint,
            Cint(lm)::Cint,
            Cint(ln)::Cint,
            Cint(0)::Cint,  # i
            Cint(0)::Cint,  # j
            Cint(lm)::Cint,  # m
            Cint(ln)::Cint,  # n
            Cint(P)::Cint,
            Cint(Q)::Cint,
            Cint(kp)::Cint,
            Cint(kq)::Cint,
            Cint(0)::Cint,  # ip
            Cint(0)::Cint   # jq
        )::Cvoid
        
        # Set matrix key (profiling helper if available). No-op if not compiled with PARSEC_PROF_TRACE.
        key_str = "dcA"
        # Note: parsec_data_collection_set_key might be compiled out; handled on C side when available.
        
        # Allocate data buffer via PaRSEC helpers to match descriptor layout
        mat.mat_ptr = @ccall "libstencil_jl".jl_block_cyclic_alloc_data(mat.c_ptr::Ptr{Cvoid})::Ptr{Cvoid}
        if mat.mat_ptr == C_NULL
            error("Failed to allocate matrix data buffer")
        end
        
        # Store matrix info (super fields already set on the C side)
        mat.m = lm
        mat.n = ln
        mat.mb = mb
        mat.nb = nb
        mat.mt = div(lm + mb - 1, mb)
        mat.nt = div(ln + nb - 1, nb)
        mat.key = key_str
        mat.initialized = true
        
    catch e
        error("Failed to initialize matrix: $e")
    end
end

"""
    parsec_apply(ctx::ParsecCoreContext, uplo::Int, mat::ParsecMatrix, radius::Int)

Initialize matrix tiles using parsec_apply with the standard initialization operator.
"""
function parsec_apply(ctx::ParsecCoreContext, uplo::Int, mat::ParsecMatrix, radius::Int)
    if !ctx.initialized
        error("PaRSEC context not initialized")
    end
    if !mat.initialized
        error("Matrix not initialized")
    end
    
    try
        # Prepare radius parameter
        radius_ptr = Ref{Cint}(Cint(radius))
        
        # Call jl_parsec_apply_init from our wrapper to use stencil_1D_init_ops
        ret = @ccall "libstencil_jl".jl_parsec_apply_init(
            ctx.c_ptr::Ptr{Cvoid},
            mat.c_ptr::Ptr{Cvoid},
            Cint(radius)::Cint
        )::Cint
        
        if ret != 0
            error("parsec_apply failed with return code $ret")
        end
    catch e
        error("Error in parsec_apply: $e")
    end
end

"""
    parsec_stencil_1D(ctx::ParsecCoreContext, mat::ParsecMatrix, iterations::Int, radius::Int)

Run the 1D stencil kernel.
"""
function parsec_stencil_1D(ctx::ParsecCoreContext, mat::ParsecMatrix, iterations::Int, radius::Int)
    if !ctx.initialized
        error("PaRSEC context not initialized")
    end
    if !mat.initialized
        error("Matrix not initialized")
    end
    
    try
        # Ensure weights are initialized, then call the kernel via wrapper
        @ccall "libstencil_jl".jl_init_weight_1D(Cint(radius)::Cint)::Cvoid
        ret = @ccall "libstencil_jl".jl_parsec_stencil_1D(
            ctx.c_ptr::Ptr{Cvoid},
            mat.c_ptr::Ptr{Cvoid},
            Cint(iterations)::Cint,
            Cint(radius)::Cint
        )::Cint
        
        if ret != 0
            error("parsec_stencil_1D failed with return code $ret")
        end
    catch e
        error("Error in parsec_stencil_1D: $e")
    end
end

"""
    parsec_redistribute(ctx, src, dst, size_row, size_col; disi_Y=0, disj_Y=0, disi_T=0, disj_T=0)

Redistribute a submatrix from `src` to `dst` using PaRSEC PTG redistribute.
"""
function parsec_redistribute(ctx::ParsecCoreContext,
                            src::ParsecMatrix,
                            dst::ParsecMatrix,
                            size_row::Int,
                            size_col::Int;
                            disi_Y::Int=0,
                            disj_Y::Int=0,
                            disi_T::Int=0,
                            disj_T::Int=0)
    if !ctx.initialized
        error("PaRSEC context not initialized")
    end
    if !src.initialized || !dst.initialized
        error("source/target matrix not initialized")
    end

    ret = @ccall "libparsec".parsec_redistribute(
        ctx.c_ptr::Ptr{Cvoid},
        src.c_ptr::Ptr{Cvoid},
        dst.c_ptr::Ptr{Cvoid},
        Cint(size_row)::Cint,
        Cint(size_col)::Cint,
        Cint(disi_Y)::Cint,
        Cint(disj_Y)::Cint,
        Cint(disi_T)::Cint,
        Cint(disj_T)::Cint
    )::Cint
    ret != 0 && error("parsec_redistribute failed (rc=$ret)")
    return nothing
end

# Cleanup finalizer
function Base.finalizer(ctx::ParsecCoreContext)
    parsec_fini(ctx)
end

function Base.finalizer(mat::ParsecMatrix)
    if mat.c_ptr != C_NULL
        @ccall "libstencil_jl".jl_block_cyclic_destroy(mat.c_ptr::Ptr{Cvoid})::Cvoid
        mat.c_ptr = C_NULL
        mat.mat_ptr = C_NULL
    end
end

end  # module StencilCore
