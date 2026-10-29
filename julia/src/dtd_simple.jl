"""
Simplified DTD Interface for PaRSEC4Julia

This module provides a clean, Pythonic DTD API for Julia, designed to match
the dtd_simple_gemm.py interface while leveraging existing PaRSEC4Julia infrastructure.
"""

module DTDSimple

using MPI
using LinearAlgebra
using Printf


# Import types from the main module (will be included)
export ParsecDTDContext, ParsecDTDTaskpool, ParsecDTDTaskClass, ParsecMatrixBlockCyclic,
       start!, wait, add_taskpool, create_tile_full_arena, destroy_arena_datatype, fini,
       free, release, insert_task_with_task_class, create_task_class, add_chore_to_task_class,
       init, tile_of, dtd_data_collection_init, flush_all, destroy, local_buffer,
       parsec_redistribute_dtd, parsec_redistribute, parsec_redistribute_ptg, get_kernel_by_name,
    insert_task_with_callback, create_callback_task_class,
       PARSEC_INPUT, PARSEC_INOUT, PARSEC_OUTPUT, PARSEC_AFFINITY, PARSEC_VALUE, PARSEC_PUSHOUT, PASSED_BY_REF,
       PARSEC_DEV_CPU, PARSEC_DEV_CUDA,
       SIZEOF_INT, SIZEOF_DOUBLE, SIZEOF_PTR, PARSEC_DTD_EMPTY_FLAG,
    register_kernel!, unregister_all_kernels!, start_julia_workers, stop_julia_workers,
    drain_callbacks!,
       KERNEL_ID_INIT_TILE, KERNEL_ID_GEMM, prepare_mpi!

# ============================================================================
# MPI bootstrap
# ============================================================================

"""
    prepare_mpi!()

OpenMPI 4.1.8 on this cluster is not built with Slurm PMI. Inside an
`srun` allocation it still sees `SLURM_JOB_ID` and tries a direct srun
launch, which fails. For an interactive shell child, drop Slurm/PMI
variables so MPI_Init behaves like a login-node singleton.

Verified on compute: unsetting SLURM_/PMIX_/PMI_ lets both redistribute
examples run back-to-back. OMPI_MCA_ess=singleton does not work here.
"""
function prepare_mpi!()
    parent = try
        ppid = ccall(:getppid, Cint, ())
        strip(read("/proc/$ppid/comm", String))
    catch
        ""
    end
    interactive = parent in ("bash", "zsh", "sh", "fish", "csh", "tcsh")
    under_slurm = haskey(ENV, "SLURM_JOB_ID") || haskey(ENV, "SLURM_JOBID")
    inherited_pmix = haskey(ENV, "PMIX_NAMESPACE") || haskey(ENV, "PMIX_RANK") || haskey(ENV, "PMI_FD")
    interactive && (under_slurm || inherited_pmix) || return

    for key in collect(keys(ENV))
        if startswith(key, "SLURM_") || startswith(key, "PMIX_") || startswith(key, "PMI_") ||
           startswith(key, "OMPI_MCA_")
            delete!(ENV, key)
        end
    end
    return
end

function __init__()
    prepare_mpi!()
end

# ============================================================================
# Constants (from parsec/interfaces/dtd/insert_function.h)
# ============================================================================

# Op types (upper 20 bits)
const PARSEC_INPUT =       0x100000
const PARSEC_OUTPUT =      0x200000
const PARSEC_INOUT =       0x300000
const PARSEC_ATOMIC_WRITE = 0x400000
const PARSEC_SCRATCH =     0x500000
const PARSEC_VALUE =       0x600000
const PARSEC_REF =         0x700000
const PARSEC_GET_OP_TYPE = 0xf00000

# Flags  
const PARSEC_AFFINITY =    (1 << 16)
const PARSEC_DONT_TRACK =  (1 << 17)
const PARSEC_PUSHOUT =     (1 << 18)
const PARSEC_PULLIN =      (1 << 19)

# Size indicators
const PASSED_BY_REF = -2
const PARSEC_DTD_ARG_END = -1
const PARSEC_DTD_EMPTY_FLAG = 0

const PARSEC_DEV_CPU = 1      # Device bit flag
const PARSEC_DEV_CUDA = 2     # Device bit flag

const SIZEOF_INT = sizeof(Cint)
const SIZEOF_DOUBLE = sizeof(Cdouble)
const SIZEOF_PTR = sizeof(Ptr{Cvoid})

# Path to wrapper library
const libdtd_wrapper = joinpath(@__DIR__, "libdtd_wrapper.so")

# Keep VALUE argument buffers alive for the lifetime of the process.
# Using module-local storage avoids Julia 1.11+ restrictions on creating
# globals via `Main.some_name = ...` from other modules.
const _parsec_value_buffers_global = Any[]

# ============================================================================
# Type Definitions
# ============================================================================

mutable struct ParsecDTDContext
    ctx::Ptr{Cvoid}
end

mutable struct ParsecDTDTaskpool
    tp::Ptr{Cvoid}
    ctx::Ref{ParsecDTDContext}
end

mutable struct ParsecDTDTaskClass
    tc::Ptr{Cvoid}
    tp::Ref{ParsecDTDTaskpool}
    nargs::Int
    types::Vector{Cint}
end

mutable struct ParsecMatrixBlockCyclic
    dc::Ptr{Cvoid}
    mt::Int
    nt::Int
    mb::Int
    nb::Int
    m::Int
    n::Int
    P::Int
    Q::Int
end

# ============================================================================
# ParsecDTDContext Functions
# ============================================================================

function ParsecDTDContext(cores::Int=-1)
    ctx = ccall((:jl_parsec_init, libdtd_wrapper), Ptr{Cvoid}, (Cint,), Cint(cores))
    ctx == C_NULL && error("Failed to initialize PaRSEC")
    return ParsecDTDContext(ctx)
end

function start!(ctx::ParsecDTDContext)
    ret = ccall((:jl_parsec_context_start, libdtd_wrapper), Cint, (Ptr{Cvoid},), ctx.ctx)
    ret != 0 && error("Failed to start context")
end

function wait(ctx::ParsecDTDContext)
    ret = ccall((:jl_parsec_context_wait, libdtd_wrapper), Cint, (Ptr{Cvoid},), ctx.ctx)
    ret != 0 && error("Failed to wait on context")
end

function add_taskpool(ctx::ParsecDTDContext, tp::ParsecDTDTaskpool)
    ret = ccall((:jl_parsec_context_add_taskpool, libdtd_wrapper), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), ctx.ctx, tp.tp)
    ret != 0 && error("Failed to add taskpool")
end

function create_tile_full_arena(ctx::ParsecDTDContext, mb::Int, nb::Int)
    arena_id_ref = Ref{Cint}(0)
    ret = ccall((:jl_create_tile_full_arena, libdtd_wrapper), Cint, 
                (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}), ctx.ctx, Cint(mb), Cint(nb), arena_id_ref)
    ret != 0 && error("Failed to create arena")
    return arena_id_ref[]
end

function destroy_arena_datatype(ctx::ParsecDTDContext, arena_id::Int)
    ccall((:jl_destroy_arena_datatype, libdtd_wrapper), Cvoid, (Ptr{Cvoid}, Cint), ctx.ctx, Cint(arena_id))
end

function fini(ctx::ParsecDTDContext)
    if ctx.ctx != C_NULL
        ret = ccall((:jl_parsec_fini, libdtd_wrapper), Cint, (Ptr{Cvoid},), ctx.ctx)
        ret != 0 && @warn "parsec_fini failed"
        ctx.ctx = C_NULL
    end
end

# ============================================================================
# ParsecDTDTaskpool Functions
# ============================================================================

function ParsecDTDTaskpool()
    tp = ccall((:jl_parsec_dtd_taskpool_new, libdtd_wrapper), Ptr{Cvoid}, ())
    tp == C_NULL && error("Failed to create taskpool")
    result = ParsecDTDTaskpool(tp, Ref{ParsecDTDContext}(ParsecDTDContext(C_NULL)))
    return result
end

function ParsecDTDTaskpool(ctx::ParsecDTDContext)
    tp = ccall((:jl_parsec_dtd_taskpool_new, libdtd_wrapper), Ptr{Cvoid}, ())
    tp == C_NULL && error("Failed to create taskpool")
    return ParsecDTDTaskpool(tp, Ref(ctx))
end

function wait(tp::ParsecDTDTaskpool)
    ret = ccall((:jl_parsec_taskpool_wait, libdtd_wrapper), Cint, (Ptr{Cvoid},), tp.tp)
    ret < 0 && error("Failed to wait on taskpool (ret=$ret)")
end

function free(tp::ParsecDTDTaskpool)
    ccall((:jl_parsec_taskpool_free, libdtd_wrapper), Cvoid, (Ptr{Cvoid},), tp.tp)
end

"""
    create_task_class(tp::ParsecDTDTaskpool, name::String, ::Nothing,
                      params::Vector{Tuple{Int, Int}})::ParsecDTDTaskClass

Create a task class with specified parameters.

# Arguments
- `tp`: Taskpool to add the task class to
- `name`: Name of the task class
- `::Nothing`: Placeholder for compatibility with Python API
- `params`: Vector of (type, flags) tuples specifying task parameters

# Example
```julia
tc = create_task_class(tp, "init", nothing, [
    (PASSED_BY_REF, PARSEC_INOUT | tile_full | PARSEC_AFFINITY),
    (SIZEOF_INT, PARSEC_VALUE),
    (SIZEOF_INT, PARSEC_VALUE),
])
```
"""
function create_task_class(tp::ParsecDTDTaskpool, name::String, ::Nothing,
                           params::Vector{Tuple{Int, Int}})::ParsecDTDTaskClass
    nargs = length(params)
    types = Vector{Cint}(undef, nargs)
    flags = Vector{Cint}(undef, nargs)
    
    for (i, (type_sz, flag)) in enumerate(params)
        types[i] = Cint(type_sz)
        flags[i] = Cint(flag)
    end
    
    tc = ccall((:jl_parsec_dtd_create_task_class, libdtd_wrapper), Ptr{Cvoid},
               (Ptr{Cvoid}, Cstring, Cint, Ptr{Cint}, Ptr{Cint}),
               tp.tp, name, Cint(nargs), types, flags)
    tc == C_NULL && error("Failed to create task class")
    return ParsecDTDTaskClass(tc, Ref(tp), nargs, types)
end

"""
    create_task_class(tp, name, nargs, data_types, affinity_flags)

Create a task class with separate arrays for data types and affinity flags.
This matches the pattern used in simple_dtd_gemm_julia.jl.

data_types contains size-or-type for each argument (PARSEC_OUTPUT, PARSEC_VALUE, etc.)
affinity_flags contains affinity info (PARSEC_AFFINITY, PARSEC_DTD_EMPTY_FLAG, etc.)
"""
function create_task_class(tp::ParsecDTDTaskpool, name::String, nargs::Int,
                           data_types::Vector{Int}, affinity_flags::Vector{Int})::ParsecDTDTaskClass
    @assert length(data_types) == nargs
    @assert length(affinity_flags) == nargs
    
    types = Vector{Cint}(data_types)
    flags = Vector{Cint}(affinity_flags)
    
    tc = ccall((:jl_parsec_dtd_create_task_class, libdtd_wrapper), Ptr{Cvoid},
               (Ptr{Cvoid}, Cstring, Cint, Ptr{Cint}, Ptr{Cint}),
               tp.tp, name, Cint(nargs), types, flags)
    tc == C_NULL && error("Failed to create task class")
    return ParsecDTDTaskClass(tc, Ref(tp), nargs, types)
end

"""
    add_chore_to_task_class(tp::ParsecDTDTaskpool, tc::ParsecDTDTaskClass, 
                            device::Int, kernel::Union{Ptr{Cvoid}, Nothing})

Add a kernel (chore) to a task class for a specific device.

# Arguments
- `tp`: Taskpool containing the task class
- `tc`: Task class to add kernel to
- `device`: Device type (PARSEC_DEV_CPU or PARSEC_DEV_CUDA)
- `kernel`: Kernel function pointer (C_NULL for built-in GPU kernels)
"""
function add_chore_to_task_class(tp::ParsecDTDTaskpool, tc::ParsecDTDTaskClass, 
                                 device::Int, kernel::Union{Ptr{Cvoid}, Nothing})
    kernel_ptr = kernel === nothing ? C_NULL : kernel
    
    ret = ccall((:jl_parsec_dtd_task_class_add_chore, libdtd_wrapper), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Cvoid}),
                tp.tp, tc.tc, Cint(device), kernel_ptr)
    ret != 0 && error("Failed to add chore to task class")
end

"""
    release(tc::ParsecDTDTaskClass)

Release a task class and free its resources.
"""
function release(tc::ParsecDTDTaskClass)
    ccall((:jl_parsec_dtd_task_class_release, libdtd_wrapper), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}), tc.tp[].tp, tc.tc)
end

"""
    insert_task_with_task_class(tp::ParsecDTDTaskpool, tc::ParsecDTDTaskClass,
                                priority::Int, device::Int, name::String,
                                args::Vector{Tuple{Int, Any}})

Insert a task into the taskpool with specified arguments.

# Arguments
- `tp`: Target taskpool
- `tc`: Task class defining the task structure
- `priority`: Task priority (0 = normal)
- `device`: Target device (PARSEC_DEV_CPU or PARSEC_DEV_CUDA)
- `name`: Task instance name
- `args`: Vector of (flag, value) tuples for task arguments
"""
function insert_task_with_task_class(tp::ParsecDTDTaskpool, tc::ParsecDTDTaskClass,
                                      priority::Int, device::Int, name::String,
                                      args::AbstractVector{<:Tuple{<:Integer, Any}})
    nargs = length(args)
    nargs != tc.nargs && error("Argument count mismatch: expected $(tc.nargs), got $nargs")
    
    ins_flags = Vector{Cint}(undef, nargs)
    cargs = Vector{Ptr{Cvoid}}(undef, nargs)
    
    # First pass: collect flags and calculate total VALUE buffer size
    total_value_bytes = 0
    for i in 1:nargs
        flag, _ = args[i]

        if i <= length(tc.types)
            type_sz = tc.types[i]
            is_value_param = (type_sz == SIZEOF_INT || type_sz == SIZEOF_PTR || type_sz == SIZEOF_DOUBLE)
            if is_value_param
                # VALUE parameters must carry PARSEC_VALUE so the runtime treats them as scalars
                ins_flags[i] = Cint(flag | PARSEC_VALUE)
                total_value_bytes += type_sz
            else
                ins_flags[i] = Cint(flag)
            end
        else
            ins_flags[i] = Cint(flag)
        end
    end
    
    # Allocate single contiguous buffer for all VALUE parameters
    valbuf = nothing
    value_buffer_ref = nothing
    if total_value_bytes > 0
        # Create a byte buffer
        valbuf = Vector{UInt8}(undef, total_value_bytes)
        value_buffer_ref = valbuf  # Keep reference alive
    end
    
    # Second pass: fill in arguments, copying VALUE parameters into the buffer
    offset = 0
    for i in 1:nargs
        flag, val = args[i]
        
        if i <= length(tc.types)
            type_sz = tc.types[i]
            is_value_param = (type_sz == SIZEOF_INT || type_sz == SIZEOF_PTR || type_sz == SIZEOF_DOUBLE)
            
            if is_value_param
                # VALUE parameter: copy value into buffer at current offset
                if type_sz == SIZEOF_INT
                    # Use Cint (4 bytes) not Int64 (8 bytes)
                    int_val = convert(Cint, val)
                    unsafe_store!(Ptr{Cint}(pointer(valbuf) + offset), int_val)
                elseif type_sz == SIZEOF_PTR
                    # Pointer value (for signal addresses, etc.)
                    ptr_val = convert(UInt64, val)
                    unsafe_store!(Ptr{UInt64}(pointer(valbuf) + offset), ptr_val)
                elseif type_sz == SIZEOF_DOUBLE
                    float_val = convert(Cdouble, val)
                    unsafe_store!(Ptr{Cdouble}(pointer(valbuf) + offset), float_val)
                end
                cargs[i] = Ptr{Cvoid}(pointer(valbuf) + offset)
                offset += type_sz
            else
                # Data dependency parameter - convert to pointer (tile reference)
                cargs[i] = Ptr{Cvoid}(convert(UInt, val))
            end
        else
            # Fallback for unknown parameter type
            cargs[i] = Ptr{Cvoid}(convert(UInt, val))
        end
    end
    
    # Store buffer reference globally to keep it alive throughout program lifetime
    # (matching Python/Cython behavior where buffers are not freed)
    if value_buffer_ref !== nothing
        push!(_parsec_value_buffers_global, value_buffer_ref)
    end
    
    ret = ccall((:jl_parsec_dtd_insert_task, libdtd_wrapper), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Ptr{Cvoid}}),
                tp.tp, tc.tc, Cint(priority), Cint(device), Cint(nargs), ins_flags, cargs)
    ret != 0 && error("Failed to insert task")
end

# ============================================================================
# ParsecMatrixBlockCyclic Functions
# ============================================================================

function ParsecMatrixBlockCyclic()
    obj = ParsecMatrixBlockCyclic(C_NULL, 0, 0, 0, 0, 0, 0, 0, 0)
    return obj
end

"""
    ParsecMatrixBlockCyclic(rank, mb, nb, m, n, i, j, lm, ln, P, Q, kp, kq, name)

Create and initialize a matrix (matching dtd_test_simple_gemm.c pattern).
"""
function ParsecMatrixBlockCyclic(rank::Int, mb::Int, nb::Int, 
                                m::Int, n::Int, i::Int, j::Int, 
                                lm::Int, ln::Int, P::Int, Q::Int, 
                                kp::Int, kq::Int, name::String)
    obj = ParsecMatrixBlockCyclic()
    init(obj, name, rank, mb, nb, m, n, P, Q)
    return obj
end

"""
    init(mat::ParsecMatrixBlockCyclic, name::String, rank::Int,
         mb::Int, nb::Int, m::Int, n::Int, P::Int, Q::Int)

Initialize a block-cyclic distributed matrix.
"""
function init(mat::ParsecMatrixBlockCyclic, name::String, rank::Int,
              mb::Int, nb::Int, m::Int, n::Int, P::Int, Q::Int)
    dc = ccall((:jl_matrix_bc_alloc, libdtd_wrapper), Ptr{Cvoid}, ())
    dc == C_NULL && error("Failed to allocate matrix")
    
    ret = ccall((:jl_matrix_bc_init, libdtd_wrapper), Cint,
                (Ptr{Cvoid}, Cstring, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
                dc, name,
                Cint(0), Cint(3), Cint(rank),  # PARSEC_MATRIX_DOUBLE=0, PARSEC_MATRIX_TILE=3
                Cint(mb), Cint(nb),
                Cint(m), Cint(n),
                Cint(0), Cint(0),
                Cint(m), Cint(n),
                Cint(P), Cint(Q),
                Cint(1), Cint(1))  # kp=1, kq=1 (not 0, 0!)
    
    ret != 0 && error("Failed to init matrix: $ret")
    
    mat.dc = dc
    mat.mt = div(m + mb - 1, mb)
    mat.nt = div(n + nb - 1, nb)
    mat.mb = mb
    mat.nb = nb
    mat.m = m
    mat.n = n
    mat.P = P
    mat.Q = Q
end

"""
    tile_of(mat::ParsecMatrixBlockCyclic, m::Int, n::Int)

Get the tile reference for matrix[m, n].
"""
function tile_of(mat::ParsecMatrixBlockCyclic, m::Int, n::Int)
    tile = ccall((:jl_dtd_tile_of, libdtd_wrapper), Ptr{Cvoid},
                 (Ptr{Cvoid}, Cint, Cint), mat.dc, Cint(m), Cint(n))
    tile == C_NULL && error("tile_of returned NULL for ($m, $n)")
    return UInt(tile)
end

"""
    dtd_data_collection_init(mat::ParsecMatrixBlockCyclic)

Initialize the data collection for a matrix.
"""
function dtd_data_collection_init(mat::ParsecMatrixBlockCyclic)
    ccall((:jl_dtd_data_collection_init, libdtd_wrapper), Cvoid, (Ptr{Cvoid},), mat.dc)
end

"""
    flush_all(tp::ParsecDTDTaskpool, mat::ParsecMatrixBlockCyclic)

Flush all data associated with a matrix in the taskpool.
"""
function flush_all(tp::ParsecDTDTaskpool, mat::ParsecMatrixBlockCyclic)
    ccall((:jl_dtd_data_flush_all, libdtd_wrapper), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}), tp.tp, mat.dc)
end

"""
    destroy(mat::ParsecMatrixBlockCyclic)

Destroy a matrix and free its resources.
"""
function destroy(mat::ParsecMatrixBlockCyclic)
    if mat.dc != C_NULL
        ccall((:jl_matrix_bc_destroy, libdtd_wrapper), Cvoid, (Ptr{Cvoid},), mat.dc)
        mat.dc = C_NULL
    end
end

"""
    local_buffer(mat::ParsecMatrixBlockCyclic) -> Vector{Float64}

Return a Julia view of the local matrix storage backing this PaRSEC descriptor.
This is the Julia equivalent of Python's `ParsecMatrixBlockCyclic.local_buffer()`.
"""
function local_buffer(mat::ParsecMatrixBlockCyclic)
    mat.dc == C_NULL && error("matrix not initialized")
    ntile = ccall((:jl_matrix_bc_nb_local_tiles, libdtd_wrapper), Cint, (Ptr{Cvoid},), mat.dc)
    bsiz = ccall((:jl_matrix_bc_bsiz, libdtd_wrapper), Cint, (Ptr{Cvoid},), mat.dc)
    ptr_u = ccall((:jl_matrix_bc_mat_ptr, libdtd_wrapper), UInt, (Ptr{Cvoid},), mat.dc)
    n = Int(ntile) * Int(bsiz)
    n < 0 && error("invalid local buffer size")
    ptr_u == 0 && error("matrix local buffer pointer is NULL")
    return unsafe_wrap(Vector{Float64}, Ptr{Float64}(ptr_u), n; own=false)
end

"""
    parsec_redistribute_dtd(ctx, src, dst, size_row, size_col, disi_Y=0, disj_Y=0, disi_T=0, disj_T=0)
    parsec_redistribute_dtd(ctx, src, dst; size_row=src.m, size_col=src.n, disi_Y=0, disj_Y=0, disi_T=0, disj_T=0)

Redistribute a submatrix from `src` to `dst` using PaRSEC DTD redistribute.

Matches the Python `py_parsec.dtd.parsec_redistribute_dtd` signature. Optional
displacements keep PaRSEC's Y/T names (`dcY` is the source, `dcT` is the target).
The keyword form is Julia-specific and defaults the submatrix size to `src`.
"""
function parsec_redistribute_dtd(ctx::ParsecDTDContext,
                                 src::ParsecMatrixBlockCyclic,
                                 dst::ParsecMatrixBlockCyclic,
                                 size_row::Int, size_col::Int,
                                 disi_Y::Int=0, disj_Y::Int=0,
                                 disi_T::Int=0, disj_T::Int=0)
    (ctx.ctx == C_NULL || src.dc == C_NULL || dst.dc == C_NULL) && error("invalid context or matrix")
    ret = ccall((:jl_parsec_redistribute_dtd, libdtd_wrapper), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint, Cint),
                ctx.ctx, src.dc, dst.dc,
                Cint(size_row), Cint(size_col),
                Cint(disi_Y), Cint(disj_Y),
                Cint(disi_T), Cint(disj_T))
    ret != 0 && error("parsec_redistribute_dtd failed (rc=$ret)")
    return nothing
end

function parsec_redistribute_dtd(ctx::ParsecDTDContext,
                                 src::ParsecMatrixBlockCyclic,
                                 dst::ParsecMatrixBlockCyclic;
                                 size_row::Int=src.m,
                                 size_col::Int=src.n,
                                 disi_Y::Int=0, disj_Y::Int=0,
                                 disi_T::Int=0, disj_T::Int=0)
    parsec_redistribute_dtd(ctx, src, dst, size_row, size_col, disi_Y, disj_Y, disi_T, disj_T)
end

"""
    parsec_redistribute(ctx, src, dst, size_row, size_col, disi_Y=0, disj_Y=0, disi_T=0, disj_T=0)
    parsec_redistribute(ctx, src, dst; size_row=src.m, size_col=src.n, disi_Y=0, disj_Y=0, disi_T=0, disj_T=0)

Redistribute a submatrix from `src` to `dst` using PaRSEC PTG redistribute.

Matches the Python `py_parsec.dtd.parsec_redistribute` signature.
`parsec_redistribute_ptg` is a Julia alias for the same PTG entry point.
"""
function parsec_redistribute(ctx::ParsecDTDContext,
                             src::ParsecMatrixBlockCyclic,
                             dst::ParsecMatrixBlockCyclic,
                             size_row::Int, size_col::Int,
                             disi_Y::Int=0, disj_Y::Int=0,
                             disi_T::Int=0, disj_T::Int=0)
    (ctx.ctx == C_NULL || src.dc == C_NULL || dst.dc == C_NULL) && error("invalid context or matrix")
    ret = ccall((:jl_parsec_redistribute, libdtd_wrapper), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint, Cint),
                ctx.ctx, src.dc, dst.dc,
                Cint(size_row), Cint(size_col),
                Cint(disi_Y), Cint(disj_Y),
                Cint(disi_T), Cint(disj_T))
    ret != 0 && error("parsec_redistribute failed (rc=$ret)")
    return nothing
end

function parsec_redistribute(ctx::ParsecDTDContext,
                             src::ParsecMatrixBlockCyclic,
                             dst::ParsecMatrixBlockCyclic;
                             size_row::Int=src.m,
                             size_col::Int=src.n,
                             disi_Y::Int=0, disj_Y::Int=0,
                             disi_T::Int=0, disj_T::Int=0)
    parsec_redistribute(ctx, src, dst, size_row, size_col, disi_Y, disj_Y, disi_T, disj_T)
end

const parsec_redistribute_ptg = parsec_redistribute

"""
    get_kernel_by_name(kernel_name::String, device_type::Int)::Ptr{Cvoid}

Get a kernel function pointer by name.
device_type: PARSEC_DEV_CPU or PARSEC_DEV_CUDA
"""
function get_kernel_by_name(kernel_name::String, device_type::Int)::Ptr{Cvoid}
    kernel_ptr = ccall((:jl_get_kernel_by_name, libdtd_wrapper), Ptr{Cvoid},
                      (Cstring, Cint), kernel_name, Cint(device_type))
    return kernel_ptr
end

# ============================================================================
# Julia Bridge: Request Structure (mirror C struct)
# ============================================================================

"""
Julia-side mirror of C julia_req_t structure
Must match C struct layout in dtd_wrapper.c
"""
struct JuliaRequest
    kind::Cint
    kernel_id::Cint
    signal_ptr::Ptr{Cvoid}
    A::Ptr{Cvoid}
    B::Ptr{Cvoid}
    C::Ptr{Cvoid}
    mb::Cint
    nb::Cint
    kb::Cint
    lda::Cint
    ldb::Cint
    ldc::Cint
    alpha::Cdouble
    beta::Cdouble
    # Note: lock, cv, state fields are not accessed from Julia
end

# Kernel IDs (must match C side)
const KERNEL_ID_INIT_TILE = 1
const KERNEL_ID_GEMM = 2

# Request kinds (must match C side)
const REQ_KIND_CALLBACK = 1
const REQ_KIND_GEMM = 2
const REQ_KIND_INIT = 3

# ============================================================================
# Julia Bridge: Kernel Registry
# ============================================================================

"""
Global registry mapping kernel_id -> Julia function
"""
const JULIA_KERNELS = Dict{Int, Function}()

"""
    register_kernel!(id::Int, func::Function)

Register a Julia kernel function with an ID.
"""
function register_kernel!(id::Int, func::Function)
    JULIA_KERNELS[id] = func
    if id == 1
        println("Registered Julia kernel 1: julia_kernel_init_tile!")
    elseif id == 2
        println("Registered Julia kernel 2: julia_kernel_gemm!")
    else
        println("Registered Julia kernel $id")
    end
end

"""
    unregister_all_kernels!()

Clear all registered kernels.
"""
function unregister_all_kernels!()
    empty!(JULIA_KERNELS)
end

# ============================================================================
# Julia Bridge: Worker Pool
# ============================================================================

"""
Global worker tasks array
"""
const JULIA_WORKERS = Task[]
const WORKER_SHUTDOWN = Ref(false)

"""
    julia_worker_loop()

Worker loop: blocks waiting for C requests, executes Julia kernels, signals completion.
This runs in a Julia Task (coroutine).
"""
function julia_worker_loop()
    worker_id = Threads.threadid()
    if worker_id == 1 && Threads.nthreads() > 1
        # Avoid blocking the main thread; respawn on another thread
        Threads.@spawn julia_worker_loop()
        return
    end
    debug = get(ENV, "PARSEC_JULIA_DEBUG", "") != ""
    if debug
        println(stderr, "[Worker $worker_id] Started")
    end
    
    while !WORKER_SHUTDOWN[]
        # Block waiting for request from C side
        req_ptr = ccall((:jl_parsec_pop_req, libdtd_wrapper), 
                   Ptr{Cvoid}, ())
        
        if req_ptr == C_NULL
            if WORKER_SHUTDOWN[]
                # Shutdown signal
                break
            end
            # No request yet; yield to avoid busy waiting
            sleep(0.001)
            continue
        end
        
        # Read request fields (unsafe_load interprets C struct)
        req = unsafe_load(Ptr{JuliaRequest}(req_ptr))
        
        req_kind = Int(req.kind)
        kernel_id = Int(req.kernel_id)
        
        if debug
            println(stderr, "[Worker $worker_id] req=$(req_ptr) kind=$req_kind kernel_id=$kernel_id A=$(UInt(req.A)) B=$(UInt(req.B)) C=$(UInt(req.C)) mb=$(req.mb) nb=$(req.nb) kb=$(req.kb)")
        end

        if req_kind == REQ_KIND_CALLBACK
            signal_ptr = Ptr{Cint}(req.signal_ptr)
            if signal_ptr != C_NULL
                unsafe_store!(signal_ptr, Cint(1))
            end
            ccall((:jl_parsec_mark_done, libdtd_wrapper),
                  Cvoid, (Ptr{Cvoid}, Cint), req_ptr, 0)
            continue
        end

        if req.A == C_NULL || (kernel_id == KERNEL_ID_GEMM && (req.B == C_NULL || req.C == C_NULL))
            println(stderr, "[Worker $worker_id] NULL data pointer(s) in request, skipping")
            ccall((:jl_parsec_mark_done, libdtd_wrapper),
                  Cvoid, (Ptr{Cvoid}, Cint), req_ptr, -1)
            continue
        end

        if !haskey(JULIA_KERNELS, kernel_id)
            println(stderr, "[Worker $worker_id] Unknown kernel_id: $kernel_id")
            ccall((:jl_parsec_mark_done, libdtd_wrapper), 
                  Cvoid, (Ptr{Cvoid}, Cint), req_ptr, -1)
            continue
        end
        
        # Get kernel function
        kernel_func = JULIA_KERNELS[kernel_id]
        
        try
            # Execute kernel based on ID
            if kernel_id == KERNEL_ID_INIT_TILE
                # Wrap tile as Julia array (no ownership)
                A = unsafe_wrap(Array, Ptr{Float64}(req.A), (req.mb, req.nb); own=false)
                seed_offset = req.kb  # Reused field
                kernel_func(A, req.mb, req.nb, seed_offset)
                
            elseif kernel_id == KERNEL_ID_GEMM
                # Wrap tiles as Julia arrays (no ownership)
                A = unsafe_wrap(Array, Ptr{Float64}(req.A), (req.mb, req.kb); own=false)
                B = unsafe_wrap(Array, Ptr{Float64}(req.B), (req.kb, req.nb); own=false)
                C = unsafe_wrap(Array, Ptr{Float64}(req.C), (req.mb, req.nb); own=false)
                kernel_func(A, B, C, req.alpha, req.beta)
            else
                println(stderr, "[Worker $worker_id] Unsupported kernel_id: $kernel_id")
            end
            
            # Signal completion (success)
            ccall((:jl_parsec_mark_done, libdtd_wrapper), 
                  Cvoid, (Ptr{Cvoid}, Cint), req_ptr, 0)
                  
        catch e
            println(stderr, "[Worker $worker_id] Kernel execution failed: $(e)")
            ccall((:jl_parsec_mark_done, libdtd_wrapper), 
                  Cvoid, (Ptr{Cvoid}, Cint), req_ptr, -1)
        end
    end
    
    println(stderr, "[Worker $worker_id] Exited")
end

"""
    start_julia_workers(nworkers::Int=2)

Start Julia worker tasks that will execute kernels requested by C proxy.
"""
function start_julia_workers(nworkers::Int=2)
    if !isempty(JULIA_WORKERS)
        println(stderr, "[WARN] Workers already started")
        return
    end
    
    # Initialize C-side bridge
    ccall((:parsec_julia_bridge_init, libdtd_wrapper), Cvoid, (Cint,), nworkers)
    
    WORKER_SHUTDOWN[] = false
    
    # Spawn worker tasks using Threads.@spawn (NOT @async)
    # Workers block on C calls, so they must run in separate threads
    for i in 1:nworkers
        task = Threads.@spawn julia_worker_loop()
        push!(JULIA_WORKERS, task)
    end
    
    println(stderr, "Started $nworkers Julia workers")
end

"""
    stop_julia_workers()

Signal workers to shutdown and wait for them to exit.
"""
function stop_julia_workers()
    if isempty(JULIA_WORKERS)
        return
    end
    
    WORKER_SHUTDOWN[] = true
    
    # Signal C side to wake workers
    ccall((:parsec_julia_bridge_shutdown, libdtd_wrapper), Cvoid, ())
    
    # Wait for all workers to finish
    for task in JULIA_WORKERS
        try
            Base.wait(task)
        catch e
            println(stderr, "[WARN] Worker task error during shutdown: $(e)")
        end
    end
    
    empty!(JULIA_WORKERS)
    println(stderr, "All Julia workers stopped")
end

# ============================================================================
# Callback Mechanism (路线 A: Signal-based completion notification)
# ============================================================================

"""
    insert_task_with_callback(tp::ParsecDTDTaskpool, tc::ParsecDTDTaskClass,
                              priority::Int, device::Int, name::String,
                              args::Vector{Tuple{Int, Any}},
                              callback_func::Union{Function, Nothing}=nothing)

Insert a task with completion callback support.

This function:
1. Inserts the main compute task
2. Creates a callback task that depends on the output tile
3. Sets up Julia-side signal handling to execute the callback when complete

The callback will be executed in Julia runtime after the task completes.

# Arguments
- `tp`: Target taskpool
- `tc`: Task class defining the task structure
- `priority`: Task priority
- `device`: Target device
- `name`: Task instance name  
- `args`: Vector of (flag, value) tuples for task arguments
- `callback_func`: Optional Julia function to call after task completion

# Example
```julia
function my_callback(m::Int, n::Int, k::Int)
    println("GEMM(\$m, \$n, \$k) completed")
end

insert_task_with_callback(tp, tc, 0, device, "gemm_task",
    [(PARSEC_INPUT, A.tile_of(0,0)), ...],
    my_callback)
```
"""
function insert_task_with_callback(tp::ParsecDTDTaskpool, tc::ParsecDTDTaskClass,
                                    priority::Int, device::Int, name::String,
                                    args::AbstractVector{<:Tuple{<:Integer, Any}};
                                    callback_func::Union{Function, Nothing}=nothing,
                                    callback_tc::Union{ParsecDTDTaskClass, Nothing}=nothing,
                                    tile_full::Union{Int, Nothing}=nothing,
                                    output_tile::Union{UInt, Nothing}=nothing)
    # Insert the main compute task first
    insert_task_with_task_class(tp, tc, priority, device, name, args)
    
    # If no callback function provided, we're done
    callback_func === nothing && return
    
    # Resolve output tile for dependency
    out_tile = output_tile
    if out_tile === nothing
        for (flag, val) in args
            op = flag & PARSEC_GET_OP_TYPE
            if op == PARSEC_INOUT || op == PARSEC_OUTPUT
                out_tile = UInt(val)
                break
            end
        end
    end
    out_tile === nothing && error("callback requires an output tile dependency")
    
    # Ensure callback task class exists
    if callback_tc === nothing
        tile_full === nothing && error("callback requires tile_full to build callback task class")
        callback_tc = create_callback_task_class(tp, tile_full)
    end
    
    # Allocate signal and keep it alive
    if !isdefined(Main, :_parsec_callback_signals)
        Main._parsec_callback_signals = []
    end
    signal = Ref{Cint}(0)
    push!(Main._parsec_callback_signals, signal)
    signal_ptr = Base.unsafe_convert(Ptr{Cint}, signal)
    
    # Insert callback signal task dependent on output tile
    cb_args = [
        (PARSEC_INPUT, out_tile),
        (PARSEC_DTD_EMPTY_FLAG, Ptr{Cvoid}(signal_ptr)),
    ]
    insert_task_with_task_class(tp, callback_tc, priority, PARSEC_DEV_CPU, "$(name)_callback", cb_args)
    
    # Queue callback for later draining on main thread
    if !isdefined(Main, :_parsec_callback_queue)
        Main._parsec_callback_queue = Vector{Tuple{Ptr{Cint}, Function}}()
    end
    push!(Main._parsec_callback_queue, (signal_ptr, callback_func))
end

"""
    drain_callbacks!() -> Int

Run any completed callbacks whose signal has been set. Returns the number executed.
"""
function drain_callbacks!()::Int
    if !isdefined(Main, :_parsec_callback_queue)
        return 0
    end
    queue = Main._parsec_callback_queue
    isempty(queue) && return 0

    executed = 0
    remaining = Vector{Tuple{Ptr{Cint}, Function}}()
    for (signal_ptr, callback_func) in queue
        if unsafe_load(signal_ptr) != 0
            try
                callback_func()
            catch e
                @warn "Callback execution failed" exception=e
            end
            executed += 1
        else
            push!(remaining, (signal_ptr, callback_func))
        end
    end
    Main._parsec_callback_queue = remaining
    return executed
end

"""
    create_callback_task_class(tp::ParsecDTDTaskpool, name::String="callback")::ParsecDTDTaskClass

Create a callback signal task class.

The callback task takes a single VALUE parameter: the pointer to the signal variable.
When executed, it sets *(volatile int*)signal_ptr = 1 to notify Julia.

# Arguments
- `tp`: Target taskpool
- `name`: Name for the task class

# Returns
Task class for callback signaling
"""
function create_callback_task_class(tp::ParsecDTDTaskpool, tile_full::Int;
                                    name::String="callback")::ParsecDTDTaskClass
    # Callback task has 2 arguments: output tile dependency + signal pointer
    tc = create_task_class(tp, name, nothing, [
        (Int(PASSED_BY_REF), Int(PARSEC_INPUT | tile_full)),
        (Int(SIZEOF_PTR), Int(PARSEC_VALUE)),
    ])
    
    # Add CPU kernel for callback signaling
    kernel_ptr = get_kernel_by_name("callback_signal", PARSEC_DEV_CPU)
    kernel_ptr == C_NULL && error("callback_signal kernel not found")
    add_chore_to_task_class(tp, tc, PARSEC_DEV_CPU, kernel_ptr)
    
    return tc
end

"""
    wait_for_signal(signal_ptr::Ptr{Cint}, timeout_sec::Float64=0.0)::Bool

Wait for a signal to be set (non-blocking if timeout_sec=0).

# Arguments
- `signal_ptr`: Pointer to volatile int signal variable
- `timeout_sec`: Timeout in seconds (0 = non-blocking poll)

# Returns
true if signal was set, false if timeout
"""
function wait_for_signal(signal_ptr::Ptr{Cint}, timeout_sec::Float64=0.0)::Bool
    if timeout_sec <= 0
        # Non-blocking check
        return unsafe_load(signal_ptr) != 0
    end
    
    # Blocking wait with timeout
    start_time = time()
    while true
        if unsafe_load(signal_ptr) != 0
            return true
        end
        if (time() - start_time) > timeout_sec
            return false
        end
        sleep(0.001)  # 1ms sleep to avoid busy-waiting
    end
end

"""
    reset_signal(signal_ptr::Ptr{Cint})

Reset a signal variable to 0.
"""
function reset_signal(signal_ptr::Ptr{Cint})
    unsafe_store!(signal_ptr, Cint(0))
end

end  # module DTDSimple

