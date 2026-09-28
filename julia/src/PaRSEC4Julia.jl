module PaRSEC4Julia

"""
    PaRSEC4Julia

A Julia interface to the PaRSEC (Parallel Runtime Scheduler and Execution Controller) 
framework for high-performance computing on distributed heterogeneous systems.
"""

using MPI
using LinearAlgebra
using Printf
using Random
using Statistics

# Include core modules
include("types.jl")
include("context.jl")
include("matrix.jl")
include("stencil.jl")
include("utils.jl")
include("parsec_c_wrapper.jl")
include("dtd_simple.jl")

# Export main types and functions
export ParsecContext
export ParsecMatrixBlockCyclic
export parsec_stencil_1D
export parsec_init
export parsec_fini
export parsec_apply
export parsec_matrix_block_cyclic_init
export parsec_data_allocate
export parsec_data_collection_set_key
export parsec_dtd_insert_task_c

end # module