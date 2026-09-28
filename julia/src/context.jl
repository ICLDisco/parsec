"""
PaRSEC context management functions
"""

# Include the C wrapper
include("parsec_c_wrapper.jl")

"""
    parsec_init(cores::Int = -1, argc::Int = 0, argv::Vector{String} = String[])

Initialize a PaRSEC context.

# Arguments
- `cores`: Number of cores to use (-1 for all available)
- `argc`: Number of command line arguments
- `argv`: Command line arguments

# Returns
- `ParsecContext`: Initialized PaRSEC context

# Example
```julia
ctx = parsec_init(4)  # Use 4 cores
```
"""
function parsec_init(cores::Int = -1, argc::Int = 0, argv::Vector{String} = String[])
    println("Initializing PaRSEC context with $cores cores using libparsec.so...")
    
    # Initialize MPI first (required by PaRSEC) - do this before any PaRSEC calls
    result = mpi_init_c()
    if result != 0
        error("Failed to initialize MPI - MPI_Init returned $result")
    end
    println("✓ MPI initialized")
    
    # Convert Julia strings to C strings
    c_argv = [pointer(x) for x in argv]
    c_argv_ptr = argc > 0 ? pointer(c_argv) : Ptr{Cstring}(C_NULL)
    
    # Call the actual PaRSEC function
    parsec_ptr = parsec_init_c(cores, argc, c_argv_ptr)
    
    if parsec_ptr == C_NULL
        error("Failed to initialize PaRSEC context - libparsec.so not found or initialization failed")
    end
    
    # Create Julia context wrapper
    ctx = ParsecContext(nb_cores = cores)
    ctx.c_ptr = parsec_ptr
    ctx.initialized = true
    
    # Initialize virtual processes (simulated)
    ctx.nb_vp = max(1, cores > 0 ? cores : 1)
    ctx.virtual_processes = [Dict("id" => i, "status" => "ready") for i in 1:ctx.nb_vp]
    
    println("✓ PaRSEC context initialized using libparsec.so with $(ctx.nb_vp) virtual processes")
    return ctx
end

"""
    parsec_fini(ctx::ParsecContext)

Finalize a PaRSEC context and clean up resources.

# Arguments
- `ctx`: PaRSEC context to finalize

# Example
```julia
parsec_fini(ctx)
```
"""
function parsec_fini(ctx::ParsecContext)
    if ctx.initialized && ctx.c_ptr != C_NULL
        println("Finalizing PaRSEC context using libparsec.so...")
        
        # Call the actual PaRSEC function
        result = parsec_fini_c(pointer_from_objref(ctx))
        
        if result != 0
            println("Warning: parsec_fini returned non-zero status: $result")
        end
        
        # Clean up Julia side
        ctx.c_ptr = C_NULL
        ctx.initialized = false
        
        # Clean up virtual processes
        for vp in ctx.virtual_processes
            vp["status"] = "terminated"
        end
        
        # Finalize MPI
        result = mpi_finalize_c()
        if result != 0
            println("Warning: MPI_Finalize returned non-zero status: $result")
        end
        println("✓ MPI finalized")
        
        println("✓ PaRSEC context finalized using libparsec.so")
    else
        println("Warning: Attempting to finalize uninitialized context or context already finalized")
    end
end

"""
    finalize(ctx::ParsecContext)

Alias for parsec_fini for Julia's finalization system.
"""
function finalize(ctx::ParsecContext)
    parsec_fini(ctx)
end

"""
    parsec_context_get_nb_cores(ctx::ParsecContext)

Get the number of cores in the PaRSEC context.

# Arguments
- `ctx`: PaRSEC context

# Returns
- `Int`: Number of cores
"""
function parsec_context_get_nb_cores(ctx::ParsecContext)
    return ctx.nb_cores
end

"""
    parsec_context_get_nb_vp(ctx::ParsecContext)

Get the number of virtual processes in the PaRSEC context.

# Arguments
- `ctx`: PaRSEC context

# Returns
- `Int`: Number of virtual processes
"""
function parsec_context_get_nb_vp(ctx::ParsecContext)
    return ctx.nb_vp
end

"""
    parsec_context_is_initialized(ctx::ParsecContext)

Check if the PaRSEC context is initialized.

# Arguments
- `ctx`: PaRSEC context

# Returns
- `Bool`: True if initialized, false otherwise
"""
function parsec_context_is_initialized(ctx::ParsecContext)
    return ctx.initialized
end
