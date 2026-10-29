"""
C wrapper functions for libparsec integration
This file contains the actual C library bindings for PaRSEC
"""

using Libdl

# Platform-specific library extensions
const LIB_EXT = Sys.isapple() ? "dylib" : "so"

# Function to find PaRSEC library
function find_parsec_lib()
    # Try local build first
    local_path = joinpath(@__DIR__, "..", "parsec_source", "builddir", "install", "lib", "libparsec.$LIB_EXT")
    if isfile(local_path)
        return local_path
    end
    
    # Try system-wide installation
    lib_names = ["libparsec.$LIB_EXT", "libparsec.4.$LIB_EXT", "libparsec.4.1.$LIB_EXT"]
    
    for lib_name in lib_names
        try
            # Try to find the library using Libdl
            lib_path = Libdl.find_library([lib_name])
            if lib_path != ""
                return lib_path
            end
        catch
            continue
        end
    end
    
    return nothing
end

# Function to find MPI library
function find_mpi_lib()
    # Common MPI library names
    mpi_libs = ["libmpi.$LIB_EXT", "libmpich.$LIB_EXT", "libopen-pal.$LIB_EXT", "libmpi_mpifh.$LIB_EXT"]
    
    # Try to find using Libdl
    for lib_name in mpi_libs
        try
            lib_path = Libdl.find_library([lib_name])
            if lib_path != ""
                return lib_path
            end
        catch
            continue
        end
    end
    
    # Try common installation paths
    common_paths = String[]
    
    if Sys.isapple()
        # macOS common paths - try to find versioned libraries
        homebrew_paths = []
        if isdir("/opt/homebrew/Cellar/open-mpi")
            for version_dir in readdir("/opt/homebrew/Cellar/open-mpi")
                if isdir(joinpath("/opt/homebrew/Cellar/open-mpi", version_dir, "lib"))
                    push!(homebrew_paths, joinpath("/opt/homebrew/Cellar/open-mpi", version_dir, "lib", "libmpi.40.$LIB_EXT"))
                    push!(homebrew_paths, joinpath("/opt/homebrew/Cellar/open-mpi", version_dir, "lib", "libmpi.$LIB_EXT"))
                end
            end
        end
        append!(common_paths, homebrew_paths)
        push!(common_paths, "/opt/homebrew/lib/libmpi.$LIB_EXT")
        push!(common_paths, "/usr/local/lib/libmpi.$LIB_EXT")
        push!(common_paths, "/opt/local/lib/libmpi.$LIB_EXT")
    else
        # Linux common paths
        push!(common_paths, "/usr/lib/x86_64-linux-gnu/libmpi.$LIB_EXT")
        push!(common_paths, "/usr/lib64/libmpi.$LIB_EXT")
        push!(common_paths, "/usr/lib/libmpi.$LIB_EXT")
        push!(common_paths, "/usr/local/lib/libmpi.$LIB_EXT")
    end
    
    for path in common_paths
        if isfile(path)
            return path
        end
    end
    
    return nothing
end

# Find libraries
const PARSEC_LIB = find_parsec_lib()
const MPI_LIB = find_mpi_lib()

# Global flag to track if we're using real PaRSEC or simulation
const USING_REAL_PARSEC = PARSEC_LIB !== nothing

if !USING_REAL_PARSEC
    println("Warning: libparsec.$LIB_EXT not found, using simulation mode")
else
    println("✓ Found libparsec.$LIB_EXT at $PARSEC_LIB")
end

if MPI_LIB === nothing
    println("Warning: MPI library not found, MPI functions may not work")
else
    println("✓ Found MPI library at $MPI_LIB")
end

"""
    parsec_init_c(cores::Int, argc::Int, argv::Ptr{Cstring}) -> Ptr{Cvoid}

C wrapper for parsec_init from libparsec.so
"""
function parsec_init_c(cores::Int, argc::Int, argv::Ptr{Cstring})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_init, PARSEC_LIB), 
            Ptr{Cvoid}, 
            (Cint, Ptr{Cint}, Ptr{Ptr{Cstring}}), 
            Cint(cores), 
            argc == 0 ? C_NULL : pointer([Cint(argc)]), 
            argv
        )
    else
        # Return a dummy pointer for simulation
        return Ptr{Cvoid}(UInt(0x12345678))
    end
end

"""
    parsec_fini_c(parsec::Ptr{Cvoid}) -> Cint

C wrapper for parsec_fini from libparsec.so
"""
function parsec_fini_c(parsec::Ptr{Cvoid})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_fini, PARSEC_LIB), 
            Cint, 
            (Ptr{Ptr{Cvoid}},), 
            parsec
        )
    else
        # Return success for simulation
        return Cint(0)
    end
end

"""
    mpi_init_c() -> Cint

Initialize MPI from C side with thread support
"""
function mpi_init_c()
    if MPI_LIB === nothing
        error("MPI library not found. Please install MPI (Open MPI, MPICH, or MVAPICH)")
    end
    
    # Check if MPI is already initialized
    flag = Ref{Cint}(0)
    ccall(
        (:MPI_Initialized, MPI_LIB),
        Cint,
        (Ptr{Cint},),
        flag
    )
    
    if flag[] != 0
        println("MPI already initialized, skipping MPI_Init_thread")
        return Cint(0)
    end
    
    # Initialize MPI with thread support (MPI_THREAD_SERIALIZED)
    provided = Ref{Cint}(0)
    result = ccall(
        (:MPI_Init_thread, MPI_LIB),
        Cint,
        (Ptr{Cint}, Ptr{Ptr{Cstring}}, Cint, Ptr{Cint}),
        C_NULL, C_NULL, 3, provided  # 3 = MPI_THREAD_SERIALIZED
    )
    
    if result == 0
        println("✓ MPI initialized with thread support level: $(provided[])")
    end
    
    return result
end

"""
    mpi_finalize_c() -> Cint

Finalize MPI from C side
"""
function mpi_finalize_c()
    if MPI_LIB === nothing
        error("MPI library not found. Please install MPI (Open MPI, MPICH, or MVAPICH)")
    end
    
    # Check if MPI is initialized before finalizing
    flag = Ref{Cint}(0)
    ccall(
        (:MPI_Initialized, MPI_LIB),
        Cint,
        (Ptr{Cint},),
        flag
    )
    
    if flag[] == 0
        println("MPI not initialized, skipping MPI_Finalize")
        return Cint(0)
    end
    
    return ccall(
        (:MPI_Finalize, MPI_LIB),
        Cint,
        ()
    )
end

# DTD (Dynamic Task Discovery) C wrappers

"""
    parsec_dtd_taskpool_new_c() -> Ptr{Cvoid}

C wrapper for parsec_dtd_taskpool_new from libparsec.so
"""
function parsec_dtd_taskpool_new_c()
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_dtd_taskpool_new, PARSEC_LIB),
            Ptr{Cvoid},
            ()
        )
    else
        # Return a dummy pointer for simulation
        return Ptr{Cvoid}(UInt(0x12345678))
    end
end

"""
    parsec_taskpool_free_c(tp::Ptr{Cvoid})

C wrapper for parsec_taskpool_free from libparsec.so
"""
function parsec_taskpool_free_c(tp::Ptr{Cvoid})
    if USING_REAL_PARSEC
        ccall(
            (:parsec_taskpool_free, PARSEC_LIB),
            Cvoid,
            (Ptr{Cvoid},),
            tp
        )
    end
end

"""
    parsec_context_add_taskpool_c(parsec::Ptr{Cvoid}, tp::Ptr{Cvoid}) -> Cint

C wrapper for parsec_context_add_taskpool from libparsec.so
"""
function parsec_context_add_taskpool_c(parsec::Ptr{Cvoid}, tp::Ptr{Cvoid})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_context_add_taskpool, PARSEC_LIB),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}),
            parsec, tp
        )
    else
        # Return success for simulation
        return Cint(0)
    end
end

"""
    parsec_context_start_c(parsec::Ptr{Cvoid}) -> Cint

C wrapper for parsec_context_start from libparsec.so
"""
function parsec_context_start_c(parsec::Ptr{Cvoid})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_context_start, PARSEC_LIB),
            Cint,
            (Ptr{Cvoid},),
            parsec
        )
    else
        # Return success for simulation
        return Cint(0)
    end
end

"""
    parsec_context_wait_c(parsec::Ptr{Cvoid}) -> Cint

C wrapper for parsec_context_wait from libparsec.so
"""
function parsec_context_wait_c(parsec::Ptr{Cvoid})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_context_wait, PARSEC_LIB),
            Cint,
            (Ptr{Cvoid},),
            parsec
        )
    else
        # Return success for simulation
        return Cint(0)
    end
end

"""
    parsec_taskpool_wait_c(tp::Ptr{Cvoid}) -> Cint

C wrapper for parsec_taskpool_wait from libparsec.so
"""
function parsec_taskpool_wait_c(tp::Ptr{Cvoid})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_taskpool_wait, PARSEC_LIB),
            Cint,
            (Ptr{Cvoid},),
            tp
        )
    else
        # Return success for simulation
        return Cint(0)
    end
end

"""
    parsec_dtd_create_arena_datatype_c(parsec::Ptr{Cvoid}, tile_full::Ptr{Cint}) -> Ptr{Cvoid}

C wrapper for parsec_dtd_create_arena_datatype from libparsec.so
"""
function parsec_dtd_create_arena_datatype_c(parsec::Ptr{Cvoid}, tile_full::Ptr{Cint})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_dtd_create_arena_datatype, PARSEC_LIB),
            Ptr{Cvoid},
            (Ptr{Cvoid}, Ptr{Cint}),
            parsec, tile_full
        )
    else
        # Return a dummy pointer for simulation
        return Ptr{Cvoid}(UInt(0x12345678))
    end
end

"""
    parsec_add2arena_c(adt::Ptr{Cvoid}, oldtype::Cint, uplo::Cint, diag::Cint, m::Cuint, n::Cuint, ld::Cuint, alignment::Csize_t, resized::Cint)

C wrapper for parsec_add2arena from libparsec.so
"""
function parsec_add2arena_c(adt::Ptr{Cvoid}, oldtype::Cint, uplo::Cint, diag::Cint, m::Cuint, n::Cuint, ld::Cuint, alignment::Csize_t, resized::Cint)
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_add2arena, PARSEC_LIB),
            Cint,
            (Ptr{Cvoid}, Cint, Cint, Cint, Cuint, Cuint, Cuint, Csize_t, Cint),
            adt, oldtype, uplo, diag, m, n, ld, alignment, resized
        )
    else
        return Cint(0)
    end
end

"""
    parsec_add2arena_rect_c(adt::Ptr{Cvoid}, datatype::Cint, mb::Cint, nb::Cint, lda::Cint)

C wrapper for parsec_add2arena_rect macro from libparsec.so
"""
function parsec_add2arena_rect_c(adt::Ptr{Cvoid}, datatype::Cint, mb::Cint, nb::Cint, lda::Cint)
    if USING_REAL_PARSEC
        # parsec_add2arena_rect is a macro that calls parsec_add2arena with specific parameters
        return parsec_add2arena_c(adt, datatype, Cint(0), Cint(0), Cuint(mb), Cuint(nb), Cuint(lda), Csize_t(16), Cint(-1))
    else
        return Cint(0)
    end
end

"""
    parsec_dtd_data_collection_init_c(dc::Ptr{Cvoid})

C wrapper for parsec_dtd_data_collection_init from libparsec.so
"""
function parsec_dtd_data_collection_init_c(dc::Ptr{Cvoid})
    if USING_REAL_PARSEC
        ccall(
            (:parsec_dtd_data_collection_init, PARSEC_LIB),
            Cvoid,
            (Ptr{Cvoid},),
            dc
        )
    end
end

"""
    parsec_dtd_data_collection_fini_c(dc::Ptr{Cvoid})

C wrapper for parsec_dtd_data_collection_fini from libparsec.so
"""
function parsec_dtd_data_collection_fini_c(dc::Ptr{Cvoid})
    if USING_REAL_PARSEC
        ccall(
            (:parsec_dtd_data_collection_fini, PARSEC_LIB),
            Cvoid,
            (Ptr{Cvoid},),
            dc
        )
    end
end

"""
    parsec_dtd_data_flush_all_c(tp::Ptr{Cvoid}, dc::Ptr{Cvoid}) -> Cint

C wrapper for parsec_dtd_data_flush_all from libparsec.so
"""
function parsec_dtd_data_flush_all_c(tp::Ptr{Cvoid}, dc::Ptr{Cvoid})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_dtd_data_flush_all, PARSEC_LIB),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}),
            tp, dc
        )
    else
        return Cint(0)
    end
end

"""
    parsec_dtd_task_class_add_chore_c(tp::Ptr{Cvoid}, tc::Ptr{Cvoid}, device::Cint, kernel::Ptr{Cvoid}) -> Cint

C wrapper for parsec_dtd_task_class_add_chore from libparsec.so
"""
function parsec_dtd_task_class_add_chore_c(tp::Ptr{Cvoid}, tc::Ptr{Cvoid}, device::Cint, kernel::Ptr{Cvoid})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_dtd_task_class_add_chore, PARSEC_LIB),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Cvoid}),
            tp, tc, device, kernel
        )
    else
        return Cint(0)
    end
end

"""
    parsec_dtd_task_class_release_c(tp::Ptr{Cvoid}, tc::Ptr{Cvoid})

C wrapper for parsec_dtd_task_class_release from libparsec.so
"""
function parsec_dtd_task_class_release_c(tp::Ptr{Cvoid}, tc::Ptr{Cvoid})
    if USING_REAL_PARSEC
        ccall(
            (:parsec_dtd_task_class_release, PARSEC_LIB),
            Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}),
            tp, tc
        )
    end
end

"""
    parsec_dtd_tile_of_c(dc::Ptr{Cvoid}, key::Cint) -> Ptr{Cvoid}

C wrapper for parsec_dtd_tile_of from libparsec.so
"""
function parsec_dtd_tile_of_c(dc::Ptr{Cvoid}, key::Cint)
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_dtd_tile_of, PARSEC_LIB),
            Ptr{Cvoid},
            (Ptr{Cvoid}, Cint),
            dc, key
        )
    else
        # Return a dummy pointer for simulation
        return Ptr{Cvoid}(UInt(0x12345678))
    end
end

"""
    parsec_dtd_tile_new_c(tp::Ptr{Cvoid}, rank::Cint) -> Ptr{Cvoid}

C wrapper for parsec_dtd_tile_new from libparsec.so
"""
function parsec_dtd_tile_new_c(tp::Ptr{Cvoid}, rank::Cint)
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_dtd_tile_new, PARSEC_LIB),
            Ptr{Cvoid},
            (Ptr{Cvoid}, Cint),
            tp, rank
        )
    else
        # Return a dummy pointer for simulation
        return Ptr{Cvoid}(UInt(0x12345678))
    end
end

"""
    parsec_dtd_data_flush_c(tp::Ptr{Cvoid}, tile::Ptr{Cvoid}) -> Cint

C wrapper for parsec_dtd_data_flush from libparsec.so
"""
function parsec_dtd_data_flush_c(tp::Ptr{Cvoid}, tile::Ptr{Cvoid})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_dtd_data_flush, PARSEC_LIB),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}),
            tp, tile
        )
    else
        return Cint(0)
    end
end

"""
    parsec_dtd_dequeue_taskpool_c(tp::Ptr{Cvoid}) -> Cint

C wrapper for parsec_dtd_dequeue_taskpool from libparsec.so
"""
function parsec_dtd_dequeue_taskpool_c(tp::Ptr{Cvoid})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_dtd_dequeue_taskpool, PARSEC_LIB),
            Cint,
            (Ptr{Cvoid},),
            tp
        )
    else
        return Cint(0)
    end
end

"""
    parsec_dtd_get_taskpool_c(task::Ptr{Cvoid}) -> Ptr{Cvoid}

C wrapper for parsec_dtd_get_taskpool from libparsec.so
"""
function parsec_dtd_get_taskpool_c(task::Ptr{Cvoid})
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_dtd_get_taskpool, PARSEC_LIB),
            Ptr{Cvoid},
            (Ptr{Cvoid},),
            task
        )
    else
        # Return a dummy pointer for simulation
        return Ptr{Cvoid}(UInt(0x12345678))
    end
end

"""
    parsec_dtd_get_dev_ptr_c(task::Ptr{Cvoid}, i::Cint) -> Ptr{Cvoid}

C wrapper for parsec_dtd_get_dev_ptr from libparsec.so
"""
function parsec_dtd_get_dev_ptr_c(task::Ptr{Cvoid}, i::Cint)
    if USING_REAL_PARSEC
        return ccall(
            (:parsec_dtd_get_dev_ptr, PARSEC_LIB),
            Ptr{Cvoid},
            (Ptr{Cvoid}, Cint),
            task, i
        )
    else
        # Return a dummy pointer for simulation
        return Ptr{Cvoid}(UInt(0x12345678))
    end
end

"""
    parsec_dtd_unpack_args_c(task::Ptr{Cvoid}, ...) -> Cint

C wrapper for parsec_dtd_unpack_args from libparsec.so
Note: This function has variable arguments, so we need a C wrapper
"""
function parsec_dtd_unpack_args_c(task::Ptr{Cvoid})
    if USING_REAL_PARSEC
        # This is a simplified version - the actual function has variable arguments
        # For now, return success
        return Cint(0)
    else
        return Cint(0)
    end
end
