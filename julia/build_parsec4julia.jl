#!/usr/bin/env julia
"""
Build script for PaRSEC4Julia (in-tree).

This script lives at julia/build_parsec4julia.jl inside the PaRSEC source
tree.  It:
  1. Builds PaRSEC via CMake into ../build/install  (unless --skip-parsec)
  2. Compiles the Julia C wrappers (libdtd_wrapper.so, libstencil_jl.so)
  3. Writes parsec_env.sh for runtime library paths

Usage:
    julia build_parsec4julia.jl [--enable-cuda] [--enable-hip] [--enable-opencl]
                                [--enable-blas] [--skip-parsec] [--j NJOBS]
"""

using Printf

function parse_args(args)
    opts = Dict(
        :enable_cuda => false,
        :enable_hip => false,
        :enable_opencl => false,
        :enable_blas => false,
        :skip_parsec => false,
        :njobs => Base.Sys.CPU_THREADS,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--enable-cuda"
            opts[:enable_cuda] = true
            i += 1
        elseif arg == "--enable-hip"
            opts[:enable_hip] = true
            i += 1
        elseif arg == "--enable-opencl"
            opts[:enable_opencl] = true
            i += 1
        elseif arg == "--enable-blas"
            opts[:enable_blas] = true
            i += 1
        elseif arg == "--skip-parsec"
            opts[:skip_parsec] = true
            i += 1
        elseif arg == "--j"
            opts[:njobs] = parse(Int, args[i+1])
            i += 2
        elseif arg in ["-h", "--help"]
            println("PaRSEC4Julia Build Script (in-tree)")
            println("Usage: julia build_parsec4julia.jl [options]")
            println("Options:")
            println("  --enable-cuda      Enable CUDA support in PaRSEC")
            println("  --enable-hip       Enable HIP support")
            println("  --enable-opencl    Enable OpenCL support")
            println("  --enable-blas      Compile dtd_wrapper with HAVE_BLAS")
            println("  --skip-parsec      Only rebuild C wrappers")
            println("  --j NJOBS          Parallel build jobs")
            exit(0)
        else
            i += 1
        end
    end
    return opts
end

function run_cmd(cmd::Vector; cwd=nothing, env=nothing, verbose=true)
    if verbose
        println("Running: $(join(cmd, " "))")
    end

    original_dir = pwd()
    try
        if cwd !== nothing
            cd(cwd)
        end
        if env !== nothing
            run(setenv(Cmd(cmd), env))
        else
            run(Cmd(cmd))
        end
        return true
    catch e
        println(stderr, "ERROR: Command failed")
        if cwd !== nothing
            println(stderr, "  cwd: $cwd")
        end
        println(stderr, "  cmd: $(join(cmd, " "))")
        println(stderr, "  error: $e")
        return false
    finally
        cd(original_dir)
    end
end

function program_exists(prog::String)::Bool
    try
        readchomp(`which $prog`)
        return true
    catch
        return false
    end
end

function detect_parsec_libdir(install_prefix::String)::String
    candidates = ("lib64", "lib")
    for d in candidates
        libdir = joinpath(install_prefix, d)
        if isfile(joinpath(libdir, "libparsec.so")) ||
           isfile(joinpath(libdir, "libparsec.so.4")) ||
           isfile(joinpath(libdir, "libparsec.so.4.1.0"))
            return d
        end
    end
    for d in candidates
        if isdir(joinpath(install_prefix, d))
            return d
        end
    end
    return "lib64"
end

function mpicc_flags()::Vector{String}
    flags = String[]
    if !program_exists("mpicc")
        println(stderr, "WARNING: mpicc not found, MPI support may be limited")
        return flags
    end
    try
        mpi_show = readchomp(`mpicc -show`)
        for token in split(mpi_show)
            if startswith(token, "-I") || startswith(token, "-L") ||
               startswith(token, "-l") || startswith(token, "-Wl,")
                push!(flags, token)
            end
        end
    catch
        println(stderr, "WARNING: Could not get MPI flags from mpicc")
    end
    return flags
end

function mpi_lib_dirs()::Vector{String}
    dirs = String[]
    for token in mpicc_flags()
        if startswith(token, "-L")
            path = token[3:end]
            if !isempty(path) && isdir(path) && !(path in dirs)
                push!(dirs, path)
            end
        end
    end
    return dirs
end

function collect_blas_libs()::Vector{String}
    blas_libs = String[]
    if haskey(ENV, "MKLROOT")
        mklroot = ENV["MKLROOT"]
        println("[build] Detected MKLROOT: $mklroot")
        push!(blas_libs, "-L" * joinpath(mklroot, "lib"))
        push!(blas_libs, "-lmkl_rt", "-lpthread", "-lm", "-ldl", "-fopenmp")
        return blas_libs
    end
    if program_exists("pkg-config")
        try
            append!(blas_libs, split(readchomp(`pkg-config --libs blas`)))
            return blas_libs
        catch
        end
    end
    for libname in ("openblas", "blas")
        if isfile("/usr/lib/lib$(libname).so") || isfile("/usr/lib64/lib$(libname).so")
            push!(blas_libs, "-l$libname")
            break
        end
    end
    return blas_libs
end

function compile_shared(so_path::String, sources::Vector{String}, flags::Vector{String}; env=nothing)
    cmd = vcat(["gcc", "-shared", "-fPIC", "-o", so_path], sources, flags)
    return run_cmd(cmd, env=env)
end

function main(args)
    opts = parse_args(args)

    # julia/  (this script)  ->  parent is the PaRSEC repo root
    julia_root = dirname(abspath(@__FILE__))
    parsec_repo_root = dirname(julia_root)
    src_dir = joinpath(julia_root, "src")
    build_dir = joinpath(parsec_repo_root, "build")
    install_prefix = get(ENV, "PARSEC_ROOT", joinpath(build_dir, "install"))

    println("\n" * "="^70)
    println("PaRSEC4Julia Build Script (in-tree)")
    println("="^70)
    @printf("Julia dir:     %s\n", julia_root)
    @printf("PaRSEC root:   %s\n", parsec_repo_root)
    @printf("Install to:    %s\n", install_prefix)
    println()

    if !program_exists("cmake")
        println(stderr, "ERROR: cmake not found. Please install cmake.")
        exit(1)
    end
    println("✓ cmake found")

    if opts[:enable_cuda]
        if program_exists("nvcc")
            println("✓ nvcc found (CUDA will be enabled)")
        else
            println(stderr, "WARNING: nvcc not found, CUDA support may not work")
        end
    end

    if !opts[:skip_parsec]
        println("\nStep 1: Building PaRSEC with CMake...")
        mkpath(build_dir)

        cmake_opts = [
            "-DCMAKE_INSTALL_PREFIX=$(install_prefix)",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPARSEC_GPU_WITH_CUDA=$(opts[:enable_cuda] ? "ON" : "OFF")",
            "-DPARSEC_GPU_WITH_HIP=$(opts[:enable_hip] ? "ON" : "OFF")",
            "-DPARSEC_GPU_WITH_OPENCL=$(opts[:enable_opencl] ? "ON" : "OFF")",
            "-DPARSEC_WITH_MPI=ON",
            "-DBUILD_SHARED_LIBS=ON",
        ]

        # Source dir is the parent PaRSEC repo, not a nested submodule.
        cmake_cmd = vcat(["cmake", parsec_repo_root], cmake_opts)
        if !run_cmd(cmake_cmd, cwd=build_dir)
            println(stderr, "ERROR: CMake configuration failed")
            exit(1)
        end
        if !run_cmd(["cmake", "--build", ".", "-j", string(opts[:njobs])], cwd=build_dir)
            println(stderr, "ERROR: CMake build failed")
            exit(1)
        end
        if !run_cmd(["cmake", "--install", "."], cwd=build_dir)
            println(stderr, "ERROR: CMake install failed")
            exit(1)
        end
    else
        println("\nStep 1: Skipping PaRSEC build (--skip-parsec)")
    end

    header = joinpath(install_prefix, "include", "parsec.h")
    if !isfile(header)
        println(stderr, "ERROR: parsec.h not found in $(install_prefix).")
        println(stderr, "       Build PaRSEC first, or set PARSEC_ROOT.")
        exit(1)
    end
    libdir = detect_parsec_libdir(install_prefix)
    parsec_lib_dir = joinpath(install_prefix, libdir)
    if !isdir(parsec_lib_dir)
        println(stderr, "ERROR: PaRSEC library directory not found: $parsec_lib_dir")
        exit(1)
    end
    println("✓ PaRSEC install found at $install_prefix")

    println("\nStep 2: Building Julia wrapper libraries...")

    dtd_wrapper_src = joinpath(src_dir, "dtd_wrapper.c")
    dtd_wrapper_so = joinpath(src_dir, "libdtd_wrapper.so")
    if !isfile(dtd_wrapper_src)
        println(stderr, "ERROR: dtd_wrapper.c not found at $dtd_wrapper_src")
        exit(1)
    end

    build_env = copy(ENV)
    build_env["PARSEC_INSTALL_DIR"] = install_prefix

    include_flags = [
        "-I$(joinpath(install_prefix, "include"))",
        "-I$(src_dir)",
        "-pthread",
    ]
    link_flags = [
        "-L$(parsec_lib_dir)",
        "-Wl,-rpath,$(parsec_lib_dir)",
        "-lparsec", "-lm", "-lpthread",
    ]

    mpi_flags = mpicc_flags()
    for d in mpi_lib_dirs()
        push!(link_flags, "-Wl,-rpath,$d")
    end

    blas_libs = String[]
    cflags_extra = String[]
    if opts[:enable_blas]
        blas_libs = collect_blas_libs()
        if !isempty(blas_libs)
            push!(cflags_extra, "-DHAVE_BLAS=1")
            println("  Using BLAS libraries: $(join(blas_libs, " "))")
        else
            println(stderr, "WARNING: --enable-blas set but no BLAS library found")
        end
    end

    cuda_flags = String[]
    if opts[:enable_cuda]
        cuda_home = get(build_env, "CUDA_HOME", get(ENV, "CUDA_HOME", ""))
        if isempty(cuda_home) && program_exists("nvcc")
            try
                cuda_home = dirname(dirname(readchomp(`which nvcc`)))
            catch
            end
        end
        if !isempty(cuda_home)
            build_env["CUDA_HOME"] = cuda_home
            push!(cuda_flags, "-I$(joinpath(cuda_home, "include"))")
            push!(cuda_flags, "-L$(joinpath(cuda_home, "lib64"))")
            push!(cuda_flags, "-Wl,-rpath,$(joinpath(cuda_home, "lib64"))")
            push!(cuda_flags, "-lcudart")
            println("Using CUDA_HOME=$cuda_home")
        end
    end

    common_flags = vcat(include_flags, cflags_extra, link_flags, mpi_flags, blas_libs, cuda_flags)

    println("Compiling dtd_wrapper.c...")
    if !compile_shared(dtd_wrapper_so, [dtd_wrapper_src], common_flags; env=build_env)
        println(stderr, "ERROR: Failed to compile dtd_wrapper")
        exit(1)
    end
    println("✓ libdtd_wrapper.so: $dtd_wrapper_so")

    stencil_src = joinpath(src_dir, "stencil_wrapper.c")
    stencil_so = joinpath(src_dir, "libstencil_jl.so")
    stencil_internal = joinpath(parsec_repo_root, "tests", "apps", "stencil", "stencil_internal.c")
    stencil_1d = joinpath(build_dir, "tests", "apps", "stencil", "stencil_1D.c")
    if isfile(stencil_src) && isfile(stencil_internal) && isfile(stencil_1d)
        println("Compiling stencil_wrapper.c...")
        stencil_flags = vcat(common_flags, [
            "-I$(joinpath(parsec_repo_root, "tests", "apps", "stencil"))",
            "-I$(joinpath(build_dir, "tests", "apps", "stencil"))",
        ])
        if !compile_shared(stencil_so, [stencil_src, stencil_internal, stencil_1d], stencil_flags; env=build_env)
            println(stderr, "ERROR: Failed to compile stencil wrapper")
            exit(1)
        end
        println("✓ libstencil_jl.so: $stencil_so")
    else
        println("Skipping libstencil_jl.so (need stencil_internal.c and generated stencil_1D.c)")
        if !isfile(stencil_internal)
            println("  missing $stencil_internal")
        end
        if !isfile(stencil_1d)
            println("  missing $stencil_1d (build PaRSEC tests/apps/stencil first)")
        end
    end

    println("\nStep 3: Generating parsec_env.sh...")
    env_script = joinpath(julia_root, "parsec_env.sh")
    env_content = """#!/bin/bash
# PaRSEC4Julia Environment Setup
# Generated by build_parsec4julia.jl

export PARSEC_INSTALL_DIR="$(install_prefix)"
export PARSEC_ROOT="\${PARSEC_INSTALL_DIR}"
export CPATH="\${PARSEC_INSTALL_DIR}/include:\${CPATH}"
export LIBRARY_PATH="\${PARSEC_INSTALL_DIR}/$(libdir):\${LIBRARY_PATH}"
export LD_LIBRARY_PATH="\${PARSEC_INSTALL_DIR}/$(libdir):\${LD_LIBRARY_PATH}"

which mpicc >/dev/null 2>&1 && {
    export CC=mpicc
    export CXX=mpicxx
    export FC=mpifort
}

"""
    for p in mpi_lib_dirs()
        env_content *= "export LIBRARY_PATH=\"$(p):\$LIBRARY_PATH\"\n"
        env_content *= "export LD_LIBRARY_PATH=\"$(p):\$LD_LIBRARY_PATH\"\n"
    end
    if opts[:enable_cuda]
        cuda_home = get(build_env, "CUDA_HOME", "")
        if !isempty(cuda_home)
            env_content *= """
export CUDA_HOME="$(cuda_home)"
export CPATH="\${CUDA_HOME}/include:\${CPATH}"
export LIBRARY_PATH="\${CUDA_HOME}/lib64:\${LIBRARY_PATH}"
export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH}"
"""
        end
    end
    env_content *= """
echo "PaRSEC4Julia environment loaded from $(install_prefix)"
"""
    open(env_script, "w") do f
        write(f, env_content)
    end
    chmod(env_script, 0o755)
    println("✓ parsec_env.sh created: $env_script")

    println("\n" * "="^70)
    println("Build Complete!")
    println("="^70)
    println("\nTo use PaRSEC4Julia:")
    println("  source $env_script")
    println("  julia --project=$(julia_root) examples/dtd_redistribute.jl")
    @printf("\n  PaRSEC install: %s\n", install_prefix)
    @printf("  Wrapper lib:    %s\n", dtd_wrapper_so)
    @printf("  Julia src:      %s\n", src_dir)
    println()
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
