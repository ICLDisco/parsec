"""
Setup script for PaRSEC4Julia examples and tests
This file handles all the path setup and module loading
"""

# Add src to load path
src_dir = joinpath(@__DIR__, "src")
push!(LOAD_PATH, src_dir)

# Core top-level sources
include(joinpath(src_dir, "parsec_c_wrapper.jl"))
include(joinpath(src_dir, "types.jl"))
include(joinpath(src_dir, "context.jl"))
include(joinpath(src_dir, "matrix.jl"))
include(joinpath(src_dir, "stencil.jl"))
include(joinpath(src_dir, "utils.jl"))

# Module-style sources
include(joinpath(src_dir, "dtd_simple.jl"))
include(joinpath(src_dir, "stencil_core.jl"))
include(joinpath(src_dir, "PaRSEC4Julia.jl"))


println("✓ PaRSEC4Julia modules loaded")
