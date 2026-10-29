using Test
using MPI

# Load PaRSEC4Julia modules
include(joinpath(@__DIR__, "..", "setup.jl"))

# Initialize MPI for testing
MPI.Init()

# Include test files
include("test_stencil_1d.jl")
include("test_dgemm_dtd.jl")

# Run all tests
@testset "PaRSEC4Julia Tests" begin
    @testset "Stencil 1D Tests" begin
        @test run_all_tests()
    end
    
    @testset "DGEMM DTD Tests" begin
        @test run_all_dgemm_tests()
    end
end

# Finalize MPI
MPI.Finalize()
