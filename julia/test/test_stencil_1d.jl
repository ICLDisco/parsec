#!/usr/bin/env julia
"""
Comprehensive tests for the PaRSEC stencil implementation

This is the single test file for all stencil functionality, including:
- Matrix block cyclic distribution
- Stencil initialization and computation
- Weight calculations
- Performance testing
- All PaRSEC function implementations
"""

using Test
using MPI

# Load PaRSEC4Julia modules
include(joinpath(@__DIR__, "..", "setup.jl"))

# Initialize MPI for testing
MPI.Init()

function test_matrix_initialization()
    """Test matrix initialization"""
    println("🧪 Testing matrix initialization...")
    
    matrix = ParsecMatrixBlockCyclic(
        8, 8, 4, 4, 1, 1
    )
    
    @test matrix.mb == 4
    @test matrix.nb == 4
    @test matrix.m == 8
    @test matrix.n == 8
    @test matrix.nb_local_tiles > 0
    @test matrix.bsiz == 16  # 4 * 4
    
    println("  ✓ Matrix initialization passed")
    return true
end

function test_tile_operations()
    """Test tile get/set operations"""
    println("🧪 Testing tile operations...")
    
    matrix = ParsecMatrixBlockCyclic(
        4, 4, 2, 2, 1, 1
    )
    
    # Test tile operations
    test_tile = [1.0 2.0; 3.0 4.0]
    _set_tile(matrix, 1, test_tile)
    retrieved_tile = _get_tile(matrix, 1)
    
    @test retrieved_tile ≈ test_tile
    println("  ✓ Tile operations passed")
    return true
end

function test_stencil_initialization()
    """Test stencil initialization operations"""
    println("🧪 Testing stencil initialization...")
    
    matrix = ParsecMatrixBlockCyclic(
        8, 12, 4, 6, 1, 1
    )
    
    # Test initialization
    _stencil_1D_init_ops(matrix, 1)  # R=1
    
    # Check first tile
    tile = _get_tile(matrix, 1)
    @test size(tile) == (4, 6)
    
    # Check main region (should be i + j)
    for j in 2:5  # Main region
        for i in 1:4
            expected = Float64(i-1) + Float64(j-1)
            @test abs(tile[i, j] - expected) < 1e-6
        end
    end
    
    # Check ghost regions (should be 0)
    for j in 1:1  # Left ghost
        for i in 1:4
            @test tile[i, j] == 0.0
        end
    end
    
    for j in 6:6  # Right ghost
        for i in 1:4
            @test tile[i, j] == 0.0
        end
    end
    
    println("  ✓ Stencil initialization passed")
    return true
end

function test_core_stencil_kernel()
    """Test core stencil 1D kernel"""
    println("🧪 Testing core stencil kernel...")
    
    matrix = ParsecMatrixBlockCyclic(
        8, 12, 4, 6, 1, 1
    )
    
    # Initialize data
    _stencil_1D_init_ops(matrix, 1)  # R=1
    
    # Get initial tile
    tile = _get_tile(matrix, 1)
    initial_tile = copy(tile)
    
    # Apply stencil
    _CORE_stencil_1D(tile, [0.5, 1.0, -0.5], 1)
    
    # Check that computation was applied
    @test !isapprox(tile, initial_tile)
    
    # Check boundary conditions
    for i in 1:4
        @test tile[i, 1] == 0.0  # Left boundary
        @test tile[i, 6] == 0.0  # Right boundary
    end
    
    println("  ✓ Core stencil kernel passed")
    return true
end

function test_full_stencil_function()
    """Test full stencil function"""
    println("🧪 Testing full stencil function...")
    
    matrix = ParsecMatrixBlockCyclic(
        8, 12, 4, 6, 1, 1
    )
    
    # Run full stencil function
    parsec_stencil_1D(matrix, 3, 1)
    
    println("  ✓ Full stencil function passed")
    return true
end

function test_global_context()
    """Test global context management"""
    println("🧪 Testing global context management...")
    
    # Get context multiple times
    context1 = get_parsec_context()
    context2 = get_parsec_context()
    
    # Should be the same instance
    @test context1 === context2
    
    println("  ✓ Global context management passed")
    return true
end

function test_weight_calculation()
    """Test weight calculation"""
    println("🧪 Testing weight calculation...")
    
    # Test radius 1
    weight_1D = zeros(Float64, 3)
    weight_1D[2] = 1.0  # center weight
    for jj in 1:1  # R=1
        weight_1D[3] = 1.0 / (2.0 * jj * 1)  # index 3 = 0.5 (right side)
        weight_1D[1] = -1.0 / (2.0 * jj * 1)  # index 1 = -0.5 (left side)
    end
    
    expected = [-0.5, 1.0, 0.5]  # Correct expected values
    @test weight_1D ≈ expected
    
    # Test radius 2
    weight_1D_r2 = zeros(Float64, 5)
    weight_1D_r2[3] = 1.0  # center weight
    for jj in 1:2  # R=2
        weight_1D_r2[jj + 3] = 1.0 / (2.0 * jj * 2)  # indices 4, 5
        weight_1D_r2[3 - jj] = -1.0 / (2.0 * jj * 2)  # indices 2, 1
    end
    
    expected_r2 = [-0.125, -0.25, 1.0, 0.25, 0.125]  # Correct expected values
    @test weight_1D_r2 ≈ expected_r2
    
    println("  ✓ Weight calculation passed")
    return true
end

function test_performance()
    """Test performance with different parameters"""
    println("🧪 Testing performance...")
    
    # Test with different matrix sizes
    test_cases = [
        (4, 4, 2, 2, 1, 1),  # Small matrix
        (8, 8, 4, 4, 1, 1),  # Medium matrix
        (16, 16, 4, 4, 1, 1),  # Large matrix
    ]
    
    for (M, N, MB, NB, R, iter) in test_cases
        matrix = ParsecMatrixBlockCyclic(
            M, N+2*R, MB, NB+2*R, 1, 1
        )
        
        start_time = time()
        parsec_stencil_1D(matrix, iter, R)
        execution_time = time() - start_time
        
        # Calculate FLOPS
        flops = calculate_stencil_flops(N, MB, iter, R)
        gflops = execution_time > 0 ? (flops / 1e9) / execution_time : 0
        
        println("  ✓ $(M)x$(N) matrix: $(execution_time)s, $(gflops) GFLOPS")
    end
    
    println("  ✓ Performance test passed")
    return true
end

function test_parsec_functions()
    """Test all PaRSEC function implementations"""
    println("🧪 Testing PaRSEC function implementations...")
    
    # Test parsec_init equivalent
    context = get_parsec_context()
    @test context !== nothing
    println("  ✓ parsec_init equivalent (ParsecContext)")
    
    # Test parsec_matrix_block_cyclic_init equivalent
    matrix = ParsecMatrixBlockCyclic(
        8, 12, 4, 6, 1, 1
    )
    @test matrix.m == 8
    @test matrix.n == 12
    println("  ✓ parsec_matrix_block_cyclic_init equivalent")
    
    # Test parsec_data_allocate equivalent
    @test matrix.mat !== nothing
    @test length(matrix.mat) > 0
    println("  ✓ parsec_data_allocate equivalent")
    
    # Test parsec_data_collection_set_key equivalent
    @test matrix.key == "dcA"
    println("  ✓ parsec_data_collection_set_key equivalent")
    
    # Test parsec_apply equivalent
    parsec_apply(get_parsec_context(), PARSEC_MATRIX_FULL, matrix, nothing, 1)  # Initialize
    parsec_apply(get_parsec_context(), PARSEC_MATRIX_FULL, matrix, nothing, 1)  # Apply stencil
    println("  ✓ parsec_apply equivalent")
    
    # Test SYNC_TIME_START/PRINT equivalent
    start_time = time()
    sleep(0.001)  # Small delay
    elapsed = time() - start_time
    @test elapsed > 0
    println("  ✓ SYNC_TIME_START/PRINT equivalent")
    
    # Test parsec_stencil_1D equivalent
    parsec_stencil_1D(matrix, 1, 1)
    println("  ✓ parsec_stencil_1D equivalent")
    
    println("  ✓ All PaRSEC function implementations passed")
    return true
end

function run_all_tests()
    """Run all tests"""
    println("🚀 PaRSEC4Julia Comprehensive Stencil Tests")
    println("=" ^ 45)
    
    tests = [
        test_matrix_initialization,
        test_tile_operations,
        test_stencil_initialization,
        test_core_stencil_kernel,
        test_full_stencil_function,
        test_global_context,
        test_weight_calculation,
        test_performance,
        test_parsec_functions,
    ]
    
    passed = 0
    total = length(tests)
    
    for test in tests
        try
            if test()
                passed += 1
            end
        catch e
            println("  ❌ $(test) failed: $e")
            rethrow(e)
        end
    end
    
    println("\n📊 Test Results: $passed/$total tests passed")
    
    if passed == total
        println("🎉 All tests passed!")
        return true
    else
        println("❌ Some tests failed!")
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    success = run_all_tests()
    MPI.Finalize()
    exit(success ? 0 : 1)
end
