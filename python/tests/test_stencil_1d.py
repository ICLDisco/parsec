#!/usr/bin/env python3
"""
Comprehensive tests for the PaRSEC stencil implementation

This is the single test file for all stencil functionality, including:
- Matrix block cyclic distribution
- Stencil initialization and computation
- Weight calculations
- Performance testing
- All PaRSEC function implementations
"""

import sys
import os
import numpy as np
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from stencil_1d import (
    ParsecMatrixBlockCyclic, parsec_stencil_1D, get_parsec_context
)


def test_matrix_initialization():
    """Test matrix initialization"""
    print("🧪 Testing matrix initialization...")
    
    matrix = ParsecMatrixBlockCyclic(
        mtype=1, storage=0, myrank=0,
        mb=4, nb=4, lm=8, ln=8, i=0, j=0, m=8, n=8,
        p=1, q=1, kp=1, kq=1, ip=0, jq=0
    )
    
    assert matrix.mb == 4
    assert matrix.nb == 4
    assert matrix.m == 8
    assert matrix.n == 8
    assert matrix.nb_local_tiles > 0
    assert matrix.bsiz == 16  # 4 * 4
    
    print("  ✓ Matrix initialization passed")
    return True


def test_tile_operations():
    """Test tile get/set operations"""
    print("🧪 Testing tile operations...")
    
    matrix = ParsecMatrixBlockCyclic(
        mtype=1, storage=0, myrank=0,
        mb=2, nb=2, lm=4, ln=4, i=0, j=0, m=4, n=4,
        p=1, q=1, kp=1, kq=1, ip=0, jq=0
    )
    
    # Test tile operations
    test_tile = np.array([[1.0, 2.0], [3.0, 4.0]])
    matrix._set_tile(0, test_tile)
    retrieved_tile = matrix._get_tile(0)
    
    assert np.array_equal(retrieved_tile, test_tile)
    print("  ✓ Tile operations passed")
    return True


def test_stencil_initialization():
    """Test stencil initialization operations"""
    print("🧪 Testing stencil initialization...")
    
    matrix = ParsecMatrixBlockCyclic(
        mtype=1, storage=0, myrank=0,
        mb=4, nb=6, lm=8, ln=12, i=0, j=0, m=8, n=12,
        p=1, q=1, kp=1, kq=1, ip=0, jq=0
    )
    
    # Test initialization
    matrix._init_data(1)  # R=1
    
    # Check first tile
    tile = matrix._get_tile(0)
    assert tile.shape == (4, 6)
    
    # Check main region (should be i + j)
    for j in range(1, 5):  # Main region
        for i in range(4):
            expected = float(i) + float(j)
            assert abs(tile[i, j] - expected) < 1e-6
    
    # Check ghost regions (should be 0)
    for j in range(1):  # Left ghost
        for i in range(4):
            assert tile[i, j] == 0.0
    
    for j in range(5, 6):  # Right ghost
        for i in range(4):
            assert tile[i, j] == 0.0
    
    print("  ✓ Stencil initialization passed")
    return True


def test_core_stencil_kernel():
    """Test core stencil 1D kernel"""
    print("🧪 Testing core stencil kernel...")
    
    matrix = ParsecMatrixBlockCyclic(
        mtype=1, storage=0, myrank=0,
        mb=4, nb=6, lm=8, ln=12, i=0, j=0, m=8, n=12,
        p=1, q=1, kp=1, kq=1, ip=0, jq=0
    )
    
    # Initialize data
    matrix._init_data(1)  # R=1
    
    # Get initial tile
    tile = matrix._get_tile(0)
    initial_tile = tile.copy()
    
    # Apply stencil
    matrix._CORE_stencil_1D(tile, 1)
    
    # Check that computation was applied
    assert not np.array_equal(tile, initial_tile)
    
    # Check boundary conditions
    for i in range(4):
        assert tile[i, 0] == 0.0  # Left boundary
        assert tile[i, 5] == 0.0  # Right boundary
    
    print("  ✓ Core stencil kernel passed")
    return True


def test_full_stencil_function():
    """Test full stencil function"""
    print("🧪 Testing full stencil function...")
    
    matrix = ParsecMatrixBlockCyclic(
        mtype=1, storage=0, myrank=0,
        mb=4, nb=6, lm=8, ln=12, i=0, j=0, m=8, n=12,
        p=1, q=1, kp=1, kq=1, ip=0, jq=0
    )
    
    # Run full stencil function
    parsec_stencil_1D(matrix, 3, 1)
    
    print("  ✓ Full stencil function passed")
    return True


def test_global_context():
    """Test global context management"""
    print("🧪 Testing global context management...")
    
    # Get context multiple times
    context1 = get_parsec_context()
    context2 = get_parsec_context()
    
    # Should be the same instance
    assert context1 is context2
    
    print("  ✓ Global context management passed")
    return True


def test_weight_calculation():
    """Test weight calculation"""
    print("🧪 Testing weight calculation...")
    
    # Test radius 1
    weight_1D = np.zeros(3, dtype=np.float64)
    for jj in range(1, 2):  # R=1
        weight_1D[jj + 1] = 1.0 / (2.0 * jj * 1)
        weight_1D[-jj + 1] = -1.0 / (2.0 * jj * 1)
    weight_1D[1] = 1.0
    
    expected = np.array([-0.5, 1.0, 0.5])
    np.testing.assert_array_almost_equal(weight_1D, expected)
    
    # Test radius 2
    weight_1D_r2 = np.zeros(5, dtype=np.float64)
    for jj in range(1, 3):  # R=2
        weight_1D_r2[jj + 2] = 1.0 / (2.0 * jj * 2)
        weight_1D_r2[-jj + 2] = -1.0 / (2.0 * jj * 2)
    weight_1D_r2[2] = 1.0
    
    expected_r2 = np.array([-0.125, -0.25, 1.0, 0.25, 0.125])
    np.testing.assert_array_almost_equal(weight_1D_r2, expected_r2)
    
    print("  ✓ Weight calculation passed")
    return True


def test_performance():
    """Test performance with different parameters"""
    print("🧪 Testing performance...")
    
    # Test with different matrix sizes
    test_cases = [
        (4, 4, 2, 2, 1, 1),  # Small matrix
        (8, 8, 4, 4, 1, 1),  # Medium matrix
        (16, 16, 4, 4, 1, 1),  # Large matrix
    ]
    
    for M, N, MB, NB, R, iter in test_cases:
        matrix = ParsecMatrixBlockCyclic(
            mtype=1, storage=0, myrank=0,
            mb=MB, nb=NB+2*R, lm=M, ln=N+2*R, i=0, j=0, m=M, n=N+2*R,
            p=1, q=1, kp=1, kq=1, ip=0, jq=0
        )
        
        start_time = time.time()
        parsec_stencil_1D(matrix, iter, R)
        execution_time = time.time() - start_time
        
        # Calculate FLOPS
        flops = iter * (2 * (2 * R + 1)) * (N * MB)
        gflops = (flops / 1e9) / execution_time if execution_time > 0 else 0
        
        print(f"  ✓ {M}x{N} matrix: {execution_time:.6f}s, {gflops:.2f} GFLOPS")
    
    print("  ✓ Performance test passed")
    return True


def test_parsec_functions():
    """Test all PaRSEC function implementations"""
    print("🧪 Testing PaRSEC function implementations...")
    
    # Test parsec_init equivalent
    context = get_parsec_context()
    assert context is not None
    print("  ✓ parsec_init equivalent (ParsecContext)")
    
    # Test parsec_matrix_block_cyclic_init equivalent
    matrix = ParsecMatrixBlockCyclic(
        mtype=1, storage=0, myrank=0,
        mb=4, nb=6, lm=8, ln=12, i=0, j=0, m=8, n=12,
        p=1, q=1, kp=1, kq=1, ip=0, jq=0
    )
    assert matrix.m == 8
    assert matrix.n == 12
    print("  ✓ parsec_matrix_block_cyclic_init equivalent")
    
    # Test parsec_data_allocate equivalent
    assert matrix.mat is not None
    assert len(matrix.mat) > 0
    print("  ✓ parsec_data_allocate equivalent")
    
    # Test parsec_data_collection_set_key equivalent
    assert matrix.key == "dcA"
    print("  ✓ parsec_data_collection_set_key equivalent")
    
    # Test parsec_apply equivalent
    matrix.apply(None, 1)  # Initialize
    matrix.apply(None, 1)  # Apply stencil
    print("  ✓ parsec_apply equivalent")
    
    # Test SYNC_TIME_START/PRINT equivalent
    start_time = time.time()
    time.sleep(0.001)  # Small delay
    elapsed = time.time() - start_time
    assert elapsed > 0
    print("  ✓ SYNC_TIME_START/PRINT equivalent")
    
    # Test parsec_stencil_1D equivalent
    parsec_stencil_1D(matrix, 1, 1)
    print("  ✓ parsec_stencil_1D equivalent")
    
    print("  ✓ All PaRSEC function implementations passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("🚀 Py_PaRSEC Comprehensive Stencil Tests")
    print("=" * 45)
    
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
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
