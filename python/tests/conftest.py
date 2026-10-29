"""Pytest configuration and fixtures for Py_PaRSEC tests."""

import pytest
import numpy as np
from unittest.mock import Mock, patch


@pytest.fixture
def mock_parsec_context():
    """Mock PaRSEC context for testing."""
    # Create a mock context without patching non-existent C functions
    mock_context = Mock()
    mock_context.nb_cores = 1
    return mock_context


@pytest.fixture
def sample_data():
    """Sample numpy array for testing."""
    return np.random.random((100, 100)).astype(np.float64)


@pytest.fixture
def sample_task_function():
    """Sample task function for testing."""
    def add_arrays(a, b):
        return a + b
    return add_arrays


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


# Skip tests that require MPI if not available
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "mpi: mark test as requiring MPI"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip MPI tests if MPI is not available."""
    try:
        import mpi4py
        mpi_available = True
    except ImportError:
        mpi_available = False
    
    for item in items:
        if "mpi" in item.keywords and not mpi_available:
            item.add_marker(pytest.mark.skip(reason="MPI not available"))
