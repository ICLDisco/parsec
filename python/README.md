# Py_PaRSEC

A Python interface for PaRSEC (Parallel Runtime System for Extreme Scale Computing).

This directory (`python/`) lives inside the PaRSEC source tree and provides
Cython-based Python bindings for the PaRSEC runtime.

## Quick Start

### Single Command Build and Install

From this directory (`python/`):

```bash
# CPU-only installation (builds PaRSEC + installs bindings)
python build_parsec4python.py

# GPU installation with CUDA support
python build_parsec4python.py --enable-cuda

# GPU installation with HIP/ROCm support
python build_parsec4python.py --enable-hip
```

The script automatically:
1. Sets up a virtual environment in `python/venv/`
2. Builds PaRSEC via CMake in `../build/` (the repo root)
3. Installs PaRSEC to `../build/install/`
4. Installs the `py-parsec` Python package
5. Generates `parsec_env.sh` for environment setup

### Manual Installation (Alternative)

If you already have PaRSEC built and installed:

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate

# 2. Point to your PaRSEC installation
export PARSEC_ROOT=/path/to/parsec/install

# 3. Install Py_PaRSEC
pip install -e .
```

### Common Parameter Parser

Both stencil and DTD examples use a common parameter parsing system (`param_parser.py`) that provides:

- **Verbose control** with four levels:
  - `--verbose 0` or `--quiet`: Minimal output
  - `--verbose 1`: Normal output (default)
  - `--verbose 2`: Detailed output
  - `--verbose 10`: Very detailed output (task messages)
- **Unified parameter names** across all examples:
  - `--M`, `--N`, `--K`: Matrix dimensions
  - `--mb`, `--nb`, `--kb`: Block/tile sizes
  - `--device`: Device selection (CPU/GPU)
  - `--cores`: Number of cores

### Run Examples

```bash
source venv/bin/activate
source parsec_env.sh

# Stencil
python examples/stencil_1D.py --M 100 --mb 10 --K 5 --kb 1 --verbose 0

# DTD GEMM
python examples/dtd_simple_gemm.py --M 1024 --mb 128 --device CPU --verbose 0

# Merge sort
python examples/merge_sort.py
```

### Run Tests

```bash
source venv/bin/activate
source parsec_env.sh

pytest
```

## Requirements

- Python 3.8+
- NumPy
- mpi4py
- Cython 3.0+
- MPI library (OpenMPI or MPICH)
- PaRSEC (built from the parent directory)

### GPU Requirements (Optional)
- **CUDA**: CUDA Toolkit 4.0+
- **HIP/ROCm**: ROCm 4.0+

## License

See the top-level LICENSE.txt file for details.
