# PaRSEC4Julia

A Julia interface for PaRSEC (Parallel Runtime System for Extreme Scale Computing).

This directory (`julia/`) lives inside the PaRSEC source tree and provides
C wrappers plus Julia modules for the PaRSEC runtime (DTD, matrices,
redistribute, and the stencil 1D example).

## Quick Start

From this directory (`julia/`):

```bash
# CPU-only: build PaRSEC in ../build, then compile the Julia wrappers
julia build_parsec4julia.jl

# Optional GPU support
julia build_parsec4julia.jl --enable-cuda
julia build_parsec4julia.jl --enable-hip
```

The script:
1. Configures and builds PaRSEC via CMake in `../build/`
2. Installs PaRSEC to `../build/install/`
3. Compiles `src/libdtd_wrapper.so` (and `src/libstencil_jl.so` when stencil sources exist)
4. Writes `parsec_env.sh` for runtime library paths

Alternatively, enable the wrappers from the PaRSEC CMake build:

```bash
cmake -S .. -B ../build -DPARSEC_JULIA_BINDINGS=ON
cmake --build ../build
```

### If PaRSEC is already built

```bash
export PARSEC_ROOT=/path/to/parsec/install   # or use ../build/install
julia build_parsec4julia.jl                  # rebuilds only if needed
source parsec_env.sh
```

## Run examples

```bash
source parsec_env.sh

julia examples/dtd_redistribute.jl
julia examples/ptg_redistribute.jl
julia examples/simple_dtd_gemm.jl
julia examples/stencil_1d.jl 16 16 4 4 20 2
```

On a cluster, launch with the same MPI that PaRSEC was built against, e.g.
`mpirun -np 1 julia examples/dtd_redistribute.jl`.

## Tests

```bash
source parsec_env.sh
julia test/runtests.jl
```

## Layout

```
julia/
├── src/                 # Julia modules + C wrappers
│   ├── dtd_simple.jl    # DTD API (matches py-parsec names)
│   ├── dtd_wrapper.c    # C ABI for ccall
│   ├── stencil_core.jl
│   └── stencil_wrapper.c
├── examples/
├── test/
├── setup.jl
├── Project.toml
└── build_parsec4julia.jl
```

C wrapper include paths for stencil internals are the parent-repo
`tests/apps/stencil` sources (and JDF-generated files under `../build/`),
not a nested `parsec/` submodule.

## Requirements

- Julia 1.6+
- MPI.jl (configured against the same MPI as PaRSEC)
- C compiler and CMake
- MPI library (OpenMPI or MPICH)
- PaRSEC (built from the parent directory)

## License

See the top-level LICENSE.txt file for details.
