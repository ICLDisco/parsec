# PaRSEC-Agent: Step 1 - Build Verification

## Purpose
The goal of this step is to verify that the PaRSEC installation is correct and that the build system (CMake) can correctly link against PaRSEC and MPI.

## Key Concepts
- **parsec_init**: Initializes the PaRSEC runtime environment.
- **parsec_fini**: Safely shuts down the PaRSEC context.
- **MPI_Init_thread**: Required because PaRSEC is a multi-threaded runtime and needs MPI to support the `MPI_THREAD_MULTIPLE` level.

## How to Build and Run
1. Enter the build directory: `mkdir build && cd build`
2. Configure: `cmake ..`
3. Build: `make`
4. Run: `LD_LIBRARY_PATH=$HOME/parsec/builddir/install/lib:$LD_LIBRARY_PATH mpirun -n 1 ./parsec_ok`
