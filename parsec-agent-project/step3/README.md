# PaRSEC-Agent: Step 3 - Recursive Task Insertion

## Purpose
This step demonstrates the core primitive for an agentic loop: the ability for a task to insert its successor while it is executing.

## Key Concepts
- **parsec_dtd_get_taskpool**: Used inside a task body to retrieve the handle to the taskpool so new tasks can be inserted.
- **Recursive insertion**: Task N calls `parsec_dtd_insert_task` for Task N+1 before completing.
- **Thread Safety**: PaRSEC ensures that inserting tasks from within other tasks is thread-safe across all worker cores.

## How to Build and Run
1. Enter the build directory: `mkdir build && cd build`
2. Configure: `cmake ..`
3. Build: `make`
4. Run: `LD_LIBRARY_PATH=$HOME/parsec/builddir/install/lib:$LD_LIBRARY_PATH mpirun -n 1 ./dtd_loop`
