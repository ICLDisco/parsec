# PaRSEC-Agent: Step 2 - DTD Basics

## Purpose
This step introduces Dynamic Task Discovery (DTD) in PaRSEC. It demonstrates how to define a task class and insert independent tasks into a taskpool.

## Key Concepts
- **parsec_dtd_taskpool_new**: Creates a new taskpool dedicated to DTD.
- **parsec_dtd_create_task_class**: Defines the signature of a task (number of arguments and their types).
- **PARSEC_VALUE**: Indicates that an argument should be passed by value (copied into the task).
- **parsec_dtd_unpack_args**: Extracts the arguments from the task structure inside the task body.
- **PARSEC_DTD_ARG_END**: Sentinel value used to terminate variadic argument lists.

## How to Build and Run
1. Enter the build directory: `mkdir build && cd build`
2. Configure: `cmake ..`
3. Build: `make`
4. Run: `LD_LIBRARY_PATH=$HOME/parsec/builddir/install/lib:$LD_LIBRARY_PATH mpirun -n 1 ./dtd_hello`
