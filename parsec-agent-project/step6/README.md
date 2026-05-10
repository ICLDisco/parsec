# PaRSEC-Agent: Step 6 - Asynchronous I/O Bridge

## Purpose
This step solves the "blocking I/O problem" by moving the LLM call to a background pthread. It uses a "Wait and Poll" pattern to safely re-insert tasks into PaRSEC without crashing the runtime.

## Key Concepts
- **Background I/O Thread**: A dedicated pthread handles all libcurl calls.
- **Polling Task**: Instead of inserting a task directly from an external thread (which can cause segfaults), we insert a "poll" task that checks a status flag in memory.
- **Core Efficiency**: PaRSEC threads are freed immediately after offloading the request, allowing them to perform other computations while the LLM generates a response.

## How to Build and Run
1. Enter the build directory: `mkdir build && cd build`
2. Configure: `cmake ..`
3. Build: `make`
4. Run: `LD_LIBRARY_PATH=$HOME/parsec/builddir/install/lib:$LD_LIBRARY_PATH mpirun -n 1 ./dtd_agent_async`
