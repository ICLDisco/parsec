# PaRSEC-Agent: Step 7 - Multi-Agent Parallelism

## Purpose
The final step demonstrates the power of PaRSEC by running N independent AI agents in parallel across all available CPU cores.

## Key Concepts
- **Scalability**: All agents share a single DTD taskpool and a single I/O thread.
- **Dynamic Scheduling**: PaRSEC automatically balances the agents' `think`, `poll`, and `tool` tasks across all CPU cores.
- **Core Affinity**: The logs show which core ID is executing each task, proving that the agents are running concurrently.

## How to Build and Run
1. Enter the build directory: `mkdir build && cd build`
2. Configure: `cmake ..`
3. Build: `make`
4. Run for 4 agents: `LD_LIBRARY_PATH=$HOME/parsec/builddir/install/lib:$LD_LIBRARY_PATH mpirun -n 1 ./dtd_multi_agent 4`
