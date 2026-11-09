# PaRSEC-Agent: Step 4 - Single Synchronous Agent Loop

## Purpose
This step implements the ReAct (Reason + Act) loop logic. It uses a state object (`agent_state_t`) that is passed between tasks to maintain progress.

## Key Concepts
- **agent_state_t**: A heap-allocated structure containing the agent's memory (thoughts, actions, and step count).
- **think_task**: Analyzes current state and decides on a tool or a final answer.
- **tool_task**: Executes a tool (stubbed) and returns control to the thinking phase.
- **finish_task**: Finalizes the agent run and frees resources.

## How to Build and Run
1. Enter the build directory: `mkdir build && cd build`
2. Configure: `cmake ..`
3. Build: `make`
4. Run: `LD_LIBRARY_PATH=$HOME/parsec/builddir/install/lib:$LD_LIBRARY_PATH mpirun -n 1 ./dtd_agent`
