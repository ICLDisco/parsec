# PaRSEC-Agent Research Framework

This project implements a research framework for mapping AI agents onto the PaRSEC (Parallel Robust Scalable Applications) runtime. It follows a 7-step implementation journey to build a distributed, asynchronous, and parallel agentic system.

## Project Structure
- **step1/**: Build verification and PaRSEC initialization.
- **step2/**: Introduction to Dynamic Task Discovery (DTD).
- **step3/**: Implementation of recursive task insertion for agentic loops.
- **step4/**: Design of the ReAct state machine using tasks.
- **step5/**: Integration with Ollama for real LLM reasoning.
- **step6/**: Implementation of the Asynchronous I/O bridge to prevent core stalling.
- **step7/**: Execution of N parallel agents across multiple CPU cores.

## Prerequisites
- PaRSEC (installed at `$HOME/parsec/builddir/install`)
- OpenMPI
- libcurl
- Ollama (running with the `tinyllama` model)

## Global Build Instructions
Each step is a standalone CMake project. To build any step:
```bash
cd stepX
mkdir build && cd build
cmake ..
make
```

## Running
Ensure your `LD_LIBRARY_PATH` includes the PaRSEC library path:
```bash
export LD_LIBRARY_PATH=$HOME/parsec/builddir/install/lib:$LD_LIBRARY_PATH
```
Then run the specific executable for each step using `mpirun`.
