# PaRSEC-Agent: Step 5 - Ollama Integration (Synchronous)

## Purpose
This step replaces the stubbed LLM logic with real calls to a local Ollama server using libcurl.

## Key Concepts
- **libcurl**: Used to perform HTTP POST requests to `http://127.0.0.1:11434/api/generate`.
- **JSON Parsing**: Demonstrates how to parse LLM responses manually using standard C functions (`strstr`).
- **Synchronous Blocking**: In this step, the compute core is blocked while waiting for the LLM response, which is inefficient for HPC.

## How to Build and Run
1. Ensure Ollama is running: `ollama serve` and `ollama pull tinyllama`.
2. Enter the build directory: `mkdir build && cd build`
3. Configure: `cmake ..`
4. Build: `make`
5. Run: `LD_LIBRARY_PATH=$HOME/parsec/builddir/install/lib:$LD_LIBRARY_PATH mpirun -n 1 ./dtd_agent_ollama`
