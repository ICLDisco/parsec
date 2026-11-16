# py_parsec: Python bindings for PaRSEC

Cython bindings for the PaRSEC runtime. They follow the C API closely and
keep its structure: one context, one kind of taskpool for every DSL, data
collections that are not tied to a DSL.

| Python | C |
| --- | --- |
| `py_parsec.Context(nb_cores, args)` | `parsec_init` / `parsec_fini` |
| `ctx.add_taskpool(tp)`, `ctx.start()`, `ctx.wait()` | `parsec_context_add_taskpool`, `parsec_context_start`, `parsec_context_wait` |
| `py_parsec.Taskpool`: `wait()`, `free()` | `parsec_taskpool_t` of any DSL |
| `py_parsec.BlockCyclic(ctx, m, n, mb, nb, dtype, P)` | `parsec_matrix_block_cyclic_t` |
| `py_parsec.dtd.Taskpool()` | `parsec_dtd_taskpool_new` |
| `tp.task_class(name, dtd.Data(...), dtd.Value(...))`, `tc.add_chore(dev, body)` | `parsec_dtd_create_task_class`, `parsec_dtd_task_class_add_chore` |
| `tp.insert_task(tc, dtd.tile_of(A, m, n), 3, ...)` | `parsec_dtd_insert_task_with_task_class` |
| `tp.data_flush_all(A)` | `parsec_dtd_data_flush_all` |
| `py_parsec.matrix.redistribute_new(src, dst, ...)` | `parsec_redistribute_New` (a PTG taskpool) |

DTD and PTG taskpools are scheduled on the same context and can work on the
same matrices (see `examples/redistribute.py`). `Taskpool.from_address()`
adopts a `parsec_taskpool_t *` built by C code, for instance a JDF compiled
into a separate library, and `Context.address` / `TiledMatrix.address` hand
the context and the matrices to such code.

## Example

```python
import numpy as np
import py_parsec
from py_parsec import dtd

with py_parsec.Context() as ctx:
    A = py_parsec.BlockCyclic(ctx, 1000, 1000, 100, 100, name="A")
    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)
    ctx.start()

    scale = tp.task_class("scale", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), dtd.Value("d"))

    def body(tile, alpha):      # tile: (100, 100) column-major view of the PaRSEC data
        tile *= alpha

    scale.add_chore(py_parsec.DEV_CPU, body)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(scale, dtd.tile_of(A, m, n), 2.0)
    tp.data_flush_all(A)
    tp.wait()

    for m, n, tile in A.local_tiles():   # tiles owned by this process
        ...
```

Run it on one process or under `mpiexec`. Task bodies are Python callables
(CPU) or C function addresses (any device; an int, a PyCapsule or a ctypes
function pointer with the DTD chore signature). An exception raised by a
body is reported by the next `wait`.

## Performance

- **One BLAS thread per task.** Each task is one core's work, but NumPy's
  BLAS (OpenBLAS, MKL) starts its own threads on every call by default, so
  every worker oversubscribes the cores PaRSEC already uses. Set
  `OPENBLAS_NUM_THREADS=1` (and `OMP_NUM_THREADS`, `MKL_NUM_THREADS`) before
  NumPy is imported, as `examples/dtd_gemm.py` does.
- **Enough work per task.** Inserting a task from Python costs about 1 µs
  more than from C. A Python body runs with the GIL, which NumPy releases in
  its heavy calls: one worker pays about 1 µs per task, but many workers of
  one process queue for the GIL and pay tens of µs. Give Python bodies about
  a millisecond of work or more.
- **Fewer threads per process for small tasks.** Each process has its own
  GIL, so several ranks per node with a few cores each scale better than one
  rank with many cores when tasks are short.
- **C bodies for fine-grained tasks.** A C function address runs without the
  GIL.

Tiled GEMM C += A B, 8192 x 8192 doubles, 16 cores of a Xeon Gold 6140,
NumPy bodies with one BLAS thread, compared with the same tile kernel running
on the 16 cores without PaRSEC:

| tile (time per task) | 1 rank x 16 cores | 4 ranks x 4 cores |
| --- | --- | --- |
| 512 (6.8 ms) | 400 GFlop/s, 91% | 340 GFlop/s, 77% |
| 256 (0.9 ms) | 307 GFlop/s, 65% | 325 GFlop/s, 69% |
| 128 (0.16 ms) | 54 GFlop/s, 17% | 189 GFlop/s, 58% |

With BLAS left multi-threaded, the 256 case drops to 130-180 GFlop/s on one
rank and 43-95 GFlop/s on four.

## Build

With the PaRSEC build: configure PaRSEC with `-DPARSEC_PYTHON_BINDINGS=ON`
(and `-DPython_EXECUTABLE=...` to pick the interpreter, which needs Cython 3
and NumPy). The bindings are built in `<build>/python/python.test`, tested by
`ctest -R python`, and installed under the PaRSEC prefix by `make install`
(`<prefix>/lib/pythonX.Y/site-packages`).

Against an installed PaRSEC:

```bash
PARSEC_ROOT=/path/to/parsec/install pip install ./python
```

`setup.py` reads the include directories from `pkg-config parsec` (looking
in `$PARSEC_ROOT/lib*/pkgconfig` first) and links the extension modules
with `libparsec`, with a run path to it.

## Tests

```bash
pip install pytest
python -m pytest python/tests
mpiexec -n 4 python -m pytest python/tests
```
