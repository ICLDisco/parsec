<!--
Copyright (c) 2026 NVIDIA Corporation.  All rights reserved.
-->

Dynamic Task Discovery Model {#dtd_model}
=========================================

Dynamic Task Discovery (DTD) is the PaRSEC interface for applications that
discover work at runtime instead of describing the full task graph in a JDF
file. A DTD application creates a PaRSEC context, creates one or more DTD
taskpools, describes the data handles used by tasks, inserts tasks, and lets the
runtime derive the dependencies from the order and access mode of the inserted
data flows.

This page is an application-oriented guide. It complements the API reference in
@ref DTD_INTERFACE and uses the tests in `tests/dsl/dtd` as executable examples.

DTD Mental Model
----------------

A DTD task has a task class, one or more device incarnations called chores, and
a list of parameters. Parameters are either values, references, scratch buffers,
or tracked PaRSEC data tiles. Dependencies are inferred only from tracked data
tiles:

- `PARSEC_INPUT` reads a tile and depends on the last writer.
- `PARSEC_OUTPUT` writes a tile without reading the previous value.
- `PARSEC_INOUT` reads and writes a tile, and therefore serializes with prior
  reads and writes.
- `PARSEC_DONT_TRACK` disables DTD dependency tracking for that tile parameter.
  The application is then responsible for all ordering.

The insertion order is meaningful. For each DTD tile, PaRSEC records the
sequence of readers and writers and releases successors when their predecessors
complete. In distributed memory, the selected task rank and the current owner of
the latest tile version also drive communication.

Minimal Lifecycle
-----------------

A small DTD application usually follows this shape:

```c
#if defined(PARSEC_HAVE_MPI)
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
#endif

parsec_context_t *parsec = parsec_init(cores, &argc, &argv);
parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

/* Describe the datatype carried by a tile region. */
int TILE_FULL;
parsec_arena_datatype_t *adt =
    parsec_matrix_adt_new_rect(parsec_datatype_int32_t, nb, 1, nb);
parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);

/* Create or initialize a data collection, then expose it to DTD. */
parsec_data_collection_t *A = ...;
parsec_dtd_data_collection_init(A);

parsec_context_add_taskpool(parsec, tp);
parsec_context_start(parsec);

/* Insert tasks here. */

parsec_dtd_data_flush_all(tp, A);
parsec_taskpool_wait(tp);
parsec_context_wait(parsec);

parsec_taskpool_free(tp);
parsec_dtd_data_collection_fini(A);

parsec_dtd_free_arena_datatype(parsec, TILE_FULL);
PARSEC_OBJ_RELEASE(adt->arena);
parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);
parsec_fini(&parsec);

#if defined(PARSEC_HAVE_MPI)
MPI_Finalize();
#endif
```

The taskpool must be added to a context before tasks are inserted. Tests use
both common orders:

- Add the taskpool, start the context, insert tasks.
- Start the context once, add a new taskpool, insert tasks.

The first form is the simplest for batch programs. The second form is useful
when one PaRSEC context stays alive while the application creates several DTD
taskpools. In optimized applications, associate the taskpool with the context
before adding chores so device validation sees the devices enabled for that
taskpool.

Data Collections, Tiles, and Arenas
-----------------------------------

DTD tracks dependencies on `parsec_dtd_tile_t` handles. For a normal PaRSEC
data collection, initialize the collection before using it with DTD:

```c
parsec_data_collection_set_key(A, "A");
parsec_dtd_data_collection_init(A);
```

Then obtain DTD tiles by key:

```c
parsec_dtd_tile_t *tile = PARSEC_DTD_TILE_OF_KEY(A, key);
```

or, for two-dimensional matrix data collections:

```c
parsec_dtd_tile_t *tile = PARSEC_DTD_TILE_OF(A, i, j);
```

The lower 16 bits of the DTD operation flags identify the arena or region used
for a data flow. The common pattern is to create one full-tile arena id and OR
that id into the flow flags:

```c
PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key),
PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY
```

Register an arena before the runtime needs to allocate, move, or describe that
region. In distributed memory, missing arena metadata is a fatal application
error because PaRSEC cannot describe the remote data movement correctly.

Temporary DTD-only tiles can be created with:

```c
parsec_dtd_tile_t *tmp = parsec_dtd_tile_new(tp, rank);
```

`rank` is the owner rank for the new tile. The `dtd_test_new_tile` test shows
how to create temporary tiles, use them through several tasks, flush them, and
release the task classes that operate on them.

Parameter Kinds
---------------

The direct insertion API describes each parameter as a triplet:

```c
sizeof(int), &alpha, PARSEC_VALUE,
sizeof(void *), user_ptr, PARSEC_REF,
scratch_bytes, NULL, PARSEC_SCRATCH,
PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key), PARSEC_INPUT | TILE_FULL,
PARSEC_DTD_ARG_END
```

`PARSEC_VALUE` copies bytes at insertion time. Use it for scalars and small
plain data structures. `PARSEC_REF` passes a pointer value without creating a
data dependency. `PARSEC_SCRATCH` asks the runtime to allocate temporary task
storage. `PASSED_BY_REF` with a DTD tile creates a tracked data flow.

Task bodies recover parameters in insertion order:

```c
parsec_hook_return_t
scale_cpu(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    double *A;
    double alpha;

    (void)es;
    parsec_dtd_unpack_args(this_task, &A, &alpha);

    /* compute */
    return PARSEC_HOOK_RETURN_DONE;
}
```

The unpack order must match the insertion or task-class parameter order. A task
may unpack a prefix of the parameter list by passing `0` as the final unpack
argument.

Direct Insertion
----------------

`parsec_dtd_insert_task()` is the most compact API:

```c
parsec_dtd_insert_task(tp, scale_cpu, 0, PARSEC_DEV_CPU, "scale",
                       PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key),
                       PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                       sizeof(double), &alpha, PARSEC_VALUE,
                       PARSEC_DTD_ARG_END);
```

For direct insertion, the device type must be exactly one concrete device type
supported by the taskpool, for example `PARSEC_DEV_CPU` or `PARSEC_DEV_CUDA`.
Device masks are not accepted here.

Direct insertion is useful for simple programs, control tasks, and tests. It is
not the recommended hot path for optimized applications because the runtime has
to process the varargs, discover or reuse an internal task class, and set up
metadata on the insertion path. Performance-sensitive applications should use
explicit task classes.

Explicit Task Classes
---------------------

An explicit task class separates task metadata from task instances. Once the
taskpool has been associated with a context, create the class once, add concrete
chores, then insert many tasks from it:

```c
parsec_task_class_t *scale_tc =
    parsec_dtd_create_task_class(tp, "scale",
                                 PASSED_BY_REF,
                                 PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY,
                                 sizeof(double), PARSEC_VALUE,
                                 PARSEC_DTD_ARG_END);

parsec_dtd_task_class_add_chore(tp, scale_tc,
                                PARSEC_DEV_CPU, scale_cpu);
```

Each chore must use one concrete CPU/GPU device type that is available in the
taskpool. A chore is a device-specific implementation of the same task class.
For CUDA:

```c
parsec_dtd_task_class_add_chore(tp, scale_tc,
                                PARSEC_DEV_CUDA, scale_cuda);
```

Once the chores are registered, insert task instances with:

```c
parsec_dtd_insert_task_with_task_class(tp, scale_tc, priority, PARSEC_DEV_ALL,
                                       PARSEC_DTD_EMPTY_FLAG,
                                       PARSEC_DTD_TILE_OF_KEY(A, key),
                                       PARSEC_DTD_EMPTY_FLAG, &alpha,
                                       PARSEC_DTD_ARG_END);
```

For `parsec_dtd_insert_task_with_task_class()`, `device_type` may be a mask such
as `PARSEC_DEV_ALL`. The mask selects among the already registered concrete
chores of the task class. A mask that selects no registered chore is an
application error.

Per-instance flags in the insertion call are normally `PARSEC_DTD_EMPTY_FLAG`.
Use non-empty flags when the instance needs to add information such as
`PARSEC_AFFINITY`, `PARSEC_PUSHOUT`, or `PARSEC_PULLIN` that is not already
fully encoded in the class parameter description.

Release explicit task classes after all tasks using them have completed:

```c
parsec_taskpool_wait(tp);
parsec_dtd_task_class_release(tp, scale_tc);
```

CPU and CUDA Incarnations
-------------------------

CPU chores use the DTD CPU function signature:

```c
parsec_hook_return_t
scale_cpu(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    double *A;
    double alpha;

    (void)es;
    parsec_dtd_unpack_args(this_task, &A, &alpha);
    return PARSEC_HOOK_RETURN_DONE;
}
```

CUDA chores use the GPU task wrapper types and retrieve device pointers by flow
index:

```c
int
scale_cuda(parsec_device_gpu_module_t *gpu_device,
           parsec_gpu_task_t *gpu_task,
           parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    double *A_host;
    double alpha;
    double *A_dev;

    (void)gpu_device;
    parsec_dtd_unpack_args(this_task, &A_host, &alpha);

    A_dev = parsec_dtd_get_dev_ptr(this_task, 0);
    /* launch CUDA work on gpu_stream */

    return PARSEC_HOOK_RETURN_DONE;
}
```

The host pointer recovered by `parsec_dtd_unpack_args()` identifies the logical
tile. Use `parsec_dtd_get_dev_ptr(this_task, flow_index)` for the actual device
copy associated with a data flow. `dtd_test_simple_gemm` is the best complete
CPU plus CUDA example: it registers a CPU BLAS implementation and a CUDA/CUBLAS
implementation on the same explicit `GEMM` task class.

Use `PARSEC_PUSHOUT` when a GPU result must be pushed back to the host or to the
distributed owner at the end of the flow. Use `PARSEC_PULLIN` when an instance
needs a CPU-side copy after GPU execution. Avoid unnecessary push and pull flags
on intermediate GPU-only pipelines; they can force avoidable transfers.

Placement and Distributed Memory
--------------------------------

In distributed runs, each task must have a well-defined execution rank. The most
common placement mechanism is `PARSEC_AFFINITY` on a data flow:

```c
PASSED_BY_REF, PARSEC_DTD_TILE_OF_KEY(A, key),
PARSEC_INOUT | TILE_FULL | PARSEC_AFFINITY
```

The task is placed on the rank that owns that tile. `PARSEC_AFFINITY` can also
be placed on a `PARSEC_VALUE` rank parameter:

```c
sizeof(int), &rank, PARSEC_VALUE | PARSEC_AFFINITY
```

If the value rank is outside the communicator range, DTD normalizes it modulo
the communicator size and prints a warning. Prefer passing valid ranks so the
warning remains a diagnostic for real mistakes.

Flush data collections before waiting for a final distributed result:

```c
parsec_dtd_data_flush(tp, PARSEC_DTD_TILE_OF_KEY(A, key));
parsec_dtd_data_flush_all(tp, A);
```

Flushes return the latest version to the owning rank and let the runtime reclaim
DTD metadata for the tile. A flush is itself a tile operation. Do not insert new
tasks on a tile after flushing it until the taskpool has been waited.

Context Usage Modes
-------------------

DTD supports several application styles.

Batch mode: create a taskpool, add it to the context, start the context, insert
all tasks, flush final data, wait, and free the taskpool. Most tests use this
mode.

Streaming or windowed mode: start the context once, add a DTD taskpool, insert a
window of tasks, call `parsec_taskpool_wait()` to throttle or observe progress,
then continue with another taskpool or another insertion phase. The DTD window
parameters `parsec_dtd_window_size` and `parsec_dtd_threshold_size` control how
long the producer keeps inserting before helping the workers.

Explicit task object mode: create a task with `parsec_dtd_create_task()`, keep
the returned `parsec_task_t *` while adding any application-side bookkeeping,
then submit it with `parsec_insert_dtd_task()`. See
`dtd_test_explicit_task_creation`.

Task-generating mode: a DTD task may insert more DTD tasks. If the task is not
logically complete, return `PARSEC_HOOK_RETURN_AGAIN` to let PaRSEC reschedule
it later. See `dtd_test_task_insertion`, `dtd_test_task_inserting_task`, and
`dtd_test_hierarchy`.

Asynchronous mode: advanced GPU or external-runtime tasks may return
`PARSEC_HOOK_RETURN_ASYNC` and later re-enable completion through the runtime
callback path. This is specialized and should be copied from a working test,
such as `dtd_test_cuda_again_async` or `dtd_test_tp_enqueue_dequeue`, rather
than written from memory.

Concurrency Contract
--------------------

DTD allows multiple producer threads to insert tasks into one taskpool, but the
application owns part of the synchronization contract.

Create and configure task classes serially. This includes
`parsec_dtd_create_task_class()`, `parsec_dtd_task_class_add_chore()`,
direct-insert first use that auto-creates a task class, and
`parsec_dtd_task_class_release()`.

Create or look up DTD tile handles serially per data collection while the DTD
tile table for that data collection can still change. This includes
`parsec_dtd_tile_of()`, `PARSEC_DTD_TILE_OF()`, and
`PARSEC_DTD_TILE_OF_KEY()` for handles that might not already exist.

After task classes and tile handles are created, parallel producers may insert
tasks only over disjoint DTD tile sets, or under application synchronization for
every operation that can touch the same DTD tile. Flushes must be serialized
with insertions that touch the same tile.

Performance Checklist
---------------------

- Use explicit task classes in hot loops. Direct insertion is for low-volume
  tasks, control logic, and prototyping.
- Register CPU and GPU chores once, using concrete device types, then insert
  with a mask only when selecting among registered chores.
- Pre-create tile handles before multi-producer insertion.
- Use `PARSEC_VALUE` for small immutable scalars; use `PARSEC_REF` only when
  the referenced object remains valid until the task executes.
- Add `PARSEC_AFFINITY` on a stable placement flow, especially in distributed
  memory. For write tasks, the affinity should match the intended data owner or
  producer location.
- Register arena datatypes for every region used by tracked data flows.
- Flush only when the application needs final ownership or intends to stop
  using the tile in the current taskpool.
- Avoid `PARSEC_DONT_TRACK` unless the application has an independent ordering
  mechanism for the tile.
- Minimize forced GPU host transfers. Use `PARSEC_PUSHOUT` and
  `PARSEC_PULLIN` when the data movement is semantically needed, not as default
  decoration.
- Release explicit task classes after the taskpool has finished all tasks that
  use them.

Test Map
--------

The DTD tests are useful examples and regression anchors:

| Test | What it demonstrates |
| ---- | -------------------- |
| `dtd_test_insert_task_interface` | Direct insertion, value parameters, references, and basic data flows. |
| `dtd_test_explicit_task_creation` | Creating a task object before inserting it. |
| `dtd_test_simple_gemm` | Explicit task classes with CPU and CUDA chores, arena registration, GPU device pointers, and flush-all. |
| `dtd_test_cuda_task_insert` | Device masks on explicit task-class insertion and CUDA-only chores. |
| `dtd_test_new_tile` | DTD-created temporary tiles and CPU/GPU chores over those tiles. |
| `dtd_test_data_flush` | Single-tile and full-collection flush behavior. |
| `dtd_test_broadcast`, `dtd_test_reduce`, `dtd_test_allreduce` | Distributed collective-style dependency patterns. |
| `dtd_test_pingpong`, `dtd_test_war` | Cross-rank ordering and write-after-read behavior. |
| `dtd_test_task_insertion`, `dtd_test_task_inserting_task`, `dtd_test_hierarchy` | Tasks that insert tasks and use `PARSEC_HOOK_RETURN_AGAIN`. |
| `dtd_test_tp_enqueue_dequeue` | Dynamic taskpool enqueue/dequeue and asynchronous completion patterns. |
| `dtd_test_multiple_handle_wait` | Repeated waits on DTD taskpools. |
| `dtd_test_flag_dont_track` | Explicitly disabling dependency tracking for a parameter. |
| `dtd_test_null_as_tile` | Passing NULL-like tile handles through the DTD interface. |
| `dtd_test_task_placement` | Rank affinity from value and tile parameters. |

When writing a new DTD application, start with the direct insertion tests to
understand the argument protocol, then switch to the explicit task-class tests
before writing performance-sensitive code.
