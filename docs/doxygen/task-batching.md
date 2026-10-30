<!--
Copyright (c) 2026 NVIDIA Corporation.  All rights reserved.
-->

Task Batching {#task_batching}
==============

Task batching lets a device submit hook combine several compatible ready tasks
into one device operation. The runtime still owns dependency management, data
movement, completion, and release; the submit hook only decides which pending
tasks are compatible with the task it was asked to submit.

Batching is opt-in at the incarnation level and in the device submit hook. A
batch-capable incarnation may call `parsec_gpu_task_collect_batch()` from its
submit hook; a hook that does not call the collector always submits the
singleton task it was given.

Enabling batching
-----------------

For PTG-generated tasks, mark the body with `batch = true` and call the
collector from the device body that can use a batch. The `batch` property marks
that generated incarnation as batch-capable; it does not force the runtime to
batch tasks.

```c
BODY [type=CUDA
      batch = true
      dyld=cublasDgemm dyldtype=cublas_dgemm_t]
{
    int nb_batched = parsec_gpu_task_collect_batch(gpu_stream, gpu_task,
                                                   gemm_batch_match, NULL);
    if( nb_batched < 0 ) {
        return nb_batched;
    }

    /* Submit gpu_task, whose ring may now contain 1 + nb_batched tasks. */
}
```

For DTD tasks, add `PARSEC_DEV_CHORE_ALLOW_BATCH` to the device type when
registering the chore that can batch, then use the same collection approach
inside the registered device hook:

```c
parsec_dtd_task_class_add_chore(tp, tc,
                                PARSEC_DEV_CUDA | PARSEC_DEV_CHORE_ALLOW_BATCH,
                                kernel_cuda);
```

The head task's selected device type must support batching at runtime, and the
selected incarnation of the task passed to the submit hook must be marked
batch-capable. The collector checks these conditions before iterating over
pending work. If either condition fails, it leaves `gpu_task` as a singleton and
returns 0. While scanning the stream, it skips pending tasks whose selected
incarnation is not batch-capable on that same selected device before calling the
user callback.

The MCA parameter `device_enable_batching` defaults to the compile-time batching
capability and can be used to disable batching globally at runtime. It is
read-only when batching support is not compiled in.

A batch-capable task gains from waiting for more compatible tasks. By default,
an execution stream does not submit batch-capable tasks while it still has work
in flight: they could not start earlier anyway, and the tasks arriving in the
meantime can join their batch. Tasks with a lower priority than the next
batch-capable one wait as well, so they cannot keep the stream busy, while
higher-priority tasks that cannot be batched are still submitted. A ring
waiting for its hook to be continued after `PARSEC_HOOK_RETURN_AGAIN` does not
count as work in flight: the hook may be waiting for one of the tasks held back.
An idle execution stream takes over a batch-capable task held back on another
one. The MCA parameter `device_cuda_exec_delay_batch`
(`device_level_zero_exec_delay_batch` for Level Zero) set to 0 submits
batch-capable tasks as soon as there is room on the stream instead.

Recommended collection helper
-----------------------------

The preferred interface for GPU submit hooks is
`parsec_gpu_task_collect_batch()`. The runtime passes the submit hook a
singleton `parsec_gpu_task_t *gpu_task` on its initial invocation. The hook
calls the collector with a callback that decides, for each task currently
pending on the same stream, whether that candidate can be added to the batch
headed by `gpu_task`. After a batched hook returns `PARSEC_HOOK_RETURN_AGAIN`,
the continuation invocation receives the same intact ring. Calling the
collector again is safe: it returns the existing follower count without
scanning or modifying the pending FIFO.

The callback has the type `parsec_gpu_task_batch_cb_t` and receives:

- `candidate`: a pending task (see below for the queues that are scanned);
- `batch_head`: the task originally passed to the submit hook;
- `callback_data`: user data passed through by the caller.

The callback return value controls the iterator:

- `PARSEC_GPU_TASK_BATCH_ACCEPT`: remove `candidate` from the pending FIFO and
  append it to `batch_head`'s task ring;
- `PARSEC_GPU_TASK_BATCH_REJECT`: leave `candidate` pending and continue to the
  next pending task;
- `PARSEC_GPU_TASK_BATCH_STOP`: end collection successfully, leaving the
  current candidate and all unvisited candidates pending.

The collector first scans the pending queues of `gpu_stream`. An execution
stream keeps its batch-capable tasks in `gpu_stream->fifo_batch`, apart from the
other tasks in `gpu_stream->fifo_pending`, and both are scanned. The collector
then scans the batch-capable tasks pending on the other execution streams of the
device: these tasks are done with the stage in, so they can execute on any
execution stream, and taking them builds larger batches.

The collector does not impose a scan or batch-size bound. The submit hook owns
that policy and its callback should return `PARSEC_GPU_TASK_BATCH_STOP` once it
has accepted enough work. The callback executes inside the collector's
traversal of the pending queues, which are private to the GPU manager thread
and therefore walked without locking, so it must remain short and nonblocking.
It must not modify these queues, call the collector recursively, or otherwise
reenter pending-task operations on the device streams.

Example:

```c
static int
gemm_batch_match(parsec_gpu_task_t *candidate,
                 parsec_gpu_task_t *batch_head,
                 void *callback_data)
{
    (void)callback_data;

    if( (batch_head->ec->task_class == candidate->ec->task_class) &&
        (batch_head->ec->selected_chore == candidate->ec->selected_chore) &&
        (batch_head->ec->selected_device == candidate->ec->selected_device) ) {
        return PARSEC_GPU_TASK_BATCH_ACCEPT;
    }
    return PARSEC_GPU_TASK_BATCH_REJECT;
}

int
gemm_kernel_cuda(parsec_device_gpu_module_t *gpu_device,
                 parsec_gpu_task_t *gpu_task,
                 parsec_gpu_exec_stream_t *gpu_stream)
{
    int nb_batched;
    parsec_gpu_task_t *current;

    (void)gpu_device;

    nb_batched = parsec_gpu_task_collect_batch(gpu_stream, gpu_task,
                                               gemm_batch_match, NULL);
    if( nb_batched < 0 ) {
        return nb_batched;
    }

    current = gpu_task;
    do {
        parsec_task_t *task = current->ec;

        /* Submit one device operation for task, or use the whole ring to
         * issue a real batched operation.
         */

        current = (parsec_gpu_task_t *)current->list_item.list_next;
    } while( current != gpu_task );

    return PARSEC_HOOK_RETURN_DONE;
}
```

`parsec_gpu_task_collect_batch()` returns the number of additional tasks
appended to the ring on success, including when the callback stops collection.
A return value of 0 means no task was batched, either because the callback
stopped before accepting one, no compatible pending task was found, batching
is disabled or unsupported by the head task's selected device, or the head
task's selected incarnation is not batch-capable. Any callback value outside
the three actions above returns `PARSEC_HOOK_RETURN_ERROR`. Tasks accepted
before that error remain attached to `gpu_task`; tasks not accepted remain
pending.

The submit hook does not need a completion callback merely to return the ring to
the runtime. If a batched submit hook returns a non-singleton task ring, the GPU
progress engine automatically chains that ring into the next stream's pending
FIFO after the recorded device event completes. Every stream insertion merges
all ring members according to the stream's priority policy; equal-priority
members retain their ring order.

The data movement stages handle batch-capable tasks by rings as well,
independently of the rings formed by submit hooks. The GPU manager moves up to
`PARSEC_DEVICE_MAX_BATCH_SIZE` consecutive tasks that selected the same
batch-capable incarnation from the device queue into the device at once. The
stage-in and stage-out streams then process rings of up to that many pending
tasks sharing an incarnation: the transfers of all members relying on the
default stage functions are merged into batched copies, and the members share a
single stream event. Ownership, pushout, epilog, and completion are still
decided for each task separately, and a member that has to wait (for device
memory, or for a transfer to complete) leaves its ring without holding the
others back.

The submit result settles ownership of the complete ring:

- `PARSEC_HOOK_RETURN_DONE` commits every member to the submitted device work;
- `PARSEC_HOOK_RETURN_AGAIN` commits the ring as continuation state, waits for
  work queued by the hook, and re-enters the hook with the same ring after the
  stream event completes. The hook must have submitted progress for every ring
  member before returning `AGAIN`;
- `PARSEC_HOOK_RETURN_NEXT` immediately restores the followers and applies the
  existing singleton `NEXT` path to the head;
- `PARSEC_HOOK_RETURN_ASYNC` transfers every execution context in the ring to
  the hook, which must eventually complete or reschedule each context. The GPU
  engine releases all corresponding device-task wrappers;
- `PARSEC_HOOK_RETURN_ERROR` and `PARSEC_HOOK_RETURN_DISABLE` quiesce the
  affected execution stream and clean every member of the failed batch before
  propagating the terminal status.

Device-wide recovery after `PARSEC_HOOK_RETURN_DISABLE` is not implemented.
The terminal path settles the batch that observed the error, but it does not
drain other stream FIFOs, event slots, or the device-wide pending queue.

Because `AGAIN` preserves a collected ring, hooks must check predictable
resource constraints before calling `parsec_gpu_task_collect_batch()`. Returning
`AGAIN` before collection is a normal singleton retry; returning it afterward
means that the collected batch has started and must continue as one unit.

The runtime never disbands a ring on its own. A submit hook that no longer
wants to continue with every member calls `parsec_gpu_task_split_batch()`
before returning, which hands the declined members back to the pending queues
of `gpu_stream` under the ordering policy of the stream. If the hook returns
`AGAIN` after a split, only what remains attached to `gpu_task` is retained by
the event. A hook may instead keep
members itself, but then it owns completing those execution contexts and
releasing their wrappers; merely breaking the links leaves them unreachable.

Batch composition and progress
------------------------------

The collector reports the tasks already pending on the stream at the moment it
is called. The runtime does not add tasks to that queue while a submit hook is
running, so the batch a hook receives is the batch it has for that invocation.

A hook must not return `PARSEC_HOOK_RETURN_AGAIN` in order to wait for more
tasks to join the batch. A thread that enters the GPU manager stays there until
the device has no further work, and does not return to the scheduler in the
meantime, so ready tasks held by the scheduler are not drained while it runs.
New work is submitted to the device by other threads. When a single core is
running PaRSEC tasks there is no other thread to submit it, and a hook that
waits for a second task waits indefinitely: one task stays in flight, the
pending queue stays empty, and the hook never completes.

`parsec_init()` derives the number of cores from what hwloc reports for the
process, so a restricted cpuset or a single-core allocation produces this
configuration without it being requested explicitly.

A hook should make progress with the batch it is given, submitting a singleton
when no other task is pending; later invocations batch whatever has
accumulated by then. A hook that requires a minimum batch size must bound its
own wait and fall back to submitting the tasks it already has.

Ownership examples
------------------

### Coroutine-style `AGAIN`

Calling the collector on every coroutine entry is safe. The initial call may
build a batch; continuation calls only report the followers already attached
to the same ring. The application-specific progress routine must submit work
for every member before reporting that another continuation is needed:

```c
int
submit_coroutine_batch(parsec_device_gpu_module_t *gpu_device,
                       parsec_gpu_task_t *gpu_task,
                       parsec_gpu_exec_stream_t *gpu_stream)
{
    int nb_batched;

    (void)gpu_device;
    nb_batched = parsec_gpu_task_collect_batch(gpu_stream, gpu_task,
                                               batch_match, NULL);
    if( nb_batched < 0 ) {
        return nb_batched;
    }

    /* Application code: resume and submit one progress step for every member
     * of gpu_task's ring. The same ring returns here after AGAIN completes.
     */
    return submit_coroutine_step(gpu_task, gpu_stream)
               ? PARSEC_HOOK_RETURN_AGAIN
               : PARSEC_HOOK_RETURN_DONE;
}
```

### Batched `ASYNC`

`ASYNC` transfers every execution context, not the device-task wrappers, to the
hook. Save or enqueue each `ec` before returning; the GPU engine releases the
wrappers after the hook returns. The new owner must eventually complete or
reschedule every saved context:

```c
parsec_gpu_task_t *current = gpu_task;

do {
    parsec_gpu_task_t *next =
        (parsec_gpu_task_t *)current->list_item.list_next;
    parsec_task_t *ec = current->ec;

    PARSEC_LIST_ITEM_SINGLETON(ec);
    async_owner_enqueue(ec);  /* Eventually complete or reschedule ec. */
    current = next;
} while( current != gpu_task );

return PARSEC_HOOK_RETURN_ASYNC;
```

The asynchronous owner must retain the execution contexts, not pointers to
`parsec_gpu_task_t`, because those wrappers become invalid after the return.

### Splitting an `AGAIN` ring

`parsec_gpu_task_split_batch()` is the inverse of the collector. It walks the
members attached to `gpu_task` and asks the callback which ones to keep;
`PARSEC_GPU_TASK_SPLIT_RETURN` sends a member back to the pending queues of
`gpu_stream`, `PARSEC_GPU_TASK_SPLIT_KEEP` leaves it in the ring,
and `PARSEC_GPU_TASK_SPLIT_STOP` ends the walk with the current and all
unvisited members still attached. Call it from the submit hook after the
collector returns, never from the collector callback while the pending FIFO is
being traversed.

```c
static int
drop_when_out_of_workspace(parsec_gpu_task_t *member,
                           parsec_gpu_task_t *batch_head,
                           void *cb_data)
{
    size_t *budget = (size_t *)cb_data;

    (void)batch_head;
    if( *budget < member_footprint(member) ) {
        return PARSEC_GPU_TASK_SPLIT_RETURN;
    }
    *budget -= member_footprint(member);
    return PARSEC_GPU_TASK_SPLIT_KEEP;
}

/* ... in the submit hook, after parsec_gpu_task_collect_batch(): */
nb_returned = parsec_gpu_task_split_batch(gpu_stream, gpu_task,
                                          drop_when_out_of_workspace, &budget);
if( nb_returned < 0 ) {
    return nb_returned;
}
```

`gpu_task` is never visited and always stays attached, so the hook always has a
valid head to return `AGAIN`, `NEXT`, or `DONE` with. The hook must submit
progress for whatever remains in the ring before returning `AGAIN`; the
returned members already have a new owner through the pending queues.

Returned members are still in the device's care, so the manager's outstanding
task count does not change, and each one keeps its open execution-profiling
state. Because they go back on the normal GPU path, those intervals are closed
exactly once at final completion.

Profiling semantics
-------------------

GPU execution profiling is owned by the common device runtime, not by the
submit hook. After a hook first returns `PARSEC_HOOK_RETURN_DONE` or
`PARSEC_HOOK_RETURN_AGAIN`, the runtime emits one task start for every member of
the finalized ring. It emits one matching task end for every member when the
submitted execution finally completes. A task therefore contributes exactly
one start/end pair whether it was submitted alone or as part of a batch.

An `AGAIN` event only resumes the same logical execution. The runtime keeps all
member intervals open across any number of repeated `AGAIN` continuations and
does not emit progress markers for those intermediate steps. A committed
non-singleton ring has already been fully accounted, so repeated continuations
do not rescan every member looking for new profiling starts.

The open profiling state belongs to each device-task wrapper. If a hook
manually disbands an `AGAIN` ring and returns followers to the GPU stream's
pending FIFO, their intervals follow them and close when those wrappers
eventually complete. A hook must not simply release an open detached wrapper,
because that would leave its profiling interval without a matching end.

Iterating over the returned ring
--------------------------------

A batched submit hook should treat `gpu_task` as the head of a circular task
ring. This works for both singleton and batched cases:

```c
parsec_gpu_task_t *current = gpu_task;

do {
    parsec_task_t *task = current->ec;

    /* Use task. */

    current = (parsec_gpu_task_t *)current->list_item.list_next;
} while( current != gpu_task );
```

Original direct collection style
--------------------------------

The helper above is intentionally conservative: it keeps FIFO ownership inside
the device layer and exposes only a compatibility callback to the submit hook.
In very high load scenarios, the repeated callback call can become visible. A
specialized submit hook can still use the original direct style and manipulate
the pending FIFO and task ring itself.

This style is more fragile and should be reserved for code that is already
device-runtime aware. The hook must preserve FIFO correctness and keep rejected
tasks pending. The FIFO is private to the GPU manager thread, so the walk uses
the `nolock` list operations. Execution streams keep the batch-capable tasks in
`fifo_batch`, other streams in `fifo_pending`.

```c
parsec_list_t *pending = (NULL != gpu_stream->fifo_batch) ? gpu_stream->fifo_batch
                                                          : gpu_stream->fifo_pending;
parsec_list_item_t *item;
parsec_list_item_t *next;
int batch_count = 1;

PARSEC_LIST_ITEM_SINGLETON(&gpu_task->list_item);

for(item = (parsec_list_item_t *)pending->ghost_element.list_next;
    item != &pending->ghost_element;
    item = next) {
    parsec_gpu_task_t *candidate;

    next = (parsec_list_item_t *)item->list_next;
    candidate = (parsec_gpu_task_t *)item;

    if( compatible_with_batch(candidate, gpu_task) ) {
        (void)parsec_list_nolock_remove(pending, item);
        (void)parsec_list_item_ring_push(&gpu_task->list_item, item);
        batch_count++;
    }
}
```

The direct style avoids the generic iterator and callback dispatch, and it can
fold the compatibility test into a tight kernel-specific loop. The cost is that
the submit hook now depends on internal list and stream details and must be
updated if the GPU stream internals change.
