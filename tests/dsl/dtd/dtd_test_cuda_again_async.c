/*
 * Copyright (c) 2023      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec.h"
#include "parsec/arena.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "tests/tests_data.h"

#include <limits.h>

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

/* Update this test's private PARSEC_VALUE arguments in place to model a
 * coroutine continuation. DTD parameter descriptors are a counted array, not
 * NULL-terminated, so never scan beyond the task class's declared parameters.
 */
static void
cuda_pack_value_args(parsec_task_t *this_task, ...)
{
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)this_task;
    parsec_dtd_task_class_t *tc =
        (parsec_dtd_task_class_t *)current_task->super.task_class;
    parsec_dtd_task_param_t *current_param = GET_HEAD_OF_PARAM_LIST(current_task);
    void *tmp_val;
    va_list arguments;

    va_start(arguments, this_task);
    for( int i = 0; i < tc->count_of_params; i++, current_param++ ) {
        assert(PARSEC_VALUE ==
               (current_param->op_type & PARSEC_GET_OP_TYPE));
        tmp_val = va_arg(arguments, void *);
        memcpy(current_param->pointer_to_tile, tmp_val,
               current_param->arg_size);
    }
    va_end(arguments);
}

#define STATUS_TASKS 32

/* ASYNC transfers execution-context ownership to the submit hook. Controller
 * tasks retrieve the contexts from this table and reschedule each one once;
 * the completion arrays detect both lost and duplicate logical tasks.
 */
static parsec_task_t *array_of_async_tasks[STATUS_TASKS];
static parsec_gpu_task_t *again_batch_tasks[STATUS_TASKS];
static parsec_gpu_task_t *again_batch_order[STATUS_TASKS];
static int async_completed[STATUS_TASKS];
static int again_batch_completed[STATUS_TASKS];
static int next_completed[STATUS_TASKS];
static int async_batch_observed;
static int again_batch_observed;
static int again_batch_yields;
static int again_batch_release_count;
static int again_profile_state_valid = 1;
static int next_batch_observed;
static int next_returned;
static int priority_batch_observed;
static int priority_release_count;
static int priority_order_valid = 1;
static int priority_last_released = INT_MAX;
static parsec_gpu_task_t *stop_candidate;
static int stop_observed;
static int stop_candidate_completed;

/* Verify that final event completion closes the logical execution interval
 * before the device wrapper is released.
 */
static void
cuda_again_release(parsec_gpu_task_t *gpu_task)
{
#if defined(PARSEC_PROF_TRACE)
    if( PARSEC_GPU_TASK_PROF_EXEC_OPEN == gpu_task->prof_exec_state ) {
        again_profile_state_valid = 0;
    }
#endif
    again_batch_release_count++;
    PARSEC_OBJ_RELEASE(gpu_task);
}

/* Only collect tasks from the same test phase and task class. Phase matching
 * prevents resubmitted ASYNC contexts from joining their initial submission.
 */
static int
cuda_batch_match_same_phase(parsec_gpu_task_t *candidate,
                            parsec_gpu_task_t *batch_head,
                            void *callback_data)
{
    int candidate_id, candidate_phase;
    int head_id, head_phase;

    (void)callback_data;
    if( candidate->ec->task_class != batch_head->ec->task_class ) {
        return PARSEC_GPU_TASK_BATCH_REJECT;
    }
    parsec_dtd_unpack_args(candidate->ec, &candidate_id, &candidate_phase);
    parsec_dtd_unpack_args(batch_head->ec, &head_id, &head_phase);
    (void)candidate_id;
    (void)head_id;
    return (candidate_phase == head_phase) ? PARSEC_GPU_TASK_BATCH_ACCEPT
                                           : PARSEC_GPU_TASK_BATCH_REJECT;
}

/**
 * Exercise batched AGAIN, ASYNC, and NEXT returns, successful-ring priority
 * ordering, and early collection STOP. AGAIN verifies that repeated
 * coroutine-style yields preserve the exact ring. ASYNC hands every execution
 * context to controller tasks, which reschedule them once. NEXT verifies that
 * all restored followers complete exactly once. PRIORITY verifies ordering in
 * the next GPU stream after a deliberately unsorted batch succeeds. STOP
 * verifies that its current candidate remains pending and later executes.
 */

int cuda_task_async(parsec_device_gpu_module_t *gpu_device,
                    parsec_gpu_task_t *gpu_task,
                    parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_gpu_task_t *current;
    int batch_count, i, first;

    (void)gpu_device;
    batch_count = 1 + parsec_gpu_task_collect_batch(gpu_stream, gpu_task,
                                                    cuda_batch_match_same_phase,
                                                    NULL);
    parsec_dtd_unpack_args(gpu_task->ec, &i, &first);
    if( 0 == first ) {
        /* Fill the execution stream before handing tasks out asynchronously so
         * at least one invocation exercises ASYNC with a real task ring.
         */
        if( (1 == batch_count) && !async_batch_observed ) {
            return PARSEC_HOOK_RETURN_AGAIN;
        }
        if( batch_count > 1 ) {
            async_batch_observed = 1;
        }
        current = gpu_task;
        do {
            parsec_task_t *this_task = current->ec;
            int stored;

            parsec_dtd_unpack_args(this_task, &i, &first);
            assert(0 == first);
            first = 1;
            cuda_pack_value_args(this_task, &i, &first);
            PARSEC_LIST_ITEM_SINGLETON(this_task);
            assert(i >= 0 && i < STATUS_TASKS);
            stored = parsec_atomic_cas_ptr(&array_of_async_tasks[i], NULL, this_task);
            assert(stored);
            if( !stored ) {
                return PARSEC_HOOK_RETURN_ERROR;
            }
            current = (parsec_gpu_task_t *)current->list_item.list_next;
        } while( current != gpu_task );
        return PARSEC_HOOK_RETURN_ASYNC;
    }

    current = gpu_task;
    do {
        parsec_dtd_unpack_args(current->ec, &i, &first);
        assert(1 == first);
        assert(i >= 0 && i < STATUS_TASKS);
        assert(0 == async_completed[i]);
        async_completed[i] = 1;
        current = (parsec_gpu_task_t *)current->list_item.list_next;
    } while( current != gpu_task );
    return PARSEC_HOOK_RETURN_DONE;
}

/* Wait until an ASYNC hook publishes the matching execution context, claim it
 * atomically, and return it to normal runtime scheduling exactly once.
 */
int cuda_task_again(parsec_device_gpu_module_t *gpu_device,
                    parsec_gpu_task_t *gpu_task,
                    parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_task_t *async_task;
    int i;

    (void)gpu_device; (void)gpu_stream;
    parsec_dtd_unpack_args(this_task, &i);
    assert(i >= 0 && i < STATUS_TASKS);
    async_task = array_of_async_tasks[i];
    if( (NULL == async_task) ||
        !parsec_atomic_cas_ptr(&array_of_async_tasks[i], async_task, NULL) ) {
        return PARSEC_HOOK_RETURN_AGAIN;
    }
    __parsec_reschedule(parsec_my_execution_stream(), async_task);
    return PARSEC_HOOK_RETURN_DONE;
}

/* Model a coroutine batch that submits progress twice before completing. The
 * first batched AGAIN commits the ring; each continuation must receive exactly
 * the same wrappers, while later phase-zero tasks remain independent work.
 */
int cuda_task_batch_again(parsec_device_gpu_module_t *gpu_device,
                          parsec_gpu_task_t *gpu_task,
                          parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_gpu_task_t *current;
    uint32_t seen = 0;
    int batch_count, count = 0, id, phase;

    (void)gpu_device;
    parsec_dtd_unpack_args(gpu_task->ec, &id, &phase);
    if( 0 == phase ) {
        if( again_batch_observed ) {
            if( (id < 0) || (id >= STATUS_TASKS) ||
                (0 != again_batch_completed[id]) ) {
                return PARSEC_HOOK_RETURN_ERROR;
            }
            again_batch_completed[id] = 1;
            return PARSEC_HOOK_RETURN_DONE;
        }

        batch_count = 1 + parsec_gpu_task_collect_batch(gpu_stream, gpu_task,
                                                        cuda_batch_match_same_phase,
                                                        NULL);
        if( 1 == batch_count ) {
            /* No work was collected, so this remains a normal singleton retry. */
            return PARSEC_HOOK_RETURN_AGAIN;
        }

        again_batch_observed = batch_count;
        current = gpu_task;
        do {
            parsec_dtd_unpack_args(current->ec, &id, &phase);
            if( (id < 0) || (id >= STATUS_TASKS) || (0 != phase) ||
                (NULL != again_batch_tasks[id]) || (seen & (1U << id)) ) {
                return PARSEC_HOOK_RETURN_ERROR;
            }
            seen |= (1U << id);
            again_batch_tasks[id] = current;
            again_batch_order[count] = current;
            current->release_device_task = cuda_again_release;
            phase = 1;
            cuda_pack_value_args(current->ec, &id, &phase);
            count++;
            current = (parsec_gpu_task_t *)current->list_item.list_next;
        } while( current != gpu_task );
        if( count != batch_count ) {
            return PARSEC_HOOK_RETURN_ERROR;
        }
        again_batch_yields++;
        return PARSEC_HOOK_RETURN_AGAIN;
    }

    /* Verify ring identity before calling the collector, so reconstructing a
     * batch from followers returned to fifo_pending cannot satisfy the test.
     */
    current = gpu_task;
    do {
        int member_phase;

        parsec_dtd_unpack_args(current->ec, &id, &member_phase);
        if( (count >= STATUS_TASKS) || (id < 0) || (id >= STATUS_TASKS) ||
            (member_phase != phase) ||
            (again_batch_tasks[id] != current) ||
            (again_batch_order[count] != current) || (seen & (1U << id)) ) {
            return PARSEC_HOOK_RETURN_ERROR;
        }
#if defined(PARSEC_PROF_TRACE)
        /* Every member starts once after the first finalized submission. The
         * interval must remain open on each coroutine continuation.
         */
        if( gpu_stream->prof_event_track_enable && parsec_profile_enabled &&
            (PARSEC_GPU_TASK_PROF_EXEC_OPEN != current->prof_exec_state) ) {
            return PARSEC_HOOK_RETURN_ERROR;
        }
#endif
        seen |= (1U << id);
        count++;
        current = (parsec_gpu_task_t *)current->list_item.list_next;
    } while( current != gpu_task );
    if( count != again_batch_observed ) {
        return PARSEC_HOOK_RETURN_ERROR;
    }

    /* Calling the collector on a continuation must report the existing ring
     * without changing it or adding pending phase-zero tasks.
     */
    batch_count = 1 + parsec_gpu_task_collect_batch(gpu_stream, gpu_task,
                                                    cuda_batch_match_same_phase,
                                                    NULL);
    if( batch_count != again_batch_observed ) {
        return PARSEC_HOOK_RETURN_ERROR;
    }

    if( phase < 2 ) {
        current = gpu_task;
        do {
            parsec_dtd_unpack_args(current->ec, &id, &phase);
            phase++;
            cuda_pack_value_args(current->ec, &id, &phase);
            current = (parsec_gpu_task_t *)current->list_item.list_next;
        } while( current != gpu_task );
        again_batch_yields++;
        return PARSEC_HOOK_RETURN_AGAIN;
    }

    current = gpu_task;
    do {
        parsec_dtd_unpack_args(current->ec, &id, &phase);
        if( (id < 0) || (id >= STATUS_TASKS) || (2 != phase) ||
            (0 != again_batch_completed[id]) ) {
            return PARSEC_HOOK_RETURN_ERROR;
        }
        again_batch_completed[id] = 1;
        current = (parsec_gpu_task_t *)current->list_item.list_next;
    } while( current != gpu_task );
    return PARSEC_HOOK_RETURN_DONE;
}

/* Force one multi-task NEXT result, then verify that the restored followers
 * and delayed singleton head all return and complete exactly once.
 */
int cuda_task_next(parsec_device_gpu_module_t *gpu_device,
                   parsec_gpu_task_t *gpu_task,
                   parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_gpu_task_t *current;
    int batch_count, i, phase;

    (void)gpu_device;
    batch_count = 1 + parsec_gpu_task_collect_batch(gpu_stream, gpu_task,
                                                    cuda_batch_match_same_phase,
                                                    NULL);
    if( !next_returned ) {
        /* AGAIN fills the stream until NEXT can be returned with followers.
         * NEXT must restore those followers and requeue only the head.
         */
        if( 1 == batch_count ) {
            return PARSEC_HOOK_RETURN_AGAIN;
        }
        next_batch_observed = batch_count;
        next_returned = 1;
        return PARSEC_HOOK_RETURN_NEXT;
    }

    current = gpu_task;
    do {
        parsec_dtd_unpack_args(current->ec, &i, &phase);
        assert(0 == phase);
        assert(i >= 0 && i < STATUS_TASKS);
        assert(0 == next_completed[i]);
        next_completed[i] = 1;
        current = (parsec_gpu_task_t *)current->list_item.list_next;
    } while( current != gpu_task );
    return PARSEC_HOOK_RETURN_DONE;
}

/* Record only wrappers from the deliberately unsorted priority batch. Other
 * tasks may complete between them, but the marked wrappers must be released
 * in non-increasing priority order after the ring enters the next stream.
 */
static void
cuda_priority_release(parsec_gpu_task_t *gpu_task)
{
    if( gpu_task->priority > priority_last_released ) {
        priority_order_valid = 0;
    }
    priority_last_released = gpu_task->priority;
    priority_release_count++;
    PARSEC_OBJ_RELEASE(gpu_task);
}

/* Build one successful batch, then make its linked order intentionally differ
 * from its priority order. The next stream must sort the complete ring instead
 * of appending it as-is.
 */
int cuda_task_priority(parsec_device_gpu_module_t *gpu_device,
                       parsec_gpu_task_t *gpu_task,
                       parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_gpu_task_t *current;
    int batch_count, position = 0;

    (void)gpu_device;
    if( priority_batch_observed ) {
        /* Leave later tasks as singletons so the next stream must merge a
         * successful batch with independently submitted work.
         */
        return PARSEC_HOOK_RETURN_DONE;
    }
    batch_count = 1 + parsec_gpu_task_collect_batch(gpu_stream, gpu_task,
                                                    cuda_batch_match_same_phase,
                                                    NULL);
    if( 1 == batch_count ) {
        return PARSEC_HOOK_RETURN_AGAIN;
    }
    priority_batch_observed = batch_count;
    current = gpu_task;
    do {
        /* Put the lowest-priority task at the ring head and descending
         * positive priorities behind it, guaranteeing an unsorted ring.
         */
        current->ec->priority = (0 == position) ? 0 : batch_count - position + 1;
        current->release_device_task = cuda_priority_release;
        position++;
        current = (parsec_gpu_task_t *)current->list_item.list_next;
    } while( current != gpu_task );
    return PARSEC_HOOK_RETURN_DONE;
}

/* Stop at the first compatible follower without accepting it. The collector
 * must report success and leave this exact wrapper in the pending FIFO.
 */
static int
cuda_batch_stop_first(parsec_gpu_task_t *candidate,
                      parsec_gpu_task_t *batch_head,
                      void *callback_data)
{
    (void)callback_data;
    if( candidate->ec->task_class != batch_head->ec->task_class ) {
        return PARSEC_GPU_TASK_BATCH_REJECT;
    }
    stop_candidate = candidate;
    stop_observed++;
    return PARSEC_GPU_TASK_BATCH_STOP;
}

/* Retry until a compatible pending task lets the callback exercise STOP. Once
 * STOP succeeds, every task completes as a singleton; observing the saved
 * wrapper later proves that STOP did not remove or lose its current candidate.
 */
int cuda_task_batch_stop(parsec_device_gpu_module_t *gpu_device,
                         parsec_gpu_task_t *gpu_task,
                         parsec_gpu_exec_stream_t *gpu_stream)
{
    int nb_batched;

    (void)gpu_device;
    if( stop_observed ) {
        if( gpu_task == stop_candidate ) {
            stop_candidate_completed = 1;
        }
        return PARSEC_HOOK_RETURN_DONE;
    }

    nb_batched = parsec_gpu_task_collect_batch(gpu_stream, gpu_task,
                                               cuda_batch_stop_first, NULL);
    if( nb_batched < 0 ) {
        return nb_batched;
    }
    if( !stop_observed ) {
        return PARSEC_HOOK_RETURN_AGAIN;
    }
    return (0 == nb_batched) ? PARSEC_HOOK_RETURN_DONE
                             : PARSEC_HOOK_RETURN_ERROR;
}

int main(int argc, char* argv[])
{
    int ret;
    parsec_context_t *parsec_context = NULL;
    int rank, world;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif
    (void)rank;
    (void)world;

    parsec_context = parsec_init(-1, &argc, &argv);

    /* Every task class below carries a CUDA chore only, so without an
     * accelerator task creation fatals on an unsatisfiable device mask. Report
     * a skip instead (see SKIP_RETURN_CODE in tests/CMakeLists.txt). */
    if( parsec_context_query(parsec_context, PARSEC_CONTEXT_QUERY_DEVICES, PARSEC_DEV_CUDA) <= 0 ) {
        parsec_warning("This test can only run if at least one CUDA device is present");
        parsec_fini(&parsec_context);
#if defined(PARSEC_HAVE_MPI)
        MPI_Finalize();
#endif
        return -PARSEC_ERR_DEVICE;
    }

    // Create new DTD taskpool
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_task_class_t *again_tc, *batch_again_tc, *async_tc, *next_tc;
    parsec_task_class_t *priority_tc, *stop_tc;

    ret = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(ret, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    ret = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(ret, "parsec_context_add_taskpool");

    again_tc = parsec_dtd_create_task_class(tp, "AGAIN",
                                            sizeof(int), PARSEC_VALUE,  /* i */
                                            PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, again_tc, PARSEC_DEV_CUDA, cuda_task_again);

    batch_again_tc = parsec_dtd_create_task_class(tp, "BATCH AGAIN",
                                                  sizeof(int), PARSEC_VALUE,  /* i */
                                                  sizeof(int), PARSEC_VALUE,  /* phase */
                                                  PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, batch_again_tc,
                                    PARSEC_DEV_CUDA | PARSEC_DEV_CHORE_ALLOW_BATCH,
                                    cuda_task_batch_again);

    async_tc = parsec_dtd_create_task_class(tp, "ASYNC",
                                            sizeof(int), PARSEC_VALUE,  /* i */
                                            sizeof(int), PARSEC_VALUE,  /* phase */
                                            PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, async_tc,
                                    PARSEC_DEV_CUDA | PARSEC_DEV_CHORE_ALLOW_BATCH,
                                    cuda_task_async);

    next_tc = parsec_dtd_create_task_class(tp, "NEXT",
                                           sizeof(int), PARSEC_VALUE,  /* i */
                                           sizeof(int), PARSEC_VALUE,  /* phase */
                                           PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, next_tc,
                                    PARSEC_DEV_CUDA | PARSEC_DEV_CHORE_ALLOW_BATCH,
                                    cuda_task_next);

    priority_tc = parsec_dtd_create_task_class(tp, "PRIORITY",
                                               sizeof(int), PARSEC_VALUE,  /* i */
                                               sizeof(int), PARSEC_VALUE,  /* phase */
                                               PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, priority_tc,
                                    PARSEC_DEV_CUDA | PARSEC_DEV_CHORE_ALLOW_BATCH,
                                    cuda_task_priority);

    stop_tc = parsec_dtd_create_task_class(tp, "BATCH STOP",
                                           sizeof(int), PARSEC_VALUE,  /* i */
                                           PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, stop_tc,
                                    PARSEC_DEV_CUDA | PARSEC_DEV_CHORE_ALLOW_BATCH,
                                    cuda_task_batch_stop);

    int zero = 0;
    for( int i = 0; i < STATUS_TASKS; ++i ) {
        parsec_dtd_insert_task_with_task_class(tp, async_tc, 0, PARSEC_DEV_ALL,
                                               PARSEC_VALUE, &i,
                                               PARSEC_VALUE, &zero,
                                               PARSEC_DTD_ARG_END);
    }

    for( int i = 0; i < STATUS_TASKS; ++i ) {
        parsec_dtd_insert_task_with_task_class(tp, again_tc, 0, PARSEC_DEV_ALL,
                                               PARSEC_VALUE, &i,
                                               PARSEC_DTD_ARG_END);
    }

    for( int i = 0; i < STATUS_TASKS; ++i ) {
        parsec_dtd_insert_task_with_task_class(tp, batch_again_tc, 0, PARSEC_DEV_ALL,
                                               PARSEC_VALUE, &i,
                                               PARSEC_VALUE, &zero,
                                               PARSEC_DTD_ARG_END);
    }

    for( int i = 0; i < STATUS_TASKS; ++i ) {
        parsec_dtd_insert_task_with_task_class(tp, next_tc, 0, PARSEC_DEV_ALL,
                                               PARSEC_VALUE, &i,
                                               PARSEC_VALUE, &zero,
                                               PARSEC_DTD_ARG_END);
    }

    for( int i = 0; i < STATUS_TASKS; ++i ) {
        parsec_dtd_insert_task_with_task_class(tp, priority_tc, 0, PARSEC_DEV_ALL,
                                               PARSEC_VALUE, &i,
                                               PARSEC_VALUE, &zero,
                                               PARSEC_DTD_ARG_END);
    }

    for( int i = 0; i < STATUS_TASKS; ++i ) {
        parsec_dtd_insert_task_with_task_class(tp, stop_tc, 0, PARSEC_DEV_ALL,
                                               PARSEC_VALUE, &i,
                                               PARSEC_DTD_ARG_END);
    }

    // Wait for task completion
    ret = parsec_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(ret, "parsec_taskpool_wait");

    ret = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(ret, "parsec_context_wait");

    if( !async_batch_observed || (next_batch_observed <= 1) ) {
        parsec_warning("GPU batch status test did not form ASYNC and NEXT task rings\n");
        ret = 1;
    }
    if( (again_batch_observed <= 1) || (2 != again_batch_yields) ) {
        parsec_warning("GPU AGAIN batch size=%d yielded=%d times\n",
                       again_batch_observed, again_batch_yields);
        ret = 1;
    }
    if( !again_profile_state_valid ||
        (again_batch_release_count != again_batch_observed) ) {
        parsec_warning("GPU AGAIN profiling state valid=%d released=%d expected=%d\n",
                       again_profile_state_valid, again_batch_release_count,
                       again_batch_observed);
        ret = 1;
    }
    if( (priority_batch_observed <= 1) || !priority_order_valid ||
        (priority_release_count != priority_batch_observed) ) {
        parsec_warning("GPU priority batch size=%d released=%d ordered=%d\n",
                       priority_batch_observed, priority_release_count,
                       priority_order_valid);
        ret = 1;
    }
    if( (1 != stop_observed) || !stop_candidate_completed ) {
        parsec_warning("GPU batch STOP observed=%d candidate completed=%d\n",
                       stop_observed, stop_candidate_completed);
        ret = 1;
    }
    for( int i = 0; i < STATUS_TASKS; i++ ) {
        if( (1 != async_completed[i]) || (1 != again_batch_completed[i]) ||
            (1 != next_completed[i]) ) {
            parsec_warning("GPU batch status task %d completed ASYNC=%d AGAIN=%d NEXT=%d times\n",
                           i, async_completed[i], again_batch_completed[i],
                           next_completed[i]);
            ret = 1;
        }
    }

    parsec_dtd_task_class_release(tp, again_tc);
    parsec_dtd_task_class_release(tp, batch_again_tc);
    parsec_dtd_task_class_release(tp, async_tc);
    parsec_dtd_task_class_release(tp, next_tc);
    parsec_dtd_task_class_release(tp, priority_tc);
    parsec_dtd_task_class_release(tp, stop_tc);

    parsec_taskpool_free(tp);

    parsec_fini(&parsec_context);

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif
    return ret;
}
