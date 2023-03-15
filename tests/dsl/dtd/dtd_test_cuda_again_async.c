#include "parsec.h"
#include "parsec/arena.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "tests/tests_data.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

void parsec_dtd_pack_args( parsec_task_t *this_task, ... )
{
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)this_task;
    parsec_dtd_task_param_t *current_param = GET_HEAD_OF_PARAM_LIST(current_task);
    int i = 0;
    void *tmp_val;
    void **tmp_ref;
    va_list arguments;

    va_start(arguments, this_task);
    while( current_param != NULL) {
        if((current_param->op_type & PARSEC_GET_OP_TYPE) == PARSEC_VALUE ) {
            tmp_val = va_arg(arguments, void*);
            memcpy(current_param->pointer_to_tile, tmp_val, current_param->arg_size);
        } else if((current_param->op_type & PARSEC_GET_OP_TYPE) == PARSEC_SCRATCH ||
                  (current_param->op_type & PARSEC_GET_OP_TYPE) == PARSEC_REF ) {
            tmp_ref = va_arg(arguments, void**);
            current_param->pointer_to_tile = *tmp_ref;
        } else if((current_param->op_type & PARSEC_GET_OP_TYPE) == PARSEC_INPUT ||
                  (current_param->op_type & PARSEC_GET_OP_TYPE) == PARSEC_INOUT ||
                  (current_param->op_type & PARSEC_GET_OP_TYPE) == PARSEC_OUTPUT ) {
            tmp_ref = va_arg(arguments, void**);
            this_task->data[i].data_in = *tmp_ref;
            assert(0);
            i++;
        } else {
            parsec_warning("/!\\ Flag is not recognized in parsec_dtd_unpack_args /!\\.\n");
            assert(0);
        }
        current_param = current_param->next;
    }
    va_end(arguments);
}

static int max_repeat = 50;
static parsec_task_t* array_of_async_tasks[100];

/**
 * This test check the correct handling of the PARSEC_HOOK_RETURN_ASYNC and
 * PARSEC_HOOK_RETURN_AGAIN. The cuda_task_async will atomically save the task 
 * onto a predefined array, and the cuda_task_again will repeat itself until
 * the async task appears on the array. At that point it reenable the async task
 * and continue it's execution until a predefined number of iteration have been
 * reached. If more iterations have been already done it will return asap.
 */

int cuda_task_async(parsec_device_gpu_module_t *gpu_device,
                    parsec_gpu_task_t *gpu_task,
                    parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    int i, first;

    (void)gpu_device; (void)gpu_stream;
    parsec_dtd_unpack_args(this_task, &i, &first);
    if( 0 == first ) {
        first += 1;  /* mark the second call to this task */
        parsec_dtd_pack_args(this_task, &i, &first);
        PARSEC_LIST_ITEM_SINGLETON(this_task);
        fprintf(stdout, "Task %p preparing for async behavior\n", this_task);
        parsec_atomic_cas_ptr(&array_of_async_tasks[i], NULL, this_task);
        return PARSEC_HOOK_RETURN_ASYNC;
    }
    fprintf(stdout, "Task %p is back alive after an async. Complete and leave\n", this_task);
    return PARSEC_HOOK_RETURN_DONE;
}

int cuda_task_again(parsec_device_gpu_module_t *gpu_device,
                    parsec_gpu_task_t *gpu_task,
                    parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    int i, repeat, done;

    (void)gpu_device; (void)gpu_stream;
    parsec_dtd_unpack_args(this_task, &i, &repeat, &done);
    if( !done  ) {
        repeat += 1;
        if( NULL == array_of_async_tasks[i] ) {
            fprintf(stdout, "Task [%d] %p waiting for the async task (repeat %d)\n", i, this_task, repeat - 1);
        } else {
            /* There is a small opportunity for race conditions between the insertion of the async task and
               it's reschedule by the again task.
             */
            sleep(1);
            parsec_task_t* async_task = array_of_async_tasks[i];
            fprintf(stdout, "Async Task %p is reinserted into the runtime\n", async_task);
            array_of_async_tasks[i] = NULL;
            parsec_execution_stream_t* local_es = parsec_my_execution_stream();
            __parsec_reschedule(local_es, async_task);
            done = 1;
        }
        parsec_dtd_pack_args(this_task, &i, &repeat, &done);
        return PARSEC_HOOK_RETURN_AGAIN;
    }
    if(repeat < max_repeat) {
        fprintf(stdout, "Task [%d] %p is cycling while waiting for repeat (%d)\n", i, this_task, repeat);
        repeat += 1;
        parsec_dtd_pack_args(this_task, &i, &repeat, &done);
        return PARSEC_HOOK_RETURN_AGAIN;
    }
    fprintf(stdout, "Task [%d] %p is now officially complete\n", i, this_task);
    return PARSEC_HOOK_RETURN_DONE;
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

    parsec_context = parsec_init(-1, NULL, NULL);
    // Create new DTD taskpool
    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    parsec_task_class_t *again_tc, *async_tc;

    ret = parsec_context_start(parsec_context);
    PARSEC_CHECK_ERROR(ret, "parsec_context_start");

    // Registering the dtd_handle with PARSEC context
    ret = parsec_context_add_taskpool(parsec_context, tp);
    PARSEC_CHECK_ERROR(ret, "parsec_context_add_taskpool");

    again_tc = parsec_dtd_create_task_class(tp, "AGAIN",
                                            sizeof(int), PARSEC_VALUE,  /* i */
                                            sizeof(int), PARSEC_VALUE,  /* repeat */
                                            sizeof(int), PARSEC_VALUE,  /* done */
                                            PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, again_tc, PARSEC_DEV_CUDA, cuda_task_again);

    async_tc = parsec_dtd_create_task_class(tp, "ASYNC",
                                            sizeof(int), PARSEC_VALUE,  /* i */
                                            sizeof(int), PARSEC_VALUE,  /* repeat */
                                            PARSEC_DTD_ARG_END);
    parsec_dtd_task_class_add_chore(tp, async_tc, PARSEC_DEV_CUDA, cuda_task_async);

    int zero = 0, done = 0;
    for( int i = 0; i < world; ++i ) {
        parsec_dtd_insert_task_with_task_class(tp, async_tc, 0, PARSEC_DEV_ALL,
                                               PARSEC_AFFINITY, &i,
                                               PARSEC_VALUE, &zero,
                                               PARSEC_DTD_ARG_END);
    }

    for( int i = 0; i < world; ++i ) {
        parsec_dtd_insert_task_with_task_class(tp, again_tc, 0, PARSEC_DEV_ALL,
                                               PARSEC_AFFINITY, &i,
                                               PARSEC_VALUE, &zero,
                                               PARSEC_VALUE, &done,
                                               PARSEC_DTD_ARG_END);
    }

    // Wait for task completion
    ret = parsec_taskpool_wait(tp);
    PARSEC_CHECK_ERROR(ret, "parsec_taskpool_wait");

    ret = parsec_context_wait(parsec_context);
    PARSEC_CHECK_ERROR(ret, "parsec_context_wait");

    parsec_dtd_task_class_release(tp, again_tc);
    parsec_dtd_task_class_release(tp, async_tc);

    parsec_taskpool_free(tp);

    parsec_fini(&parsec_context);

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif
}
