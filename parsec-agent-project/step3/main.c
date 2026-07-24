#include <stdio.h>
#include <mpi.h>
#include <parsec.h>
#include <parsec/interfaces/dtd/insert_function.h>
#include <parsec/mca/device/device.h>

#define MAX_TASKS 5

// Forward declaration of the task body
static int dtd_loop_body(parsec_execution_stream_t *es, parsec_task_t *task);

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    parsec_context_t *parsec_context = parsec_init(-1, NULL, NULL);
    if(NULL == parsec_context) {
        MPI_Finalize();
        return -1;
    }

    parsec_taskpool_t *tp = parsec_dtd_taskpool_new();

    // Define task class
    parsec_dtd_create_task_class(tp, "loop_task",
                                 sizeof(int), PARSEC_VALUE,
                                 PARSEC_DTD_ARG_END,
                                 dtd_loop_body,
                                 0);

    parsec_context_add_taskpool(parsec_context, tp);
    parsec_context_start(parsec_context);

    // Start the loop with task 0
    int first_task = 0;
    parsec_dtd_insert_task(tp, dtd_loop_body, 0, 
                           PARSEC_DEV_CPU, 
                           "loop_task",
                           sizeof(int), &first_task, PARSEC_VALUE,
                           PARSEC_DTD_ARG_END);

    parsec_taskpool_wait(tp);
    parsec_taskpool_free(tp);

    parsec_fini(&parsec_context);
    MPI_Finalize();
    return 0;
}

static int dtd_loop_body(parsec_execution_stream_t *es, parsec_task_t *task)
{
    int current_idx;
    parsec_dtd_unpack_args(task, &current_idx);

    printf("Task N=%d executing\n", current_idx);

    if (current_idx < MAX_TASKS - 1) {
        int next_idx = current_idx + 1;
        printf("Task N=%d inserting Task N=%d\n", current_idx, next_idx);
        
        // Use parsec_dtd_get_taskpool(task) to get the taskpool
        parsec_dtd_insert_task(parsec_dtd_get_taskpool(task), dtd_loop_body, 0, 
                               PARSEC_DEV_CPU, 
                               "loop_task",
                               sizeof(int), &next_idx, PARSEC_VALUE,
                               PARSEC_DTD_ARG_END);
    }

    return PARSEC_HOOK_RETURN_DONE;
}
