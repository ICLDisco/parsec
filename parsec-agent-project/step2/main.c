#include <stdio.h>
#include <mpi.h>
#include <parsec.h>
#include <parsec/interfaces/dtd/insert_function.h>
#include <parsec/mca/device/device.h>

static int dtd_hello_body(parsec_execution_stream_t *es, parsec_task_t *task)
{
    int index;
    int value;

    parsec_dtd_unpack_args(task, &index, &value);

    printf("Task index %d received value %d\n", index, value);

    return PARSEC_HOOK_RETURN_DONE;
}

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

    parsec_dtd_create_task_class(tp, "hello_task",
                                 sizeof(int), PARSEC_VALUE,
                                 sizeof(int), PARSEC_VALUE,
                                 PARSEC_DTD_ARG_END,
                                 dtd_hello_body,
                                 0);

    parsec_context_add_taskpool(parsec_context, tp);
    parsec_context_start(parsec_context);

    for(int i = 0; i < 4; i++) {
        int val = i * 10;
        // DTD insert task takes function pointer, priority, and device_type
        parsec_dtd_insert_task(tp, dtd_hello_body, 0, 
                               PARSEC_DEV_CPU, 
                               "hello_task",
                               sizeof(int), &i, PARSEC_VALUE,
                               sizeof(int), &val, PARSEC_VALUE,
                               PARSEC_DTD_ARG_END);
    }

    parsec_taskpool_wait(tp);
    parsec_taskpool_free(tp);

    parsec_fini(&parsec_context);
    MPI_Finalize();
    return 0;
}
