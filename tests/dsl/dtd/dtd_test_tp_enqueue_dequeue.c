/*
 * Copyright (c) 2018-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* parsec things */
#include "parsec.h"
#include "parsec/profiling.h"

#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/scheduling.h"
#include "tests/tests_runtime.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

int
task(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int rc;

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool(this_task->taskpool->context, dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    parsec_taskpool_free(dtd_tp);

    return PARSEC_HOOK_RETURN_DONE;
}

int parsec_complete_tp_callback(parsec_taskpool_t* tp, void* cb_data)
{
    (void)tp;
    int rc = PARSEC_HOOK_RETURN_DONE;
    parsec_task_t *this_task = (parsec_task_t *)cb_data;
    rc = __parsec_complete_execution(this_task->taskpool->context->virtual_processes[0]->execution_streams[0],
                                     this_task);

    return rc;
}

int
task_with_callback(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    int rc, i;

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool(this_task->taskpool->context, dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    /* parsec_context_wait() will be called a lot later,
     * let's detach it explicitly so that the tps are over */
    for(i = 0; i < 1000; i++) {
        parsec_dtd_insert_task(dtd_tp, task, 0, PARSEC_DEV_CPU, "task",
                               PARSEC_DTD_ARG_END);
    }

    /* We expect to complete this taskpool asynchronously, via the call to
     * parsec_complete_tp_callback.
     */
    parsec_taskpool_set_complete_callback(dtd_tp, parsec_complete_tp_callback,
                                          (void *)this_task);
    /* Let the runtime forget about the taskpool for now, which does not mean
     * the taskpool does not progress, simply that the user returned the only
     * reference to this taskpool it was supposed to store, and thus it will
     * not be able to add more tasks to the taskpoll, such that now a call to
     * parsec_context_wait can now complete.
     */
    rc = parsec_dtd_dequeue_taskpool(dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_dequeue_taskpool");
    parsec_taskpool_free(dtd_tp);

    return PARSEC_HOOK_RETURN_ASYNC;
}

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rc, i;
    int world, cores = -1;

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv,
                                   &parsec, NULL, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    if( world != 1 ) {
        parsec_fatal( "Nope! world is not right, we need exactly one process. "
                      "Try with a single-process launcher.\n" );
    }

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    for(i = 0; i < 10000; i++) {
        parsec_dtd_insert_task(dtd_tp, task, 0, PARSEC_DEV_CPU, "task",
                               PARSEC_DTD_ARG_END);
    }

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free(dtd_tp);

    dtd_tp = parsec_dtd_taskpool_new();


    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    for(i = 0; i < 1; i++) {
        parsec_dtd_insert_task(dtd_tp, task_with_callback, 0, PARSEC_DEV_CPU, "task1",
                               PARSEC_DTD_ARG_END);
    }

    usleep(100000);

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free(dtd_tp);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
