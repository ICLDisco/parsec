/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA CORPORATION. All rights reserved.
 */

/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "tests/tests_timing.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

double time_elapsed;
double sync_time_elapsed;

int
real_task( parsec_execution_stream_t *es,
           parsec_task_t *this_task )
{
    (void)es; (void)this_task;

    parsec_output( 0, "I am %d and I am executing a real task\n", es->th_id );
    usleep(10);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_to_insert_task( parsec_execution_stream_t *es,
                     parsec_task_t *this_task )
{
    (void)es;

    parsec_taskpool_t* dtd_tp = this_task->taskpool;
    int total, increment, *count, i;

    parsec_dtd_unpack_args(this_task, &total, &count, &increment);

    parsec_output( 0, "Task inserting task by thread: %d count: %d Total: %d increment: %d total_inserted: %d\n",
                   es->th_id, *count, total, increment, *count-1 );

    for( i = 0; *count < total; i++, *count += 1 ) {
        if( i > increment ) {
            /* Return some kind of rescheduling signal */
            return PARSEC_HOOK_RETURN_AGAIN;
        }
        /* Inserting real task */
        parsec_dtd_insert_task(dtd_tp, real_task, 0, PARSEC_DEV_CPU, "Real_Task",
                               PARSEC_DTD_ARG_END );
    }

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, cores = -1, rc;

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, &rank, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    int m;
    int no_of_tasks = 1;

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new(  );

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool(parsec, (parsec_taskpool_t *)dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    SYNC_TIME_START(parsec);
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    int total_tasks = 20;
    int increment   = 5;
    int count       = 0;

    for( m = 0; m < no_of_tasks; m++ ) {
        parsec_dtd_insert_task(dtd_tp, task_to_insert_task, 0, PARSEC_DEV_CPU, "Task_inserting_Task",
                               sizeof(int), &total_tasks, PARSEC_VALUE,
                               sizeof(int), &count, PARSEC_REF,
                               sizeof(int), &increment, PARSEC_VALUE,
                               PARSEC_DTD_ARG_END );
    }

    /* finishing all the tasks inserted, but not finishing the handle */
    rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    SYNC_TIME_PRINT(parsec, rank, ("\n"));

    parsec_taskpool_free( dtd_tp );

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
