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
call_to_kernel_type( parsec_execution_stream_t *es,
                     parsec_task_t *this_task )
{
    (void)es;

    parsec_output( 0, "Executing task with null as tile in rank: %d\n", this_task->taskpool->context->my_rank );

    //this_task->data[0].data_out = (parsec_data_copy_t *)context;

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
    rc = parsec_context_add_taskpool( parsec, dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    SYNC_TIME_START(parsec);
    rc = parsec_context_start( parsec );
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");


    for( m = 0; m < no_of_tasks; m++ ) {
        parsec_dtd_insert_task(dtd_tp, call_to_kernel_type, 0, PARSEC_DEV_CPU, "Test_Task",
                               PASSED_BY_REF, NULL, PARSEC_INOUT,
                               PARSEC_DTD_ARG_END);
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
