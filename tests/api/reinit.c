/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/*
 * A process may bring the runtime up more than once: a library that owns
 * its PaRSEC context for the length of a solve, a test driver that runs
 * its cases one context at a time. parsec_fini() must then leave nothing
 * behind that the next parsec_init() trips over.
 *
 * Each round creates a context, runs a few DTD tasks in it, and destroys
 * it. MPI can only be initialized once, so the test initializes it here
 * and keeps it up across the rounds, instead of leaving it to parsec_init().
 */

#include "parsec.h"
#include "parsec/interfaces/dtd/insert_function.h"
#include "parsec/mca/device/device.h"
#include "parsec/utils/debug.h"
#include "parsec/sys/atomic.h"
#include "tests/tests_runtime.h"

#include <stdio.h>
#include <stdlib.h>

#define NB_ROUNDS 3
#define NB_TASKS  100

static int32_t counter;

static int
count_one(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es; (void)this_task;
    (void)parsec_atomic_fetch_inc_int32(&counter);
    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char *argv[])
{
    parsec_context_t *parsec;
    parsec_taskpool_t *tp;
    int rc, ret = 0;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
#endif

    for( int round = 0; round < NB_ROUNDS; round++ ) {
        parsec = parsec_init(2, &argc, &argv);
        if( NULL == parsec ) {
            fprintf(stderr, "Round %d: parsec_init failed\n", round);
            ret = 1;
            break;
        }

        counter = 0;
        tp = parsec_dtd_taskpool_new();
        rc = parsec_context_add_taskpool(parsec, tp);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");

        /* No data: every process inserts and runs all the tasks */
        for( int i = 0; i < NB_TASKS; i++ ) {
            parsec_dtd_insert_task(tp, count_one, 0, PARSEC_DEV_CPU, "count_one",
                                   PARSEC_DTD_ARG_END);
        }

        rc = parsec_taskpool_wait(tp);
        PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");
        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
        parsec_taskpool_free(tp);

        if( NB_TASKS != counter ) {
            fprintf(stderr, "Round %d: %d tasks ran, expected %d\n",
                    round, (int)counter, NB_TASKS);
            ret = 1;
        }

        rc = parsec_fini(&parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_fini");
        printf("Round %d: context created, used and destroyed\n", round);
    }

#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif
    return ret;
}
