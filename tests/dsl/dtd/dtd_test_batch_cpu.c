/*
 * Copyright (c) 2024      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/mca/device/device.h"
#include "tests/tests_runtime.h"

#include <assert.h>
#include <stdlib.h>

static int32_t observed_sum = 0;

static parsec_hook_return_t
batch_cpu_task(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    int value;

    parsec_dtd_unpack_args(this_task, &value);
    (void)parsec_atomic_fetch_add_int32(&observed_sum, value);

    (void)es;
    return PARSEC_HOOK_RETURN_DONE;
}

int
main(int argc, char **argv)
{
    parsec_context_t *parsec;
    parsec_taskpool_t *dtd_tp;
    parsec_task_class_t *tc;
    int rc, rank = 0;
    int ntasks = 32;
    int expected = 0;
    int ret = 0;

    if( NULL != argv[1] ) {
        ntasks = atoi(argv[1]);
    }
    if( ntasks <= 0 ) {
        ntasks = 32;
    }

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, &rank, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    dtd_tp = parsec_dtd_taskpool_new();

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_add_taskpool(parsec, dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    tc = parsec_dtd_create_task_class(dtd_tp, "BatchCpu",
                                      sizeof(int), PARSEC_VALUE,
                                      PARSEC_DTD_ARG_END);
    assert(NULL != tc);

    rc = parsec_dtd_task_class_add_chore(dtd_tp, tc,
                                         PARSEC_DEV_CPU | PARSEC_DEV_CHORE_ALLOW_BATCH,
                                         batch_cpu_task);
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_task_class_add_chore");
    assert(tc->incarnations[0].type & PARSEC_DEV_CPU);
    assert(tc->incarnations[0].type & PARSEC_DEV_CHORE_ALLOW_BATCH);

    for( int i = 0; i < ntasks; i++ ) {
        int value = i + 1;
        expected += value;
        parsec_dtd_insert_task_with_task_class(dtd_tp, tc, 0,
                                               PARSEC_DEV_CPU,
                                               PARSEC_DTD_EMPTY_FLAG, &value,
                                               PARSEC_DTD_ARG_END);
    }

    rc = parsec_taskpool_wait(dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    if( expected != observed_sum ) {
        parsec_warning("Rank %d observed %d, expected %d\n",
                       rank, observed_sum, expected);
        ret = 1;
    }

    parsec_dtd_task_class_release(dtd_tp, tc);
    parsec_taskpool_free(dtd_tp);
    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return ret;
}
