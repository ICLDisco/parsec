/*
 * Copyright (c) 2020-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"

#include "stage_custom.h"
parsec_taskpool_t* testing_stage_custom_New( parsec_context_t *ctx, int M, int N, int MB, int NB, int P, int *ret);

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    parsec_taskpool_t *tp;
    int size = 1;
    int M;
    int N;
    int MB;
    int NB;
    int P = 1;
    int ret = 0;
    int rc;

    /* Initialize PaRSEC */
    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, NULL, &size);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    /* can the test run? */
    assert(size == 1);
    int nb_gpus = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_DEVICES, PARSEC_DEV_CUDA);
    assert(nb_gpus >= 0);
    if(nb_gpus == 0) {
        parsec_warning("This test can only run if at least one GPU device is present");
        printf("TEST SKIPPED\n");
        rc = parsec_tests_context_fini(&parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
        return -PARSEC_ERR_DEVICE;
    }

    /* Test: comparing results when:
        - tile matrix transferred to GPU with default stage_in/stage_out
        - lapack matrix transferred to GPU with custom stage_in/stage_out */

    MB = NB = 1;
    M = N = 1;
    tp = testing_stage_custom_New(parsec, M, N, MB, NB, P, &ret);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        parsec_taskpool_free(tp);
    }

    MB = NB = 1;
    M = N = 10;
    tp = testing_stage_custom_New(parsec, M, N, MB, NB, P, &ret);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        parsec_taskpool_free(tp);
    }

    MB = NB = 4;
    M = N = 20;
    tp = testing_stage_custom_New(parsec, M, N, MB, NB, P, &ret);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        parsec_taskpool_free(tp);
    }

    MB = NB = 40;
    M = N = 240;
    tp = testing_stage_custom_New(parsec, M, N, MB, NB, P, &ret);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        parsec_taskpool_free(tp);
    }

    if( ret != 0) {
        printf("TEST FAILED (%d errors)\n", ret);
    } else {
        printf("TEST PASSED\n");
    }

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return (0 == ret)? EXIT_SUCCESS: EXIT_FAILURE;
}
