/**
 * Copyright (c) 2019-2024 The University of Tennessee and The University
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

#include "nvlink.h"
#include "nvlink_wrapper.h"

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    parsec_taskpool_t *tp;
    int rc;

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, NULL, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    /* can the test run? */
    int nb_gpus = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_DEVICES, PARSEC_DEV_CUDA);
    assert(nb_gpus >= 0);
    if(nb_gpus == 0) {
        parsec_warning("This test can only run if at least one GPU device is present");
        rc = parsec_tests_context_fini(&parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
        return -PARSEC_ERR_DEVICE;
    }
    int full_peer_access = parsec_context_query(parsec, PARSEC_CONTEXT_QUERY_DEVICES_FULL_PEER_ACCESS, PARSEC_DEV_CUDA);
    assert(full_peer_access >= 0);
    if(0 == full_peer_access) {
        parsec_warning("This system does not have a full peer access matrix between all GPU devices");
        rc = parsec_tests_context_fini(&parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
        return -PARSEC_ERR_DEVICE;
    }

    tp = testing_nvlink_New(parsec, 10, 512);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        parsec_taskpool_free(tp);
    }

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
    return 0;
}
