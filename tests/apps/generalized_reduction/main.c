/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include <stdlib.h>
#if !defined(_ISOC99_SOURCE)
# define _ISOC99_SOURCE // for using strtol()
#endif
#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"
#include "BT_reduction_wrapper.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include "reduc_data.h"

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int nt, nb, rc;
    parsec_tiled_matrix_t *dcA;
    parsec_taskpool_t *BT_reduction;

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    nb = 1;
    nt = 7;
    if( argc > 1 ){
        nt = (int)strtol(argv[1], NULL, 0);
    }

    dcA = create_and_distribute_data(rank, world, nb, nt, sizeof(int));
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    BT_reduction = BT_reduction_new(dcA, nb, nt);
    rc = parsec_context_add_taskpool(parsec, BT_reduction);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free((parsec_taskpool_t*)BT_reduction);
    free_data(dcA);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
