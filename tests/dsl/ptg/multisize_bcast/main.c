/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include <stdlib.h>

#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"
#include "check_multisize_bcast_wrapper.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include "data_gen.h"

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank = 0, world = 1, cores = -1;
    int nt = 2, nb = 1, rc;
    parsec_matrix_block_cyclic_t *dcA;
    parsec_taskpool_t *bcast;

    if( argc > 1 ) {
        char* endptr;
        long val = strtol(argv[1], &endptr, 0);
        if( endptr == argv[1] ) {
            printf("Bad argument (found %s instead of the number of tiles)\n", argv[1]);
            exit(-1);
        }
        nt = (int)val;
        if( 0 == nt ) {
            printf("Bad value for nt (cannot be zero) !!!\n");
            exit(-1);
        }
    }

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    bcast = check_multisize_bcast_new(dcA, nb, nt);

    rc = parsec_context_add_taskpool(parsec, bcast);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free((parsec_taskpool_t*)bcast);
    free_data(dcA);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
