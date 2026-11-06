/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"
#include "branching_wrapper.h"
#include "branching_data.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

volatile int32_t nb_taskA = 0;
volatile int32_t nb_taskB = 0;
volatile int32_t nb_taskC = 0;

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, cores = 1;
    int size, nb, rc;
    parsec_data_collection_t *dcA;
    parsec_taskpool_t *branching;

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    size = 256;
    if(argc != 2) {
        nb   = 10;
    } else {
        nb = atoi(argv[1]);
    }

    dcA = create_and_distribute_data(rank, world, size, nb);
    parsec_data_collection_set_key(dcA, "A");

    branching = branching_new(dcA, size, nb);
    if( NULL != branching ) {
        rc = parsec_context_add_taskpool(parsec, branching);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");

        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

        parsec_taskpool_free(branching);
    }

    free_data(dcA);

    int gnbA = nb_taskA, gnbB = nb_taskB, gnbC = nb_taskC;
    rc = parsec_tests_allreduce(parsec, NULL, &gnbA, 1,
                                parsec_datatype_int_t, PARSEC_TESTS_REDUCE_SUM);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_allreduce");
    rc = parsec_tests_allreduce(parsec, NULL, &gnbB, 1,
                                parsec_datatype_int_t, PARSEC_TESTS_REDUCE_SUM);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_allreduce");
    rc = parsec_tests_allreduce(parsec, NULL, &gnbC, 1,
                                parsec_datatype_int_t, PARSEC_TESTS_REDUCE_SUM);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_allreduce");

    printf("nb = %d, nb_taskA = %d, nb_taskB = %d, nb_taskC = %d -- %s\n", nb, 
           gnbA, gnbB, gnbC,
           gnbA == nb && gnbB == 2*nb && gnbC == nb ? "SUCCESS" : "FAILURE!");

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    if( gnbA == nb &&
        gnbB == 2*nb &&
        gnbC == nb )
        return EXIT_SUCCESS;
    return EXIT_FAILURE;
}
