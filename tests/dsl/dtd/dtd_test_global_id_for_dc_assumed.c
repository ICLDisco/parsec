/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

/* parsec things */
#include "parsec/runtime.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>

#include "tests/tests_data.h"
#include "tests/tests_runtime.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rc;
    int rank, world, cores;
    int nb, nt;
    parsec_tiled_matrix_t *dcA, *dcB, *dcC;

    uint32_t id = 0;

    nb = 1; /* tile_size */
    cores = 8;

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    nt = world; /* total no. of tiles */

    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");
    parsec_data_collection_t *A = (parsec_data_collection_t *)dcA;
    parsec_dtd_data_collection_init(A);

    dcB = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcB, "A");
    parsec_data_collection_t *B = (parsec_data_collection_t *)dcB;
    parsec_dtd_data_collection_init(B);

    dcC = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcC, "A");
    parsec_data_collection_t *C = (parsec_data_collection_t *)dcC;
    parsec_dtd_data_collection_init(C);


    /*rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");*/

    printf("rank: %d\tDC id: %"PRIu64"\n", rank, (uint64_t)A->dc_id);
    assert(A->dc_id == id);
    id++;

    printf("rank: %d\tDC id: %"PRIu64"\n", rank, (uint64_t)B->dc_id);
    assert(B->dc_id == id);
    id++;

    printf("rank: %d\tDC id: %"PRIu64"\n", rank, (uint64_t)C->dc_id);
    assert(C->dc_id == id);
    id++;

    parsec_dtd_data_collection_fini(A);
    parsec_dtd_data_collection_fini(B);
    parsec_dtd_data_collection_fini(C);
    free_data(dcA);
    free_data(dcB);
    free_data(dcC);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
