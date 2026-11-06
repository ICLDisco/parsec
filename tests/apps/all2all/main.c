/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"
#include "a2a_wrapper.h"
#include "a2a_data.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int size, repeat, rc;
    parsec_tiled_matrix_t *dcA, *dcB;
    parsec_taskpool_t *a2a;

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    size = 256;
    repeat = 10;

    dcA = create_and_distribute_data(rank, world, world*size);
    parsec_data_collection_set_key( (parsec_data_collection_t*)dcA, "A");
    dcB = create_and_distribute_data(rank, world, world*size);
    parsec_data_collection_set_key( (parsec_data_collection_t*)dcB, "B");

    a2a = a2a_new(dcA, dcB, size, repeat);
    rc = parsec_context_add_taskpool(parsec, a2a);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    a2a_free(a2a);
    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
    free_data(dcA);
    free_data(dcB);

    return 0;
}
