/*
 * Copyright (c) 2024      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
#include "parsec.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rc;

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, NULL, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();
    parsec_taskpool_free( dtd_tp );
    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
