/*
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

#include "stress.h"
#include "stress_wrapper.h"

#include <getopt.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    parsec_taskpool_t *tp;
    int tile_size = 1024;
    int depth = 80;
    int ch;
    int rc;

    /* Parse -n (tile size) and -d (depth) before parsec_init */
    while ((ch = getopt(argc, argv, "n:d:")) != -1) {
        switch (ch) {
            case 'n':
                tile_size = atoi(optarg);
                break;
            case 'd':
                depth = atoi(optarg);
                break;
        }
    }

    /* Shift argv in place to remove our options so parsec_init does not see them */
    for (int i = 1; i <= argc - optind; i++) {
        argv[i] = argv[optind + i - 1];
    }
    argc = argc - optind + 1;

    rc = parsec_tests_context_init(-1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &argc, &argv, &parsec, NULL, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    tp = testing_stress_New(parsec, depth, tile_size);
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
