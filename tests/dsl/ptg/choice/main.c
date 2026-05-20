/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2024-2026 NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "choice_wrapper.h"
#include "choice_data.h"
#include "parsec/data_distribution.h"
#include "tests/tests_runtime.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int size, nb, i, c, rc;
    parsec_data_collection_t *dcA;
    int *decision;
    parsec_taskpool_t *choice;

    size = 256;
    int pargc = 0;
    char **pargv = NULL;
    for(i = 0; i < argc; i++) {
        if( strcmp(argv[i], "--") == 0 ) {
            pargc = argc - i;
            pargv = argv + i;
            argc = i;
            break;
        }
    }

    if(argc <= 1) {
        nb = 2;
    } else {
        nb = atoi(argv[1]);
        if( 0 >= nb ) {
            printf("Incorrect argument\n");
            exit(-1);
        }
    }

    rc = parsec_tests_context_init(cores, PARSEC_TEST_THREAD_SERIALIZED,
                                   &pargc, &pargv, &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    dcA = create_and_distribute_data(rank, world, size);
    parsec_data_collection_set_key(dcA, "A");

    decision = (int*)calloc(nb+1, sizeof(int));

    choice = choice_new(dcA, size, decision, nb, world);
    rc = parsec_context_add_taskpool(parsec, choice);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free((parsec_taskpool_t*)choice);

    for(size = 0; size < world; size++) {
        if( rank == size ) {
            printf("On rank %d, the choices were: ", rank);
            for(i = 0; i <= nb; i++) {
                c = decision[i];
                printf("%c%s", c == 0 ? '#' : (c == 1 ? 'A' : 'B'), i == nb ? "\n" : ", ");
            }
        }
        rc = parsec_tests_barrier(parsec);
        if( (PARSEC_SUCCESS != rc) && (PARSEC_ERR_NOT_IMPLEMENTED != rc) ) {
            PARSEC_CHECK_ERROR(rc, "parsec_tests_barrier");
        }
    }

    free_data(dcA);
    free(decision);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
