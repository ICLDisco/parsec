/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "choice_wrapper.h"
#include "choice_data.h"
#include "parsec/data_distribution.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */
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

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

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

    parsec = parsec_init(cores, &pargc, &pargv);
    if( NULL == parsec ) {
        exit(-1);
    }
    dcA = create_and_distribute_data(rank, world, size);
    parsec_data_collection_set_key(dcA, "A");

    decision = (int*)calloc(sizeof(int), nb+1);

    choice = choice_new(dcA, size, decision, nb, world);
    rc = parsec_context_add_taskpool(parsec, choice);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free((parsec_taskpool_t*)choice);

    parsec_fini(&parsec);

    for(size = 0; size < world; size++) {
        if( rank == size ) {
            printf("On rank %d, the choices were: ", rank);
            for(i = 0; i <= nb; i++) {
                c = decision[i];
                printf("%c%s", c == 0 ? '#' : (c == 1 ? 'A' : 'B'), i == nb ? "\n" : ", ");
            }
        }
#if defined(PARSEC_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    free_data(dcA);
    free(decision);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
