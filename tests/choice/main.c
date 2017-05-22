/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec.h"
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
    int rc;
    int rank, world, cores;
    int size, nb, i, j, c;
    parsec_ddesc_t *ddescA;
    int *decision;
    parsec_taskpool_t *choice;
    char **dargv, ***pargv;

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
    dargv = NULL;
    j = 0;
    for(i = 0; i < argc; i++) {
        if( strcmp(argv[i], "--") == 0 ) {
            dargv = (char**)calloc( (argc-i+2), sizeof(char *));
            dargv[j++] = strdup(argv[0]);
            continue;
        }
        if( NULL != dargv ) {
            dargv[j++] = argv[i];
        }
    }
    if( NULL == dargv ) {
        dargv = (char**)calloc( 2, sizeof(char *));
        dargv[j++] = strdup(argv[0]);
    }
    dargv[j] = NULL;

    if(argc - j <= 1) {
        nb = 2;
    } else {
        nb = atoi(argv[1]);
        if( 0 >= nb ) {
            printf("Incorrect argument\n");
            exit(-1);
        }
    }

    cores = 1;
    pargv = &dargv;
    parsec = parsec_init(cores, &j, pargv);
    if( NULL == parsec ) {
        exit(-1);
    }
    ddescA = create_and_distribute_data(rank, world, size);
    parsec_ddesc_set_key(ddescA, "A");

    decision = (int*)calloc(sizeof(int), nb+1);

    choice = choice_new(ddescA, size, decision, nb, world);
    rc = parsec_enqueue(parsec, choice);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    choice_destroy(choice);

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

    free_data(ddescA);
    free(decision);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
