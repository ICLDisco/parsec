/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "choice_wrapper.h"
#include "choice_data.h"
#include "dague/data_distribution.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */
#if defined(HAVE_MPI)
#include <mpi.h>
#endif  /* defined(HAVE_MPI) */
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world, cores;
    int size, nb, i, j, c;
    dague_ddesc_t *ddescA;
    int *decision;
    dague_handle_t *choice;
    char **dargv, ***pargv;

#if defined(HAVE_MPI)
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
        if( dargv ) {
            dargv[j++] = argv[i];
        }
    }
    if( !dargv ) {
        dargv = (char**)calloc( 2, sizeof(char *));
        dargv[j++] = strdup(argv[0]);
    }
    dargv[j] = NULL;

    if(argc - j < 1) {
        nb = 2;
    } else {
        nb = atoi(argv[1]);
    }

    cores = 1;
    if(dargv == NULL)
        pargv = NULL;
    else
        pargv = &dargv;
    dague = dague_init(cores, &j, pargv);

    ddescA = create_and_distribute_data(rank, world, size);
    dague_ddesc_set_key(ddescA, "A");

    decision = (int*)calloc(sizeof(int), nb+1);

    choice = choice_new(ddescA, size, decision, nb, world);
    dague_enqueue(dague, choice);

    dague_context_wait(dague);

    choice_destroy(choice);

    dague_fini(&dague);

    for(size = 0; size < world; size++) {
        if( rank == size ) {
            printf("On rank %d, the choices were: ", rank);
            for(i = 0; i <= nb; i++) {
                c = decision[i];
                printf("%c%s", c == 0 ? '#' : (c == 1 ? 'A' : 'B'), i == nb ? "\n" : ", ");
            }
        }
#if defined(HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    free_data(ddescA);
    free(decision);

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
