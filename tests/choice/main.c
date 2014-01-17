/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "choice_wrapper.h"
#include "choice_data.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */
#include "dague_prof_grapher.h"

int main(int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world, cores;
    int size, nb, i, c;
    dague_ddesc_t *ddescA;
    int *decision;
    dague_object_t *choice;

#if defined(HAVE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif
    cores = 1;
    dague = dague_init(cores, &argc, &argv);

    size = 256;
    if(argc < 2) {
        nb = 2;
    } else {
        nb = atoi(argv[1]);
    }

    ddescA = create_and_distribute_data(rank, world, cores, size);
    dague_ddesc_set_key(ddescA, "A");

    decision = (int*)calloc(sizeof(int), nb+1);

    choice = choice_new(ddescA, size, decision, nb, world);
    dague_enqueue(dague, choice);

    dague_progress(dague);

    choice_destroy(choice);

#if defined(DAGUE_PROF_TRACE)
    {
        char *pname;
        asprintf(&pname, "choice-%d.profile", rank);
        dague_profiling_dump_dbp(pname);
        free(pname);
    }
#endif

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
