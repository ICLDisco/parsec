/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "branching_wrapper.h"
#include "branching_data.h"
#if defined(DAGUE_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(DAGUE_HAVE_STRING_H) */

int main(int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world, cores;
    int size, nb;
    dague_ddesc_t *ddescA;
    dague_handle_t *branching;

#if defined(DAGUE_HAVE_MPI)
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
    cores = 8;
    dague = dague_init(cores, &argc, &argv);

    size = 256;
    if(argc != 2) {
        nb   = 2;
    } else {
        nb = atoi(argv[1]);
    }

    ddescA = create_and_distribute_data(rank, world, size);
    dague_ddesc_set_key(ddescA, "A");

    branching = branching_new(ddescA, size, nb);
    if( NULL != branching ) {
        dague_enqueue(dague, branching);

        dague_context_wait(dague);
    }

    free_data(ddescA);

    dague_fini(&dague);

#ifdef DAGUE_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
