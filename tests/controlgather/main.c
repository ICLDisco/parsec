/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "ctlgat_wrapper.h"
#include "ctlgat_data.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */

int main(int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world, cores;
    int size, nb;
    dague_ddesc_t *ddescA;
    dague_handle_t *ctlgat;

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
    cores = 8;
    dague = dague_init(cores, &argc, &argv);

    size = 256;
    nb   = 4 * world;

    ddescA = create_and_distribute_data(rank, world, size, 1);
    dague_ddesc_set_key(ddescA, "A");

    ctlgat = ctlgat_new(ddescA, size, nb);
    dague_enqueue(dague, ctlgat);

    dague_progress(dague);

    free_data(ddescA);

    dague_fini(&dague);
#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
