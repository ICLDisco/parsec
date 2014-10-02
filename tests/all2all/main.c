/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "a2a_wrapper.h"
#include "a2a_data.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */

int main(int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world, cores;
    int size, repeat;
    tiled_matrix_desc_t *ddescA, *ddescB;
    dague_handle_t *a2a;

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
    cores = 1;
    dague = dague_init(cores, &argc, &argv);

    size = 256;
    repeat = 10;

    ddescA = create_and_distribute_data(rank, world, world*size);
    dague_ddesc_set_key( (dague_ddesc_t*)ddescA, "A");
    ddescB = create_and_distribute_data(rank, world, world*size);
    dague_ddesc_set_key( (dague_ddesc_t*)ddescB, "B");

    a2a = a2a_new(ddescA, ddescB, size, repeat);
    dague_enqueue(dague, a2a);

    dague_progress(dague);

    dague_handle_free(a2a);
    dague_fini(&dague);
    free_data(ddescA);
    free_data(ddescB);

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
