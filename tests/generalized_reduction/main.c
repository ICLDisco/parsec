/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdlib.h>
#if !defined(_ISOC99_SOURCE)
# define _ISOC99_SOURCE // for using strtol()
#endif
#include "dague.h"
#include "BT_reduction_wrapper.h"
#if defined(DAGUE_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(DAGUE_HAVE_STRING_H) */
#include "reduc_data.h"

int main(int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world, cores;
    int nt, nb;
    tiled_matrix_desc_t *ddescA;
    dague_handle_t *BT_reduction;

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
    cores = 1;
    dague = dague_init(cores, &argc, &argv);

    nb = 1;
    nt = 7;
    if( argc > 1 ){
        nt = (int)strtol(argv[1], NULL, 0);
    }

    ddescA = create_and_distribute_data(rank, world, nb, nt, sizeof(int));
    dague_ddesc_set_key((dague_ddesc_t *)ddescA, "A");

    BT_reduction = BT_reduction_new(ddescA, nb, nt);
    dague_enqueue(dague, BT_reduction);

    dague_context_wait(dague);

    dague_handle_free((dague_handle_t*)BT_reduction);
    free_data(ddescA);

    dague_fini(&dague);

#ifdef DAGUE_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
