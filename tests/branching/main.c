/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "branching_wrapper.h"
#include "branching_data.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, cores;
    int size, nb;
    parsec_ddesc_t *ddescA;
    parsec_handle_t *branching;

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
    cores = 8;
    parsec = parsec_init(cores, &argc, &argv);

    size = 256;
    if(argc != 2) {
        nb   = 2;
    } else {
        nb = atoi(argv[1]);
    }

    ddescA = create_and_distribute_data(rank, world, size);
    parsec_ddesc_set_key(ddescA, "A");

    branching = branching_new(ddescA, size, nb);
    if( NULL != branching ) {
        parsec_enqueue(parsec, branching);

        parsec_context_wait(parsec);
    }

    free_data(ddescA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
