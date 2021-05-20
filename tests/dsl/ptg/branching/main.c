/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
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
    int rank, world, cores = -1;
    int size, nb, rc;
    parsec_data_collection_t *dcA;
    parsec_taskpool_t *branching;

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
    parsec = parsec_init(cores, &argc, &argv);

    size = 256;
    if(argc != 2) {
        nb   = 2;
    } else {
        nb = atoi(argv[1]);
    }

    dcA = create_and_distribute_data(rank, world, size);
    parsec_data_collection_set_key(dcA, "A");

    branching = branching_new(dcA, size, nb);
    if( NULL != branching ) {
        rc = parsec_context_add_taskpool(parsec, branching);
        PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

        rc = parsec_context_start(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_start");

        rc = parsec_context_wait(parsec);
        PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
    }

    free_data(dcA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
