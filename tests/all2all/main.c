/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec.h"
#include "a2a_wrapper.h"
#include "a2a_data.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rc;
    int rank, world, cores;
    int size, repeat;
    parsec_tiled_matrix_dc_t *dcA, *dcB;
    parsec_taskpool_t *a2a;

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
    cores = 1;
    parsec = parsec_init(cores, &argc, &argv);

    size = 256;
    repeat = 10;

    dcA = create_and_distribute_data(rank, world, world*size);
    parsec_data_collection_set_key( (parsec_data_collection_t*)dcA, "A");
    dcB = create_and_distribute_data(rank, world, world*size);
    parsec_data_collection_set_key( (parsec_data_collection_t*)dcB, "B");

    a2a = a2a_new(dcA, dcB, size, repeat);
    rc = parsec_enqueue(parsec, a2a);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free(a2a);
    parsec_fini(&parsec);
    free_data(dcA);
    free_data(dcB);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
