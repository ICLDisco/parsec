/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdlib.h>
#if !defined(_ISOC99_SOURCE)
# define _ISOC99_SOURCE // for using strtol()
#endif
#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "merge_sort_wrapper.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include "sort_data.h"

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, cores = -1;
    int nt, nb, rc;
    parsec_tiled_matrix_dc_t *dcA;
    parsec_taskpool_t *msort;

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

    nb = 5;
    nt = 1234;
    if( argc > 1 ){
        nt = (int)strtol(argv[1], NULL, 0);
    }

    dcA = create_and_distribute_data(rank, world, nb, nt, sizeof(int));
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    msort = merge_sort_new(dcA, nb, nt);

    rc = parsec_context_add_taskpool(parsec, msort);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free((parsec_taskpool_t*)msort);
    free_data(dcA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
