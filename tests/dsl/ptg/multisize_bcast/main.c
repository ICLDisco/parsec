/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdlib.h>

#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "check_multisize_bcast_wrapper.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include "data_gen.h"

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank = 0, world = 1, cores = -1;
    int nt = 2, nb = 1, rc;
    parsec_matrix_block_cyclic_t *dcA;
    parsec_taskpool_t *bcast;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if( argc > 1 ) {
        char* endptr;
        long val = strtol(argv[1], &endptr, 0);
        if( endptr == argv[1] ) {
            printf("Bad argument (found %s instead of the number of tiles)\n", argv[1]);
            exit(-1);
        }
        nt = (int)val;
        if( 0 == nt ) {
            printf("Bad value for nt (cannot be zero) !!!\n");
            exit(-1);
        }
    }

    parsec = parsec_init(cores, &argc, &argv);
    if( NULL == parsec ) {
        exit(1);
    }

    dcA = create_and_distribute_data(rank, world, nb, nt);
    parsec_data_collection_set_key((parsec_data_collection_t *)dcA, "A");

    bcast = check_multisize_bcast_new(dcA, nb, nt);

    rc = parsec_context_add_taskpool(parsec, bcast);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free((parsec_taskpool_t*)bcast);
    free_data(dcA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
