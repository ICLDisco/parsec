/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec.h"
#include "rtt_wrapper.h"
#include "rtt_data.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rc;
    int rank, world;
    int size, nb;
    parsec_ddesc_t *ddescA;
    parsec_handle_t *rtt;

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

    parsec = parsec_init(-1, &argc, &argv);

    size = 256;

    ddescA = create_and_distribute_data(rank, world, size);
    parsec_ddesc_set_key(ddescA, "A");

    nb   = 4 * world;
    rtt = rtt_new(ddescA, size, nb);
    rc = parsec_enqueue(parsec, rtt);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_handle_free((parsec_handle_t*)rtt);

    free_data(ddescA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
