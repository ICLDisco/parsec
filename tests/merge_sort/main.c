/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdlib.h>
#if !defined(_ISOC99_SOURCE)
# define _ISOC99_SOURCE // for using strtol()
#endif
#include "parsec.h"
#include "merge_sort_wrapper.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include "sort_data.h"

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, cores;
    int nt, nb;
    tiled_matrix_desc_t *ddescA;
    parsec_handle_t *msort;

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

    nb = 5;
    nt = 1234;
    if( argc > 1 ){
        nt = (int)strtol(argv[1], NULL, 0);
    }

    ddescA = create_and_distribute_data(rank, world, nb, nt, sizeof(int));
    parsec_ddesc_set_key((parsec_ddesc_t *)ddescA, "A");

    msort = merge_sort_new(ddescA, nb, nt);
    parsec_enqueue(parsec, msort);

    parsec_context_wait(parsec);

    parsec_handle_free((parsec_handle_t*)msort);
    free_data(ddescA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
