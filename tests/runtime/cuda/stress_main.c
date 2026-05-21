/*
 * Copyright (c) 2019-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "stress.h"
#include "stress_wrapper.h"

#if defined(DISTRIBUTED)
#include <mpi.h>
#endif

#include <getopt.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    parsec_context_t *parsec = NULL;
    parsec_taskpool_t *tp;
    int size = 1;
    int rank = 0;
    int tile_size = 1024;
    int depth = 80;
    int ch;

    /* Parse -n (tile size) and -d (depth) before parsec_init */
    while ((ch = getopt(argc, argv, "n:d:")) != -1) {
        switch (ch) {
            case 'n':
                tile_size = atoi(optarg);
                break;
            case 'd':
                depth = atoi(optarg);
                break;
        }
    }

    /* Shift argv in place to remove our options so parsec_init does not see them */
    for (int i = 1; i <= argc - optind; i++) {
        argv[i] = argv[optind + i - 1];
    }
    argc = argc - optind + 1;

#if defined(DISTRIBUTED)
    {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* DISTRIBUTED */

    parsec = parsec_init(-1, &argc, &argv);

    tp = testing_stress_New(parsec, depth, tile_size);
    if( NULL != tp ) {
        parsec_context_add_taskpool(parsec, tp);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        parsec_taskpool_free(tp);
    }

    parsec_fini(&parsec);
#if defined(DISTRIBUTED)
    MPI_Finalize();
#endif /* DISTRIBUTED */
    return 0;
}
