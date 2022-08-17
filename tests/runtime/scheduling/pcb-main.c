/*
 * Copyright (c) 2013-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "pcb_wrapper.h"
#include "rt_data.h"
#include "parsec/os-spec-timing.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include <math.h>
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, rc;
    int nt = 64, level = 8;
    parsec_data_collection_t *dcA;
    parsec_taskpool_t *pcb;
    int parsec_argc = 0;
    char **parsec_argv = NULL;

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
    for(int a = 1; a < argc; a++) {
        if(strcmp(argv[a], "--") == 0) {
            parsec_argc = argc - a;
            parsec_argv = &argv[a];
            break;
        }
        if(strcmp(argv[a], "-t") == 0) {
            a++;
            nt = atoi(argv[a]);
            continue;
        }
        if(strcmp(argv[a], "-l") == 0) {
            a++;
            level = atoi(argv[a]);
            continue;
        }
        fprintf(stderr, "Usage: %s [-t NT] [-l LEVEL] [-- <parsec parameters]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    parsec = parsec_init(0, &parsec_argc, &parsec_argv);
    if( NULL == parsec ) {
        exit(-1);
    }

    dcA = create_and_distribute_data(rank, world, nt, 1);
    parsec_data_collection_set_key(dcA, "A");

    pcb = pcb_new(dcA, nt, level);
    rc = parsec_context_add_taskpool(parsec, pcb);
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free(pcb);

    free_data(dcA);

    parsec_fini(&parsec);
#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
