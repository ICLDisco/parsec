/*
 * Copyright (c) 2013-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "ep_wrapper.h"
#include "schedmicro_data.h"
#include "parsec/os-spec-timing.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include <math.h>
#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

static int MAXNT                 = 16384;
static int MAXLEVEL              =  1024;
static int MAXTRY                =   1;
static double MAX_RELATIVE_STDEV =   0.1;

double stdev(double sum, double sumsqr, double n)
{
    return sqrt( (sumsqr - ((sum*sum)/n))/(n - 1.0) );
}

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rank, world, rc;
    int nt, level, try;
    parsec_data_collection_t *dcA;
    parsec_taskpool_t *ep;
    parsec_time_t start, end;
    double sum, sumsqr, val;
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
            parsec_argv = argv + a;
            break;
        }
        if(strcmp(argv[a], "-t") == 0) {
            a++;
            MAXTRY = atoi(argv[a]);
            continue;
        }
        if(strcmp(argv[a], "-l") == 0) {
            a++;
            MAXLEVEL = atoi(argv[a]);
            continue;
        }
        if(strcmp(argv[a], "-n") == 0) {
            a++;
            MAXNT = atoi(argv[a]);
            continue;
        }
        if(strcmp(argv[a], "-s") == 0) {
            a++;
            MAX_RELATIVE_STDEV = atof(argv[a]);
            continue;
        }
        fprintf(stderr, "Usage: %s [-t MAXTRY] [-l MAXLEVEL] [-n MAXNT] [-s MAX_RELATIVE_STDEV] [-- <parsec parameters]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    parsec = parsec_init(0, &parsec_argc, &parsec_argv);
    if( NULL == parsec ) {
        exit(-1);
    }
    printf("#All measured values are times. Times are expressed in " TIMER_UNIT "\n");

    level   = 4 * world;

    dcA = create_and_distribute_data(rank, world, MAXNT, 1);
    parsec_data_collection_set_key(dcA, "A");

    printf("#Embarrasingly Parallel Empty Tasks\n");
    printf("#Level\tNumber of tasks (per level)\tAvg\tStdev\n");
    for( level = 1; level <= MAXLEVEL; level *= 2) {
        for( nt = 1; nt <= MAXNT; nt *= 2 ) {

            sum = 0.0;
            sumsqr = 0.0;
            for(try = 0; try < MAXTRY; try++) {
#if 0
                if( try > 2 ) {
                    if( stdev(sum, sumsqr, (double)try) / (sum/(double)try) < MAX_RELATIVE_STDEV )
                        break;
                }
#endif
                ep = ep_new(dcA, nt, level);
                rc = parsec_context_add_taskpool(parsec, ep);
                PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

                rc = parsec_context_start(parsec);
                PARSEC_CHECK_ERROR(rc, "parsec_context_start");

                start = take_time();
                rc = parsec_context_wait(parsec);
                end = take_time();
                PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

                parsec_taskpool_free(ep);

                val = (double)diff_time(start, end);
                sum = sum + val;
                sumsqr = sumsqr + val*val;
            }

            printf("%6d\t%25d\t%g\t%g\n", level, nt, sum / (double)try, stdev(sum, sumsqr, (double)try) );
        }
        printf("\n");
    }
    printf("\n"
           "\n");

    free_data(dcA);

    parsec_fini(&parsec);
#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
