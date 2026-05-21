/*
 * Copyright (c) 2013-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include <stdio.h>
#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "ep_wrapper.h"
#include "schedmicro_data.h"
#include "parsec/os-spec-timing.h"
#include "tests/tests_runtime.h"
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#include <math.h>

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

    rc = parsec_tests_context_init(0, PARSEC_TEST_THREAD_SERIALIZED,
                                   &parsec_argc, &parsec_argv,
                                   &parsec, &rank, &world);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    printf("#All measured values are times. Times are expressed in " TIMER_UNIT "\n");

    level   = 4 * world;

    dcA = create_and_distribute_data(rank, world, MAXNT, 1);
    parsec_data_collection_set_key(dcA, "A");

    printf("#Embarrassingly Parallel Empty Tasks\n");
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

                ep_free(ep);

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

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    return 0;
}
