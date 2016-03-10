/*
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "ep_wrapper.h"
#include "schedmicro_data.h"
#include "dague/os-spec-timing.h"
#if defined(DAGUE_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(DAGUE_HAVE_STRING_H) */
#include <math.h>

#define MAXNT   16384
#define MAXLEVEL 1024
#define MAXTRY    100
#define MAX_RELATIVE_STDEV 0.1

double stdev(double sum, double sumsqr, double n)
{
    return sqrt( (sumsqr - ((sum*sum)/n))/(n - 1.0) );
}

int main(int argc, char *argv[])
{
    dague_context_t* dague;
    int rank, world;
    int nt, level, try;
    dague_ddesc_t *ddescA;
    dague_handle_t *ep;
    dague_time_t start, end;
    double sum, sumsqr, val;

#if defined(DAGUE_HAVE_MPI)
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
    dague = dague_init(0, &argc, &argv);
    if( NULL == dague ) {
        exit(-1);
    }
    printf("#All measured values are times. Times are expressed in " TIMER_UNIT "\n");

    level   = 4 * world;

    ddescA = create_and_distribute_data(rank, world, MAXNT, 1);
    dague_ddesc_set_key(ddescA, "A");

    printf("#Embarrasingly Parallel Empty Tasks\n");
    printf("#Level\tNumber of tasks (per level)\tAvg\tStdev\n");
    for( level = 1; level <= MAXLEVEL; level *= 2) {
        for( nt = 1; nt <= MAXNT; nt *= 2 ) {

            sum = 0.0;
            sumsqr = 0.0;
            for(try = 0; try < MAXTRY; try++) {
                if( try > 2 ) {
                    if( stdev(sum, sumsqr, (double)try) / (sum/(double)try) < MAX_RELATIVE_STDEV )
                        break;
                }

                ep = ep_new(ddescA, nt, level);
                dague_enqueue(dague, ep);

                start = take_time();
                dague_context_wait(dague);
                end = take_time();

                ep_destroy(ep);

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

    free_data(ddescA);

    dague_fini(&dague);
#ifdef DAGUE_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
