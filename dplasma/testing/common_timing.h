#ifndef TIMING_H
#define TIMING_H

#include "dague_config.h"
#include <stdio.h>
#include <sys/time.h>

extern double time_elapsed;
extern double sync_time_elapsed;

#if defined( HAVE_MPI)
# define get_cur_time() MPI_Wtime()
#else
static inline double get_cur_time(void)
{
    struct timeval tv;
    double t;

    gettimeofday(&tv,NULL);
    t = tv.tv_sec + tv.tv_usec / 1e6;
    return t;
}
#endif

#if defined(DAGUE_PROF_TRACE)
#define DAGUE_PROFILING_START() dague_profiling_start()
#else
#define DAGUE_PROFILING_START()
#endif  /* defined(DAGUE_PROF_TRACE) */

#define TIME_START() do { time_elapsed = get_cur_time(); } while(0)
#define TIME_STOP() do { time_elapsed = get_cur_time() - time_elapsed; } while(0)
#define TIME_PRINT(rank, print) do { \
  TIME_STOP(); \
  printf("[%4d] TIME(s) %12.5f : ", rank, time_elapsed); \
  printf print; \
} while(0)

#ifdef HAVE_MPI
# define SYNC_TIME_START() do {                 \
        MPI_Barrier(MPI_COMM_WORLD);            \
        DAGUE_PROFILING_START();                \
        sync_time_elapsed = get_cur_time();     \
    } while(0)
# define SYNC_TIME_STOP() do {                                  \
        MPI_Barrier(MPI_COMM_WORLD);                            \
        sync_time_elapsed = get_cur_time() - sync_time_elapsed; \
    } while(0)
# define SYNC_TIME_PRINT(rank, print) do {                          \
        SYNC_TIME_STOP();                                           \
        if(0 == rank) {                                             \
            printf("[****] TIME(s) %12.5f : ", sync_time_elapsed);        \
            printf print;                                           \
        }                                                           \
  } while(0)

/* overload exit in MPI mode */
#   define exit(ret) MPI_Abort(MPI_COMM_WORLD, ret)

#else 
# define SYNC_TIME_START() do { sync_time_elapsed = get_cur_time(); } while(0)
# define SYNC_TIME_STOP() do { sync_time_elapsed = get_cur_time() - sync_time_elapsed; } while(0)
# define SYNC_TIME_PRINT(rank, print) do {                           \
        SYNC_TIME_STOP();                                           \
        if(0 == rank) {                                             \
            printf("[****] TIME(s) %12.5f : ", sync_time_elapsed);      \
            printf print;                                           \
        }                                                           \
    } while(0)
#endif

#endif /* TIMING_H */
