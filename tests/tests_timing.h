/*
 * Copyright (c) 2021-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
#ifndef TIMING_H
#define TIMING_H

#include "parsec/runtime.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"
#include <stdio.h>
#include <sys/time.h>

extern double time_elapsed;
extern double sync_time_elapsed;

static inline double get_cur_time(void)
{
    struct timeval tv;
    double t;

    gettimeofday(&tv, NULL);
    t = tv.tv_sec + tv.tv_usec / 1e6;
    return t;
}

#if defined(PARSEC_PROF_TRACE)
#define PARSEC_PROFILING_START() parsec_profiling_start()
#else
#define PARSEC_PROFILING_START()
#endif  /* defined(PARSEC_PROF_TRACE) */

#define TIME_START() do { time_elapsed = get_cur_time(); } while(0)
#define TIME_STOP() do { time_elapsed = get_cur_time() - time_elapsed; } while(0)
#define TIME_PRINT(rank, print) do { \
  TIME_STOP(); \
  printf("[%4d] TIME(s) %12.5f : ", rank, time_elapsed); \
  printf print; \
} while(0)

/*
 * Non-MPI communication backends do not expose a test barrier yet.  Keep the
 * timing helpers usable as local timers in that case, but still fail on real
 * barrier errors.
 */
#define SYNC_TIME_BARRIER(parsec_context) do {                               \
        int _parsec_tests_barrier_rc = parsec_tests_barrier(parsec_context); \
        if( PARSEC_ERR_NOT_IMPLEMENTED != _parsec_tests_barrier_rc ) {       \
            PARSEC_CHECK_ERROR(_parsec_tests_barrier_rc, "parsec_tests_barrier"); \
        }                                                                    \
    } while(0)
#define SYNC_TIME_START(parsec_context) do {         \
        SYNC_TIME_BARRIER(parsec_context);           \
        PARSEC_PROFILING_START();                    \
        sync_time_elapsed = get_cur_time();          \
    } while(0)
#define SYNC_TIME_STOP(parsec_context) do {                          \
        SYNC_TIME_BARRIER(parsec_context);                           \
        sync_time_elapsed = get_cur_time() - sync_time_elapsed;      \
    } while(0)
#define SYNC_TIME_PRINT(parsec_context, rank, print) do {            \
        SYNC_TIME_STOP(parsec_context);                              \
        if(0 == rank) {                                              \
            printf("[****] TIME(s) %12.5f : ", sync_time_elapsed);   \
            printf print;                                            \
        }                                                            \
  } while(0)

#endif /* TIMING_H */
