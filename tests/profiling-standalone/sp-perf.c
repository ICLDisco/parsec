/*
 * Copyright (c) 2017-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */
#include "parsec/parsec_config.h"

#include <stdlib.h>
#include <pthread.h>
#ifdef PARSEC_HAVE_PTHREAD_BARRIER_H
/* Mac OS X pthread.h does not provide the pthread_barrier by default */
#include <pthread-barrier.h>
#endif  /* PARSEC_HAVE_PTHREAD_BARRIER_H */
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include <string.h>
#include <stdio.h>
#include "parsec/profiling.h"
#include "parsec/utils/debug.h"
#include "tests/tests_runtime.h"

#if !defined(timersub)
#define timersub(a, b, result) do {                \
        (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;       \
        (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;    \
        if( (result)->tv_usec < 0 ) {                       \
            --(result)->tv_sec;                             \
            (result)->tv_usec += 1000000;                   \
        }                                                   \
    } while(0)
#endif

typedef struct {
    pthread_t                  pthread_id;
    int                        thread_index;
    parsec_profiling_stream_t *prof;
    struct timeval             duration;
    double                     dummy;
} per_thread_info_t;

static int event_startkey, event_endkey;
static pthread_barrier_t barrier;
static uint32_t tasks_per_thread = 100;
static int profiling;

#define D 32
static void cpuburn(double *a, double *b, double *c)
{
    int i, j, k;
    for(i = 0; i < D; i++)
        for(j = 0; j < D; j++)
            for(k = 0; k < D; k++)
                c[i*D+j] += a[i*D+k] * b[k*D+j];
}

static void *run_thread(void *_arg)
{
    per_thread_info_t *ti = (per_thread_info_t*)_arg;
    uint32_t i;
    struct timeval start, end;
    double a[D*D], b[D*D], c[D*D];

    if( profiling )
        ti->prof = parsec_profiling_stream_init(4096, "Thread %d", ti->thread_index);

    pthread_barrier_wait(&barrier); // We wait that all threads have called init

    for(i = 0; i < D*D; i++) {
        a[i] = (double)rand() / RAND_MAX;
        b[i] = (double)rand() / RAND_MAX;
        c[i] = (double)rand() / RAND_MAX;
    }

    pthread_barrier_wait(&barrier); // Then we wait that all threads are ready and that the main thread has called start
    gettimeofday(&start, NULL);

    for(i = 0; i < tasks_per_thread; i++) {
        if(profiling)
            parsec_profiling_trace_flags(ti->prof, event_startkey, i, ti->thread_index, NULL, 0);
        cpuburn(a, b, c);
        if(profiling)
            parsec_profiling_trace_flags(ti->prof, event_endkey, i, ti->thread_index, NULL, 0);
    }

    gettimeofday(&end, NULL);
    timersub(&end, &start, &ti->duration);
    ti->dummy = c[0];

    return NULL;
}

int main(int argc, char *argv[])
{
    int i, opt, rc;
    per_thread_info_t *thread_info;
    int nbthreads = 1;
    char *filename = NULL;
    int rank;
    parsec_context_t *parsec;
    int parsec_argc = 0;
    char **parsec_argv = NULL;

    for(i = 1; i < argc; i++) {
        if( 0 == strcmp(argv[i], "--") ) {
            parsec_argc = argc - i;
            parsec_argv = argv + i;
            argc = i;
            break;
        }
    }

    rc = parsec_tests_context_init(1, PARSEC_TEST_THREAD_SERIALIZED,
                                   &parsec_argc, &parsec_argv,
                                   &parsec, &rank, NULL);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_init");

    while ((opt = getopt(argc, argv, "f:n:N:h?")) != -1) {
        switch (opt) {
        case 'f':
            filename = strdup(optarg);
            break;
        case 'n':
            nbthreads = atoi(optarg);
            break;
        case 'N':
            tasks_per_thread = atoi(optarg);
            break;
        default: /* '?' */
            fprintf(stderr, "Usage: %s [-f filename] [-n number of threads] [-N number of tasks per thread]\n",
                    argv[0]);
            rc = parsec_tests_context_fini(&parsec);
            PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
            exit(EXIT_FAILURE);
        }
    }

    if(NULL == filename) {
        profiling = 0;
    } else {
        profiling = 1;
    }

    if( profiling ) {
        parsec_profiling_init(rank);
        if( parsec_profiling_dbp_start(filename, "PaRSEC profiling system performance evaluation" ) == -1 ) {
            parsec_profiling_fini();
            rc = parsec_tests_context_fini(&parsec);
            PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");
            exit(EXIT_FAILURE);
        }

        parsec_profiling_add_dictionary_keyword("Event", "#FF0000", 0, NULL, &event_startkey, &event_endkey);
    }

    pthread_barrier_init(&barrier, NULL, nbthreads+1);
    thread_info = (per_thread_info_t *)calloc(nbthreads, sizeof(per_thread_info_t));

    for(i = 0; i < nbthreads; i++) {
        thread_info[i].thread_index = i;
        pthread_create(&thread_info[i].pthread_id, NULL, run_thread, &thread_info[i]);
    }

    pthread_barrier_wait(&barrier); // We wait first that all threads have called thread_init
    if(profiling) {
        parsec_profiling_start();
    }
    pthread_barrier_wait(&barrier); // Then we free all compute threads to run

    for(i = 0; i < nbthreads; i++)
        pthread_join(thread_info[i].pthread_id, NULL);

    if( profiling ) {
        parsec_profiling_dbp_dump();
        parsec_profiling_fini();
    }

    for(i = 0; i < nbthreads; i++) {
        fprintf(stderr, "Thread %d Total Time (s): %d.%06d\n", i, (int)thread_info[i].duration.tv_sec, (int)thread_info[i].duration.tv_usec);
    }
    free(thread_info);

    rc = parsec_tests_context_fini(&parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_tests_context_fini");

    exit(EXIT_SUCCESS);
}
