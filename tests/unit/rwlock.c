/*
 * Copyright (c) 2017-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include "parsec/sys/atomic.h"
#include "parsec/class/parsec_rwlock.h"
#include "parsec/class/barrier.h"
#include "parsec/bindthread.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/os-spec-timing.h"

#define NB_TESTS 10
#define NB_LOOPS 300000
#define ARRAY_SIZE 12

static parsec_atomic_rwlock_t rwlock = {0};
static int large_array[ARRAY_SIZE] = {0};

static parsec_barrier_t barrier;

static void *do_test(void *param)
{
    int id = (int)(intptr_t)param;
    int before = (73*id+17) % NB_TESTS;
    int after  = NB_TESTS-before;
    int i, j, v, l;
    parsec_time_t t0, t1;
    uint64_t duration;

    parsec_bindthread(id, 0);
    parsec_barrier_wait(&barrier);
    t0 = take_time();

    for(l = 0; l < NB_LOOPS; l++) {
        for(i = 0; i < before; i++) {
            parsec_atomic_rwlock_rdlock(&rwlock);
            for(j = 1; j < ARRAY_SIZE; j++) {
                if( large_array[j-1] != large_array[j] ) {
                    raise(SIGABRT);
                }
            }
            parsec_atomic_rwlock_rdunlock(&rwlock);
        }
        parsec_atomic_rwlock_wrlock(&rwlock);
        v = large_array[0];
        if( large_array[0] != v )
            raise(SIGABRT);
        large_array[0] = v+1;
        for(j = 1; j < ARRAY_SIZE; j++) {
            if( large_array[j] != v ) {
                raise(SIGABRT);
            }
            large_array[j] = v+1;
        }
        parsec_atomic_rwlock_wrunlock(&rwlock);
        for(i = 0; i < after; i++) {
            parsec_atomic_rwlock_rdlock(&rwlock);
            for(j = 1; j < ARRAY_SIZE; j++) {
                if( large_array[j-1] != large_array[j] ) {
                    raise(SIGABRT);
                }
            }
            parsec_atomic_rwlock_rdunlock(&rwlock);
        }
    }
    t1 = take_time();
    duration = diff_time(t0, t1);

    return (void*)(uintptr_t)duration;
}

int main(int argc, char *argv[])
{
    pthread_t *threads;
    int ch, e, minthreads = 0, maxthreads = 0, nbthreads;
    char *m;
    uint64_t maxtime;
    void *retval;

    parsec_hwloc_init();

    while( (ch = getopt(argc, argv, "c:m:M:h?")) != -1 ) {
        switch(ch) {
        case 'c':
            ch = strtol(optarg, &m, 0);
            if( (ch < 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -c value");
            }
            minthreads = maxthreads = (uintptr_t)ch;
            break;
        case 'm':
            ch = strtol(optarg, &m, 0);
            if( (ch < 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -m value");
            }
            minthreads = (uintptr_t)ch;
            break;
        case 'M':
            ch = strtol(optarg, &m, 0);
            if( (ch < 0) || (m[0] != '\0') ) {
                fprintf(stderr, argv[0], "invalid -M value");
            }
            maxthreads = (uintptr_t)ch;
            break;
        case 'h':
        case '?':
        default:
            fprintf(stderr, "Usage: %s [-c nbthreads|-m minthreads -M maxthreads]\n", argv[0]);
            exit(1);
            break;
        }
    }

    if( (maxthreads < minthreads) ) {
        fprintf(stderr, "Usage: %s [-c nbthreads|-m minthreads -M maxthreads]\n", argv[0]);
        exit(1);
    }

    parsec_atomic_rwlock_init(&rwlock);

    threads = (pthread_t*)calloc(sizeof(pthread_t), maxthreads);

    for( nbthreads = minthreads; nbthreads < maxthreads; nbthreads++) {
        parsec_barrier_init(&barrier, NULL, nbthreads+1);

        for(e = 0; e < nbthreads; e++) {
            pthread_create(&threads[e], NULL, do_test, (void*)(intptr_t)e);
        }
        maxtime = (uint64_t)do_test((void*)(intptr_t)(nbthreads+1));
        for(e = 0; e < nbthreads; e++) {
            pthread_join(threads[e], &retval);
            if( (uint64_t)retval > maxtime )
                maxtime = (uint64_t)retval;
        }
        printf("%d threads %"PRIu64" "TIMER_UNIT"\n", nbthreads+1, maxtime);
        fflush(stdout);
    }
}
