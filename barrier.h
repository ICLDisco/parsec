/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DPLASMA_BARRIER_H_HAS_BEEN_INCLUDED
#define DPLASMA_BARRIER_H_HAS_BEEN_INCLUDED

#include <unistd.h>
#include <pthread.h>

/* The Linux includes are completely screwed up right now. Even if they
 * correctly export a _POSIX_BARRIER define the barrier functions are
 * not correctly defined in the pthread.h. So until we figure out
 * how to correctly identify their availability, we will have to
 * disable them.
 */
#if defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS - 20012L) >= 0 && 0

typedef pthread_barrier_t dplasma_barrier_t;
#define dplasma_barrier_init pthread_barrier_init
#define dplasma_barrier_wait pthread_barrier_wait
#define dplasma_barrier_destroy pthread_barrier_destroy
#define DPLASMA_IMPLEMENT_BARRIERS 0

#else

typedef struct dplasma_barrier_t {
    int                 count;
    int                 missing;
    pthread_mutex_t     mutex;
    pthread_cond_t      cond;
} dplasma_barrier_t;

int dplasma_barrier_init(dplasma_barrier_t*, const void*, unsigned int);
int dplasma_barrier_wait(dplasma_barrier_t*);
int dplasma_barrier_destroy(dplasma_barrier_t*);
#define DPLASMA_IMPLEMENT_BARRIERS 1

#endif


#endif  /* DPLASMA_BARRIER_H_HAS_BEEN_INCLUDED */
