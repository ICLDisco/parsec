/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DPLASMA_BARRIER_H_HAS_BEEN_INCLUDED
#define DPLASMA_BARRIER_H_HAS_BEEN_INCLUDED

#include <unistd.h>
#include <pthread.h>

#if defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS - 20012L) >= 0

typedef pthread_barrier_t dplasma_barrier_t;
#define dplasma_barrier_init pthread_barrier_init
#define dplasma_barrier_wait pthread_barrier_wait
#define dplasma_barrier_destroy pthread_barrier_destroy

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

#endif


#endif  /* DPLASMA_BARRIER_H_HAS_BEEN_INCLUDED */
