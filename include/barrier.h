/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_BARRIER_H_HAS_BEEN_INCLUDED
#define DAGUE_BARRIER_H_HAS_BEEN_INCLUDED

#include <unistd.h>
#include <pthread.h>

/* The Linux includes are completely screwed up right now. Even if they
 * correctly export a _POSIX_BARRIER define the barrier functions are
 * not correctly defined in the pthread.h. So until we figure out
 * how to correctly identify their availability, we will have to
 * disable them.
 */
#if defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS - 20012L) >= 0 && 0

BEGIN_C_DECLS

typedef pthread_barrier_t dague_barrier_t;
#define dague_barrier_init pthread_barrier_init
#define dague_barrier_wait pthread_barrier_wait
#define dague_barrier_destroy pthread_barrier_destroy
#define DAGUE_IMPLEMENT_BARRIERS 0

#else

typedef struct dague_barrier_t {
    int                 count;
    volatile int        curcount;
    volatile int        generation;
    pthread_mutex_t     mutex;
    pthread_cond_t      cond;
} dague_barrier_t;

int dague_barrier_init(dague_barrier_t *barrier, const void *pthread_mutex_attr, unsigned int count);
int dague_barrier_wait(dague_barrier_t*);
int dague_barrier_destroy(dague_barrier_t*);
#define DAGUE_IMPLEMENT_BARRIERS 1

END_C_DECLS

#endif


#endif  /* DAGUE_BARRIER_H_HAS_BEEN_INCLUDED */
