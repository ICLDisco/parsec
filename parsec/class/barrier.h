/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_BARRIER_H_HAS_BEEN_INCLUDED
#define PARSEC_BARRIER_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"

#include <unistd.h>
#include <pthread.h>

/* The Linux includes are completely screwed up right now. Even if they
 * correctly export a _POSIX_BARRIER define the barrier functions are
 * not correctly defined in the pthread.h. So until we figure out
 * how to correctly identify their availability, we will have to
 * disable them.
 */
BEGIN_C_DECLS

#if defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS - 20012L) >= 0 && 0

typedef pthread_barrier_t parsec_barrier_t;
#define parsec_barrier_init pthread_barrier_init
#define parsec_barrier_wait pthread_barrier_wait
#define parsec_barrier_destroy pthread_barrier_destroy
#define PARSEC_IMPLEMENT_BARRIERS 0

#else

typedef struct parsec_barrier_t {
    int                 count;
    volatile int        curcount;
    volatile int        generation;
    pthread_mutex_t     mutex;
    pthread_cond_t      cond;
} parsec_barrier_t;

int parsec_barrier_init(parsec_barrier_t *barrier, const void *pthread_mutex_attr, unsigned int count);
int parsec_barrier_wait(parsec_barrier_t*);
int parsec_barrier_destroy(parsec_barrier_t*);
#define PARSEC_IMPLEMENT_BARRIERS 1

#endif

END_C_DECLS

#endif  /* PARSEC_BARRIER_H_HAS_BEEN_INCLUDED */
