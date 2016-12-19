/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#include "parsec/class/barrier.h"

#if PARSEC_IMPLEMENT_BARRIERS

int parsec_barrier_init(parsec_barrier_t* barrier, const void* attr, unsigned int count)
{
    int rc;

    if( 0 != (rc = pthread_mutex_init(&(barrier->mutex), attr)) ) {
        return rc;
    }

    barrier->count      = count;
    barrier->curcount   = 0;
    barrier->generation = 0;
    if( 0 != (rc = pthread_cond_init(&(barrier->cond), NULL)) ) {
        pthread_mutex_destroy( &(barrier->mutex) );
        return rc;
    }
    return 0;
}

int parsec_barrier_wait(parsec_barrier_t* barrier)
{
    int generation;

    pthread_mutex_lock( &(barrier->mutex) );
    if( (barrier->curcount + 1) == barrier->count) {
        barrier->generation++;
        barrier->curcount = 0;
        pthread_cond_broadcast( &(barrier->cond) );
        pthread_mutex_unlock( &(barrier->mutex) );
        return 1;
    }
    barrier->curcount++;
    generation = barrier->generation;
    for(;;) {
        pthread_cond_wait( &(barrier->cond), &(barrier->mutex) );
        if( generation != barrier->generation ) {
            break;
        }
    }
    pthread_mutex_unlock( &(barrier->mutex) );
    return 0;
}

int parsec_barrier_destroy(parsec_barrier_t* barrier)
{
    pthread_mutex_destroy( &(barrier->mutex) );
    pthread_cond_destroy( &(barrier->cond) );
    barrier->count    = 0;
    barrier->curcount = 0;
    return 0;
}

#endif  /* !(defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS - 20012L) >= 0) */
