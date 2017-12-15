/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/class/barrier.h"
#include "parsec/sys/atomic.h"
#include "thread/thread.h"

#if PARSEC_IMPLEMENT_BARRIERS

int parsec_barrier_init(parsec_barrier_t* barrier, const void* attr, unsigned int count)
{
    int rc;

    if( 0 != (rc = PARSEC_THREAD_MUTEX_CREATE(&(barrier->mutex), attr)) ) {
        return rc;
    }

    barrier->count      = count;
    barrier->curcount   = 0;
    barrier->generation = 0;
    if( 0 != (rc = PARSEC_THREAD_COND_CREATE(&(barrier->cond), NULL)) ) {
        PARSEC_THREAD_MUTEX_DESTROY( &(barrier->mutex) );
        return rc;
    }
    return 0;
}

int parsec_barrier_wait(parsec_barrier_t* barrier)
{
    int generation;
    parsec_mfence();

    PARSEC_THREAD_MUTEX_LOCK( &(barrier->mutex) );
    if( (barrier->curcount + 1) == barrier->count) {
        barrier->generation++;
        barrier->curcount = 0;
        PARSEC_THREAD_COND_BROADCAST( &(barrier->cond) );
        PARSEC_THREAD_MUTEX_UNLOCK( &(barrier->mutex) );
        return 1;
    }
    barrier->curcount++;
    generation = barrier->generation;
    for(;;) {
        PARSEC_THREAD_COND_WAIT( &(barrier->cond), &(barrier->mutex) );
        if( generation != barrier->generation ) {
            break;
        }
    }
    PARSEC_THREAD_MUTEX_UNLOCK( &(barrier->mutex) );
    return 0;
}

int parsec_barrier_destroy(parsec_barrier_t* barrier)
{
    PARSEC_THREAD_MUTEX_DESTROY( &(barrier->mutex) );
    PARSEC_THREAD_COND_DESTROY( &(barrier->cond) );
    barrier->count    = 0;
    barrier->curcount = 0;
    return 0;
}

#endif  /* !(defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS - 20012L) >= 0) */
