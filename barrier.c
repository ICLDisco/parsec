/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "barrier.h"

#if DPLASMA_IMPLEMENT_BARRIERS

int dplasma_barrier_init(dplasma_barrier_t* barrier, const void* attr, unsigned int count)
{
    int rc;

    if( 0 != (rc = pthread_mutex_init(&(barrier->mutex), attr)) ) {
        return rc;
    }

    barrier->count   = count;
    barrier->missing = count;
    if( 0 != (rc = pthread_cond_init(&(barrier->cond), NULL)) ) {
        pthread_mutex_destroy( &(barrier->mutex) );
        return rc;
    }
    return 0;
}

int dplasma_barrier_wait(dplasma_barrier_t* barrier)
{
    pthread_mutex_lock( &(barrier->mutex) );
    if( 0 == --(barrier->missing) ) {
        pthread_cond_broadcast( &(barrier->cond) );
    } else {
        pthread_cond_wait( &(barrier->cond), &(barrier->mutex) );
    }
    barrier->missing++;
    pthread_mutex_unlock( &(barrier->mutex) );
    return 0;
}

int dplasma_barrier_destroy(dplasma_barrier_t* barrier)
{
    pthread_mutex_destroy( &(barrier->mutex) );
    pthread_cond_destroy( &(barrier->cond) );
    return 0;
}

#endif  /* !(defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS - 20012L) >= 0) */
