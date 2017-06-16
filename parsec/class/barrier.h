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

/**
 * @defgroup parsec_internal_classes_barrier Barrier
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief Synchronization barriers between threads of a same node
 *
 *  @details This follows the implementation of pthread_barrier(3)
 *
 */


/* The Linux includes are completely screwed up right now. Even if they
 * correctly export a _POSIX_BARRIER define the barrier functions are
 * not correctly defined in the pthread.h. So until we figure out
 * how to correctly identify their availability, we will have to
 * disable them.
 */
BEGIN_C_DECLS

/** @cond FALSE */
#if defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS - 20012L) >= 0 && 0

typedef pthread_barrier_t parsec_barrier_t;
#define parsec_barrier_init pthread_barrier_init
#define parsec_barrier_wait pthread_barrier_wait
#define parsec_barrier_destroy pthread_barrier_destroy
#define PARSEC_IMPLEMENT_BARRIERS 0

#else

/**
 * @endcond
 *
 * @brief A multithread barrier
 *
 * @details This structure is used to enable thread synchronization by
 *          providing a simple barrier mechanism.
 */
typedef struct parsec_barrier_t {
    int                 count;       /**< Number of threads expected to enter the barrier */
    volatile int        curcount;    /**< Number of threads currently inside the barrier */
    volatile int        generation;  /**< Unique number used to count how many times this
                                      *   barrier was used, and enable debugging unmatching barriers */
    pthread_mutex_t     mutex;       /**< Lock on the barrier, to make threads wait passively */
    pthread_cond_t      cond;        /**< Condition on the barrier, to allow waking up threads that wait
                                      *   passively once all threads have joined the barrier */
} parsec_barrier_t;

/**
 * @brief Initializes a barrier
 *
 * @details This sets the fields of the barrier object and define the
 * passive behavior of the barrier mutex
 *
 * @param[out] barrier the barrier to initialize
 * @param[in] pthread_mutex_attr attributes to pass to pthread_mutex_init
 * @param[in] count number of threads that will join the barrier
 * @return 0 if success another code otherwise.
 */
int parsec_barrier_init(parsec_barrier_t *barrier, const void *pthread_mutex_attr, unsigned int count);

/**
 * @brief synchronize  at  a  barrier
 *
 * @details The parsec_barrier_wait() function shall synchronize
 *       participating threads at the barrier referenced by barrier.
 *       The calling thread shall block until the required number of
 *       threads have called parsec_barrier_wait() specifying the
 *       barrier.
 *
 * @param[inout] barrier the barrier to wait upon
 * @return 0 if success another code otherwise.
 */
int parsec_barrier_wait(parsec_barrier_t* barrier);

/**
 * @brief  destroy a barrier object
 *
 * @details The parsec_barrier_destroy() function destroys the barrier
 *      referenced by barrier and release any resources used by the
 *      barrier.  The effect of subsequent use of the barrier is
 *      undefined until the barrier is reinitialized by another call
 *      to parsec_barrier_init(). The results are undefined if
 *      parsec_barrier_destroy() is called when any thread is blocked
 *      on the barrier, or if this function is called with an
 *      uninitialized barrier.
 *
 * @param[inout] barrier the barrier to destroy
 * @return 0 if success another code otherwise.
 */
int parsec_barrier_destroy(parsec_barrier_t* barrier);
#define PARSEC_IMPLEMENT_BARRIERS 1

#endif

END_C_DECLS

/**
 * @}
 */

#endif  /* PARSEC_BARRIER_H_HAS_BEEN_INCLUDED */
