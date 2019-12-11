/*
 * Copyright (c) 2017-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_RWLOCK_H_HAS_BEEN_INCLUDED
#define PARSEC_RWLOCK_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/sys/atomic.h"
#include <stdint.h>
#include <unistd.h>
#include <assert.h>

BEGIN_C_DECLS

#define PARSEC_RWLOCK_IMPL_STATE  1    /**< Simple State-based RWLock */
#define PARSEC_RWLOCK_IMPL_TICKET 2    /**< Phase-fair RWLock from http://dl.acm.org/citation.cfm?id=1842604 */
#define PARSEC_RWLOCK_IMPL_2LOCKS 3    /**< Traditional RWLock based on two atomic locks (http://www.springer.com/us/book/9783642320262)
                                        *   (Enables paranoid checks by recording what threads are in read or write state when
                                        *   PARSEC_DEBUG_PARANOID is on) */
#define PARSEC_RWLOCK_IMPL_MYTICKET 4  /**< Simpler (more portable?) version of the phase-fair RWLock */

/**
 * There are 4 implementations of the RWLocks.
 * The following define chooses which one we use.
 */
#define PARSEC_RWLOCK_IMPL PARSEC_RWLOCK_IMPL_TICKET

#if PARSEC_RWLOCK_IMPL == PARSEC_RWLOCK_IMPL_STATE

/**
 * Simple State-based implementation with atomic CAS to update the state at each step
 * Two high bits are used to set:
 *   - a thread is requesting the write lock
 *   - a thread has the write lock
 * The 30 low bits are used to count the number of readers.
 */
typedef volatile uint32_t parsec_atomic_rwlock_t;

#define PARSEC_RWLOCK_UNLOCKED 0

#elif PARSEC_RWLOCK_IMPL == PARSEC_RWLOCK_IMPL_TICKET

/**
 *  Ticket based (phase-fair) implementation
 *    http://dl.acm.org/citation.cfm?id=1842604
 */
typedef volatile struct parsec_atomic_rwlock_s {
    int32_t rin;    /**< How many readers requested to enter (3 high bytes, low byte used for writer requests) */
    int32_t rout;   /**< How many readers left (compared only to rin read values with equal) */
    int32_t win;    /**< How many writers requested to enter (3 high bytes, low byte unused) */
    int32_t wout;   /**< How many writers left (compared only to rin read values with equal) */
} parsec_atomic_rwlock_t;

#define PARSEC_RWLOCK_UNLOCKED 0, 0, 0, 0

#elif PARSEC_RWLOCK_IMPL == PARSEC_RWLOCK_IMPL_2LOCKS

#if defined(PARSEC_DEBUG_PARANOID)
/** For Paranoid checks, keep a list of MAX_READERS_TO_CHECK readers inside the critical section */
#define MAX_READERS_TO_CHECK 34
#include <pthread.h>
#endif

#include <string.h>

/**
 * Simple RWLock based on 2 locks
 * (e.g. http://www.springer.com/us/book/9783642320262) 
 * implementation based on two atomic locks. This is write-preferring. 
 */
typedef struct parsec_atomic_rwlock_s {
    parsec_atomic_lock_t r;  /**< Lock taken by readers to update nbreaders */
    parsec_atomic_lock_t w;  /**< Lock taken by writers and by the first / released by the last reader */
    int nbreaders;           /**< Counter of the number of readers inside the critical section */
#if defined(PARSEC_DEBUG_PARANOID)
    int writer;              /**< ID of the writer inside the critical section */
    int readers[MAX_READERS_TO_CHECK]; /**< Array of IDs of readers inside the critical section */
#endif
} parsec_atomic_rwlock_t;

#if defined(PARSEC_DEBUG_PARANOID)
#define PARSEC_RWLOCK_UNLOCKED {PARSEC_ATOMIC_UNLOCKED}, {PARSEC_LOCK_UNLOCK}, 0, 0,
#else
#define PARSEC_RWLOCK_UNLOCKED {PARSEC_ATOMIC_UNLOCKED}, {PARSEC_LOCK_UNLOCK}, 0
#endif

#elif PARSEC_RWLOCK_IMPL == PARSEC_RWLOCK_IMPL_MYTICKET

/**
 * Simpler Ticket based (phase-fair) implementation that does not
 * depend upon endianness, but can tolerate only up to 65536 writers.
 */
typedef volatile union parsec_atomic_rwlock_u {
    uint64_t atomic_word;         /**< Atomic switchable version of the structure */
    struct {
        uint16_t current_ticket;  /**< Current value of the ticket */
        uint16_t next_ticket;     /**< Value to use for the next ticket */
        uint16_t readers_ticket;  /**< Ticket used by readers
                                   *   The first reader takes a ticket, the next ones
                                   *   piggy back on the same ticket until they all leave
                                   *   the critical section
                                   */
        uint16_t nb_readers;      /**< Number of readers on the readers_ticket */
    } fields;
} parsec_atomic_rwlock_t;

#define PARSEC_RWLOCK_UNLOCKED 0, {0, 0, 0, 0}

#else
#error "NO RWLOCK"
#endif

/**
 * @brief Initializes a Readers-Writer Lock
 *
 * @details Should be called by a single thread to initialize the shared variable
 *          before any other call. Leaves the RW lock in a state that will allow
 *          one writer or any readers to enter the critical section.
 *
 * @param[OUT] L the Readers-Writer lock to initialize
 */
void parsec_atomic_rwlock_init(parsec_atomic_rwlock_t *L);

/**
 * @brief Take a read-lock of a Readers-Writer Lock
 *
 * @details If no writer or if some reader has the lock, 
 *          allows the calling reader to enter the critical section.
 *          Other readers will be allowed to enter it, but writers
 *          are prevented to enter the section until all readers have
 *          left it.
 *
 * @param[OUT] L the Readers-Writer lock to use for synchronization.
 */
void parsec_atomic_rwlock_rdlock(parsec_atomic_rwlock_t *L);

/**
 * @brief Release one read-lock of a Readers-Writer Lock
 *
 * @details Signifies that one of the readers that was in the
 *          critical section left it. May allow a writer to enter
 *          the critical section if one was waiting.
 *
 * @param[OUT] L the Readers-Writer lock to use for synchronization.
 */
void parsec_atomic_rwlock_rdunlock(parsec_atomic_rwlock_t *L);

/**
 * @brief Take a write-lock of a Readers-Writer Lock
 *
 * @details If no reader and if no writer reader has the lock, 
 *          allows the calling writer to enter the critical section.
 *          Other writers and readers are prevented to enter it until
 *          that writer leaves the critical section.
 *
 * @param[OUT] L the Readers-Writer lock to use for synchronization.
 */
void parsec_atomic_rwlock_wrlock(parsec_atomic_rwlock_t *L);

/**
 * @brief Release one read-lock of a Readers-Writer Lock
 *
 * @details Signifies that one of the readers that was in the
 *          critical section left it. May allow a writer or the
 *          readers to enter the critical section if some was waiting.
 *
 * @param[OUT] L the Readers-Writer lock to use for synchronization.
 */
void parsec_atomic_rwlock_wrunlock(parsec_atomic_rwlock_t *L);


END_C_DECLS

#endif  /* PARSEC_RWLOCK_H_HAS_BEEN_INCLUDED */
