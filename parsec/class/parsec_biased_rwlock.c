/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/class/parsec_biased_rwlock.h"

#include <assert.h>

#include "parsec/runtime.h"
#include "parsec/constants.h"
#include "parsec/execution_stream.h"
#include "parsec/sys/atomic.h"
#include "parsec/class/parsec_rwlock.h"

/**
 * An implementation of the BRAVO biased reader/writer lock wrapper.
 * The goal of the BRAVO lock wrapper is to avoid contending the atomic
 * rwlock with reader locks, instead having threads mark their read status
 * is an array. A writer will first take the rwlock, signal that a writer
 * is active, and then wait for all readers to complete. New readers will
 * see that a writer is active and wait for the reader lock to become available.
 *
 * This is clearly biased towards readers so this implementation is meant for
 * cases where the majority of accesses is reading and only occasional writes occur.
 *
 * The paper presenting this technique is available at:
 * https://arxiv.org/abs/1810.01553
 *
 * While the original implementation uses a global hash table, we use a smaller table
 * per lock. In PaRSEC, we know the number of threads we control up front.
 * We simply pad for a cache line. If an unknown thread tries to take the lock against
 * all odds, it falls back to taking the reader lock.
 */

struct parsec_biased_rwlock_t {
    parsec_atomic_rwlock_t    rw_lock;              /**< underlying reader-writer lock */
    int32_t                   reader_bias;          /**< whether locking is biased towards readers, will change if a writer occurs */
    uint32_t                  num_reader;           /**< size of the reader_active field */
    uint8_t                   reader_active[];      /**< array with flags signalling reading threads */
};

#define DEFAULT_CACHE_SIZE 64

int parsec_biased_rwlock_init(parsec_biased_rwlock_t **lock) {
    parsec_biased_rwlock_t *res;
    parsec_execution_stream_t *es = parsec_my_execution_stream();
    if (NULL == es) {
        /* should be called from a parsec thread */
        res = (parsec_biased_rwlock_t *)malloc(sizeof(parsec_biased_rwlock_t));
        res->num_reader = 0;
        res->reader_bias = 0; // disable reader biasing
    } else {
        uint32_t num_threads = es->virtual_process->nb_cores;
        /* one cache line per reader */
        uint32_t num_reader = num_threads*DEFAULT_CACHE_SIZE;
        res = (parsec_biased_rwlock_t *)malloc(sizeof(parsec_biased_rwlock_t) + num_reader*sizeof(uint8_t));
        res->reader_bias = 1;
        res->num_reader = num_reader;
        memset(res->reader_active, 0, num_reader);
    }
    parsec_atomic_rwlock_init(&res->rw_lock);
    *lock = res;

    return PARSEC_SUCCESS;
}

void parsec_biased_rwlock_rdlock(parsec_biased_rwlock_t *lock)
{
    parsec_execution_stream_t *es = parsec_my_execution_stream();
    if (PARSEC_UNLIKELY(NULL == es || lock->num_reader == 0)) {
        /* fall back to the underlying rwlock */
        parsec_atomic_rwlock_rdlock(&lock->rw_lock);
        return;
    }

    if (PARSEC_UNLIKELY(!lock->reader_bias)) {
        /* a writer is active, wait for the rwlock to become available */
        parsec_atomic_rwlock_rdlock(&lock->rw_lock);
        return;
    }

    /* fast-path: no writer, simply mark as active reader and make sure there is no race */
    size_t reader_entry = es->th_id*DEFAULT_CACHE_SIZE;
    assert(reader_entry >= 0 && reader_entry < lock->num_reader);
    assert(lock->reader_active[reader_entry] == 0);

    lock->reader_active[reader_entry] = 1;
    /* make sure the writer check is not moved to before setting the flag */
    parsec_atomic_rmb();
    /* double check that no writer came in between */
    if (PARSEC_UNLIKELY(!lock->reader_bias)) {
        /* a writer has become active, fallback to the rwlock */
        lock->reader_active[reader_entry] = 0;
        parsec_atomic_rwlock_rdlock(&lock->rw_lock);
    }
}

void parsec_biased_rwlock_rdunlock(parsec_biased_rwlock_t *lock)
{
    parsec_execution_stream_t *es = parsec_my_execution_stream();

    if (PARSEC_UNLIKELY(NULL == es || lock->num_reader == 0)) {
        /* fall back to the underlying rwlock */
        parsec_atomic_rwlock_rdunlock(&lock->rw_lock);
        return;
    }

    size_t reader_entry = es->th_id*DEFAULT_CACHE_SIZE;
    assert(reader_entry >= 0 && reader_entry < lock->num_reader);

    if (PARSEC_UNLIKELY(lock->reader_active[reader_entry] == 0)) {
        /* we had to take a lock, give it back */
        parsec_atomic_rwlock_rdunlock(&lock->rw_lock);
    } else {
        lock->reader_active[reader_entry] = 0;
    }
}

void parsec_biased_rwlock_wrlock(parsec_biased_rwlock_t *lock)
{
    /* acquire the writer lock first */
    parsec_atomic_rwlock_wrlock(&lock->rw_lock);

    lock->reader_bias = 0;

    /* make sure the reads below are not moved before the write */
    parsec_atomic_wmb();

    /* wait for all current reader to complete */
    for (uint32_t i = 0; i < lock->num_reader; ++i) {
        while (lock->reader_active[i] != 0) {
          static struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 };
          nanosleep(&ts, NULL);
        }
    }
}

void parsec_biased_rwlock_wrunlock(parsec_biased_rwlock_t *lock)
{
    assert(lock->reader_bias == 0);
    if (lock->num_reader > 0) {
        /* re-enable reader bias, if we support it */
        lock->reader_bias = 1;
    }
    parsec_atomic_rwlock_wrunlock(&lock->rw_lock);
}
