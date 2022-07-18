/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef _parsec_biased_rwlock_h
#define _parsec_biased_rwlock_h

#include "parsec/parsec_config.h"

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

/* fwd-decl */
typedef struct parsec_biased_rwlock_t parsec_biased_rwlock_t;

int parsec_biased_rwlock_init(parsec_biased_rwlock_t **lock);

void parsec_biased_rwlock_rdlock(parsec_biased_rwlock_t *lock);

void parsec_biased_rwlock_rdunlock(parsec_biased_rwlock_t *lock);

void parsec_biased_rwlock_wrlock(parsec_biased_rwlock_t *lock);

void parsec_biased_rwlock_wrunlock(parsec_biased_rwlock_t *lock);

#endif // _parsec_biased_rwlock_h
