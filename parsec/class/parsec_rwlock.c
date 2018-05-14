/*
 * Copyright (c)      2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/sys/atomic.h"
#include "parsec/class/parsec_rwlock.h"
#include <stdint.h>
#include <unistd.h>
#include <assert.h>

#if RWLOCK_IMPL == RWLOCK_IMPL_STATE

/*
 * Simple State-based implementation with atomic CAS to update the state at each step
 * Two high bits are used to set:
 *   - a thread is requesting the write lock
 *   - a thread has the write lock
 * The 30 low bits are used to count the number of readers.
 */

#define PARSEC_ATOMIC_RWLOCK_WRITER_BIT         ((parsec_atomic_rwlock_t)0x80000000)
#define PARSEC_ATOMIC_RWLOCK_WRITER_WAITING_BIT ((parsec_atomic_rwlock_t)0x40000000)
#define PARSEC_ATOMIC_RWLOCK_WRITER_BITS        (PARSEC_ATOMIC_RWLOCK_WRITER_BIT|PARSEC_ATOMIC_RWLOCK_WRITER_WAITING_BIT)
#define PARSEC_ATOMIC_RWLOCK_READER_BITS        (~PARSEC_ATOMIC_RWLOCK_WRITER_BITS)

void parsec_atomic_rwlock_init( parsec_atomic_rwlock_t* atomic_rwlock )
{
    *atomic_rwlock = 0u;
}

void parsec_atomic_rwlock_rdlock( parsec_atomic_rwlock_t* atomic_rwlock )
{
    parsec_atomic_rwlock_t old_state, new_state;

    do {
        old_state = *atomic_rwlock;
#if defined(PARSEC_DEBUG_PARANOID)
        assert( ((old_state & PARSEC_ATOMIC_RWLOCK_WRITER_BIT) == 0) || ((old_state & PARSEC_ATOMIC_RWLOCK_READER_BITS) == 0) );
#endif
        if( (old_state & PARSEC_ATOMIC_RWLOCK_WRITER_BITS) == 0 ) {
            /* We don't need to care for preserving the high bits: they are 0 */
            new_state = old_state + 1;
            if( parsec_atomic_cas_int32(atomic_rwlock, old_state, new_state) )
                return;
        }
    } while(1);
}

void parsec_atomic_rwlock_rdunlock( parsec_atomic_rwlock_t* atomic_rwlock )
{
    parsec_atomic_rwlock_t old_state, new_state;
    do {
        old_state = *atomic_rwlock;
#if defined(PARSEC_DEBUG_PARANOID)
        assert((old_state & PARSEC_ATOMIC_RWLOCK_READER_BITS) != 0);
        assert((old_state & PARSEC_ATOMIC_RWLOCK_WRITER_BIT) == 0);
#endif
        new_state =
            (((old_state & PARSEC_ATOMIC_RWLOCK_READER_BITS)-1) & PARSEC_ATOMIC_RWLOCK_READER_BITS) |
            (old_state & PARSEC_ATOMIC_RWLOCK_WRITER_WAITING_BIT);
    } while(!parsec_atomic_cas_int32(atomic_rwlock, old_state, new_state));
}

void parsec_atomic_rwlock_wrlock( parsec_atomic_rwlock_t* atomic_rwlock )
{
    parsec_atomic_rwlock_t old_state, new_state;
    do {
        old_state = *atomic_rwlock;
        if( ((old_state == 0) || (old_state == PARSEC_ATOMIC_RWLOCK_WRITER_WAITING_BIT)) ) {
            new_state = PARSEC_ATOMIC_RWLOCK_WRITER_BIT;
            if( parsec_atomic_cas_int32( atomic_rwlock, old_state, new_state) )
                return;
        }

        /* Just try once to set the writer waiting; we're going to loop here anyway */
        if( (old_state & PARSEC_ATOMIC_RWLOCK_WRITER_WAITING_BIT) == 0 ) {
            new_state = old_state | PARSEC_ATOMIC_RWLOCK_WRITER_WAITING_BIT;
            parsec_atomic_cas_int32(atomic_rwlock, old_state, new_state);
        }
    } while(1);
}

void parsec_atomic_rwlock_wrunlock( parsec_atomic_rwlock_t* atomic_rwlock )
{
    parsec_atomic_rwlock_t old_state, new_state;
    do {
        old_state = *atomic_rwlock;
#if defined(PARSEC_DEBUG_PARANOID)
        assert( old_state & PARSEC_ATOMIC_RWLOCK_WRITER_BIT );
        assert( (old_state & PARSEC_ATOMIC_RWLOCK_READER_BITS) == 0);
#endif
        new_state = (old_state & PARSEC_ATOMIC_RWLOCK_WRITER_WAITING_BIT);
    } while( !parsec_atomic_cas_int32(atomic_rwlock, old_state, new_state) );
}

#elif RWLOCK_IMPL == RWLOCK_IMPL_TICKET

/* Ticket based (phase-fair) implementation
 *    http://dl.acm.org/citation.cfm?id=1842604
 */

#include <string.h>

#include <time.h>
#define RINC 0x100

#define WBITS  0x3
#define PRES   0x2
#define PHID   0x1

void parsec_atomic_rwlock_init(parsec_atomic_rwlock_t *L)
{
    memset((void*)L, 0, sizeof(parsec_atomic_rwlock_t));
}

void parsec_atomic_rwlock_rdlock(parsec_atomic_rwlock_t *L)
{
    int32_t w;
    int count = 0;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 };
    w = parsec_atomic_fetch_add_int32(&L->rin, RINC) & WBITS;
    if( w == 0 )
        return;
    while( w == (L->rin & WBITS) )
        if( count++ > 1000 )
            nanosleep( &ts, NULL );
}

void parsec_atomic_rwlock_rdunlock(parsec_atomic_rwlock_t *L)
{
    parsec_atomic_fetch_add_int32(&L->rout, RINC);
}

void parsec_atomic_rwlock_wrlock(parsec_atomic_rwlock_t *L)
{
    int32_t ticket, w;
    int count = 0;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 };
    ticket = parsec_atomic_fetch_inc_int32(&L->win);
    while( L->wout != ticket )
        if( count++ > 1000 )
            nanosleep( &ts, NULL );
    w = PRES | (ticket & PHID);
    ticket = parsec_atomic_fetch_add_int32(&L->rin, w);
    count = 0;
    while( L->rout != ticket )
        if( count++ > 1000 )
            nanosleep( &ts, NULL );
}

void parsec_atomic_rwlock_wrunlock(parsec_atomic_rwlock_t *L)
{
    /* This is slightly different from the code in the cited paper:
     *   - to ensure order of operations (update of L->rin must happen before the
     *     update of L->wout)
     *   - and to avoid depending upon machine endianness
     * we use an additional explicit atomic here. The code in the cited paper
     * used (atomic) assignment of the lowest significant byte of L->rin to 0.
     * Per mutual exclusion of the writes, we should be the only thread that
     * changes wout at this time, so no atomic is needed here.
     */
    parsec_atomic_fetch_and_int32(&L->rin, 0xFFFFFF00);
    L->wout = L->wout+1;
}

#elif RWLOCK_IMPL == RWLOCK_IMPL_2LOCKS

/* Traditional (e.g. http://www.springer.com/us/book/9783642320262)
 * implementation based on two atomic locks. This is write-preferring. */

#include <string.h>

void parsec_atomic_rwlock_init(parsec_atomic_rwlock_t *L)
{
    memset(L, 0, sizeof(parsec_atomic_rwlock_t));
}

void parsec_atomic_rwlock_wrunlock(parsec_atomic_rwlock_t *L)
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert(L->nbreaders == 0);
    assert(L->writer == (int)pthread_self());
    L->writer = 0;
#endif
    parsec_atomic_unlock(&L->w);
}

void parsec_atomic_rwlock_wrlock(parsec_atomic_rwlock_t *L)
{
    parsec_atomic_lock(&L->w);
#if defined(PARSEC_DEBUG_PARANOID)
    assert(L->writer == 0);
    assert(L->nbreaders == 0);
    L->writer = (int)pthread_self();
#endif
}

void parsec_atomic_rwlock_rdunlock(parsec_atomic_rwlock_t *L)
{
#if defined(PARSEC_DEBUG_PARANOID)
        int i;
#endif
    parsec_atomic_lock(&L->r);
#if defined(PARSEC_DEBUG_PARANOID)
    assert(L->nbreaders > 0);
    for(i = 0; i < L->nbreaders; i++) {
        if( L->readers[i] == (int)pthread_self() )
            break;
    }
    assert(i < L->nbreaders);
#endif
    L->nbreaders--;
#if defined(PARSEC_DEBUG_PARANOID)
    L->readers[i] = L->readers[L->nbreaders];
    L->readers[L->nbreaders] = 0;
#endif
    if( L->nbreaders == 0 )
        parsec_atomic_unlock(&L->w);
    parsec_atomic_unlock(&L->r);
}

void parsec_atomic_rwlock_rdlock(parsec_atomic_rwlock_t *L)
{
    parsec_atomic_lock(&L->r);
    if(0 == L->nbreaders)
        parsec_atomic_lock(&L->w);
#if defined(PARSEC_DEBUG_PARANOID)
    assert(L->writer == 0);
    L->readers[L->nbreaders] = (int)pthread_self();
    assert(L->nbreaders < MAX_READERS);
#endif
    L->nbreaders++;
    parsec_atomic_unlock(&L->r);
}

#elif RWLOCK_IMPL == RWLOCK_IMPL_MYTICKET

/* Simpler Ticket based (phase-fair) implementation that does not
 * depend upon endianness, but can tolerate only up to 65536 writers.
 */

#include <string.h>

void parsec_atomic_rwlock_init(parsec_atomic_rwlock_t *L)
{
    L->fields.current_ticket = 0;
    L->fields.next_ticket    = 0;
    L->fields.readers_ticket = 0;
    L->fields.nb_readers     = 0;
}

void parsec_atomic_rwlock_rdlock(parsec_atomic_rwlock_t *L)
{
    parsec_atomic_rwlock_t old, new;
    uint16_t my_ticket;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 };

    do {
        old = *L;
        /* The new state will not be that different from the old one */
        new = old;

        /* If we are the first reader, take a ticket,
         * otherwise, piggyback on the first reader's ticket.
         */
        if( old.fields.nb_readers == 0 ) {
            new.fields.nb_readers = 1;
            my_ticket = old.fields.next_ticket;
            new.fields.readers_ticket = my_ticket;
            new.fields.next_ticket    = my_ticket+1;
        } else {
            new.fields.nb_readers = old.fields.nb_readers+1;
            my_ticket             = old.fields.readers_ticket;
        }

        /* Atomically update the RWL state, and try again if something changed */
    } while(! parsec_atomic_cas_int64(&L->atomic_word, old.atomic_word, new.atomic_word) );

    do {
        nanosleep( &ts, NULL );
        old = *L;
    }  while( old.fields.current_ticket != my_ticket );

}

void parsec_atomic_rwlock_rdunlock(parsec_atomic_rwlock_t *L)
{
    parsec_atomic_rwlock_t old, new;

    do {
        old = *L;
        /* The new state will not be that different from the old one */
        new = old;

        assert(old.fields.nb_readers > 0);
        new.fields.nb_readers = old.fields.nb_readers - 1;
        if( new.fields.nb_readers == 0 ) {
            new.fields.current_ticket = old.fields.current_ticket + 1;
        }

        /* Try until we succeed */
    } while( !parsec_atomic_cas_int64(&L->atomic_word, old.atomic_word, new.atomic_word) );
}

void parsec_atomic_rwlock_wrlock(parsec_atomic_rwlock_t *L)
{
    parsec_atomic_rwlock_t old, new;
    uint16_t my_ticket;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 };

    do {
        old = *L;
        /* The new state will not be that different from the old one */
        new = old;

        /* Take a ticket */
        my_ticket = old.fields.next_ticket;
        new.fields.next_ticket = my_ticket+1;

        /* Try to atomically update the RWL state until we succeed */
    } while( !parsec_atomic_cas_int64(&L->atomic_word, old.atomic_word, new.atomic_word) );

    do {
        nanosleep( &ts, NULL );
        old = *L;
    }  while( old.fields.current_ticket != my_ticket );
}

void parsec_atomic_rwlock_wrunlock(parsec_atomic_rwlock_t *L)
{
    parsec_atomic_rwlock_t old, new;

    do {
        old = *L;
        /* The new state will not be that different from the old one */
        new = old;

        new.fields.current_ticket = old.fields.current_ticket + 1;
    } while( !parsec_atomic_cas_int64(&L->atomic_word, old.atomic_word, new.atomic_word) );
}

#endif
