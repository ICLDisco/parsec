/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef LIFO_H_HAS_BEEN_INCLUDED
#define LIFO_H_HAS_BEEN_INCLUDED

#include "parsec_config.h"
#include "parsec/class/list_item.h"
#if defined(PARSEC_HAVE_ATOMIC_LLSC_PTR)
#include <time.h>
#endif  /* defined(PARSEC_HAVE_ATOMIC_LLSC_PTR) */

typedef struct parsec_lifo_s parsec_lifo_t;
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_lifo_t);

static inline int
parsec_lifo_is_empty( parsec_lifo_t* lifo );

static inline void
parsec_lifo_push(parsec_lifo_t* lifo, parsec_list_item_t* item);
static inline void
parsec_lifo_nolock_push(parsec_lifo_t* lifo, parsec_list_item_t* item);

static inline void
parsec_lifo_chain(parsec_lifo_t* lifo, parsec_list_item_t* items);
static inline void
parsec_lifo_nolock_chain(parsec_lifo_t* lifo, parsec_list_item_t* items);

static inline parsec_list_item_t*
parsec_lifo_pop(parsec_lifo_t* lifo);
static inline parsec_list_item_t*
parsec_lifo_try_pop(parsec_lifo_t* lifo);
static inline parsec_list_item_t*
parsec_lifo_nolock_pop(parsec_lifo_t* lifo);

/***********************************************************************
 * Interface is defined. Everything else is private thereafter */

/**
 * By default all LIFO will handle elements aligned to PARSEC_LIFO_ALIGNMENT_DEFAULT
 * bits. If a different type of alignment is needed, the trick is to manually allocate
 * the lifo and set the alignment by hand before calling OBJ_CONSTRUCT on it.
 */
#if !defined(PARSEC_LIFO_ALIGNMENT_DEFAULT)
#define PARSEC_LIFO_ALIGNMENT_DEFAULT 3
#endif  /* !defined(PARSEC_LIFO_ALIGNMENT_DEFAULT) */

#define PARSEC_LIFO_ALIGNMENT_BITS(LIFO)  ((LIFO)->alignment)
#define PARSEC_LIFO_ALIGNMENT(LIFO)       (( ( ((uintptr_t)1 << PARSEC_LIFO_ALIGNMENT_BITS(LIFO) ) < sizeof(void*) ) ? \
                                            ( sizeof(void*) ) :         \
                                            ( (uintptr_t)1 << PARSEC_LIFO_ALIGNMENT_BITS(LIFO) ) ))

#include <stdlib.h>
#include "parsec/sys/atomic.h"

/**
 * This code is imported from Open MPI.
 */
/**
 * Counted pointer to avoid the ABA problem.
 */
typedef union parsec_counted_pointer_u {
    struct {
        /** update counter used when cmpset_128 is available */
        uint64_t counter;
        /** list item pointer */
        parsec_list_item_t *item;
    } data;
#if defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_128B)
    /** used for atomics when there is a cmpset that can operate on
     * two 64-bit values */
    __uint128_t value;
#endif  /* defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_128B) */
} parsec_counted_pointer_t;

struct parsec_lifo_s {
    parsec_object_t           super;
    uint8_t                  alignment;
    parsec_list_item_t       *lifo_ghost;
    parsec_counted_pointer_t  lifo_head;
};

/* The ghost pointer will never change. The head will change via an
 * atomic compare-and-swap. On most architectures the reading of a
 * pointer is an atomic operation so we don't have to protect it. */
static inline int parsec_lifo_is_empty( parsec_lifo_t* lifo )
{
    return ((parsec_list_item_t *)lifo->lifo_head.data.item == lifo->lifo_ghost);
}
#define parsec_lifo_nolock_is_empty parsec_lifo_is_empty

#if defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_128B)
/* Add one element to the FIFO. We will return the last head of the list
 * to allow the upper level to detect if this element is the first one in the
 * list (if the list was empty before this operation).
 */
static inline int
parsec_update_counted_pointer(volatile parsec_counted_pointer_t *addr, parsec_counted_pointer_t old,
                             parsec_list_item_t *item)
{
    parsec_counted_pointer_t elem = {.data = {.counter = old.data.counter + 1, .item = item}};
    return parsec_atomic_cas_128b(&addr->value, old.value, elem.value);
}

static inline void parsec_lifo_push( parsec_lifo_t* lifo,
                                    parsec_list_item_t* item )
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)item % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    PARSEC_ITEM_ATTACH(lifo, item);

    do {
        parsec_list_item_t *next = (parsec_list_item_t *) lifo->lifo_head.data.item;

        item->list_next = next;
        parsec_atomic_wmb ();

        /* to protect against ABA issues it is sufficient to only update the counter in pop */
        if (parsec_atomic_cas_ptr(&lifo->lifo_head.data.item, next, item)) {
            return;
        }
        /* DO some kind of pause to release the bus */
    } while (1);
}
static inline void parsec_lifo_chain( parsec_lifo_t* lifo,
                                     parsec_list_item_t* ring)
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)ring % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    PARSEC_ITEMS_ATTACH(lifo, ring);

    parsec_list_item_t* tail = (parsec_list_item_t*)ring->list_prev;

    do {
        parsec_list_item_t *next = (parsec_list_item_t *) lifo->lifo_head.data.item;

        tail->list_next = next;
        parsec_atomic_wmb ();

        /* to protect against ABA issues it is sufficient to only update the counter in pop */
        if (parsec_atomic_cas_ptr(&lifo->lifo_head.data.item, next, ring)) {
            return;
        }
        /* DO some kind of pause to release the bus */
    } while (1);
}

static inline parsec_list_item_t* parsec_lifo_pop( parsec_lifo_t* lifo )
{
    parsec_list_item_t *item;

    do {
        parsec_counted_pointer_t old_head;

        old_head.data.counter = lifo->lifo_head.data.counter;
        parsec_atomic_rmb ();
        item = old_head.data.item = lifo->lifo_head.data.item;

        if (item == lifo->lifo_ghost) {
            return NULL;
        }

        if (parsec_update_counted_pointer (&lifo->lifo_head, old_head,
                                         (parsec_list_item_t *) item->list_next)) {
            parsec_atomic_wmb ();
            item->list_next = NULL;
            PARSEC_ITEM_DETACH(item);
            return item;
        }
    } while (1);
}

static inline parsec_list_item_t* parsec_lifo_try_pop( parsec_lifo_t* lifo )
{
    parsec_counted_pointer_t old_head;
    parsec_list_item_t *item;

    old_head.data.counter = lifo->lifo_head.data.counter;
    parsec_atomic_rmb();
    item = old_head.data.item = lifo->lifo_head.data.item;

    if (item == lifo->lifo_ghost) {
        return NULL;
    }

    if (parsec_update_counted_pointer (&lifo->lifo_head, old_head,
                                     (parsec_list_item_t *) item->list_next)) {
        parsec_atomic_wmb();
        item->list_next = NULL;
        PARSEC_ITEM_DETACH(item);
        return item;
    }
    return NULL;
}

#else  /* !defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_128B) */

/* Add one element to the LIFO. We will return the last head of the list
 * to allow the upper level to detect if this element is the first one in the
 * list (if the list was empty before this operation).
 */
static inline void parsec_lifo_push(parsec_lifo_t *lifo,
                                   parsec_list_item_t *item)
{
    /* item free acts as a mini lock to avoid ABA problems */
    item->aba_key = 1;
     do {
        parsec_list_item_t *next = (parsec_list_item_t *) lifo->lifo_head.data.item;
        item->list_next = next;
        parsec_atomic_wmb();
         if( parsec_atomic_cas_ptr(&lifo->lifo_head.data.item, next, item) ) {
            parsec_atomic_wmb();
            /* now safe to pop this item */
            item->aba_key = 0;
            return;
        }
        /* DO some kind of pause to release the bus */
    } while (1);
}

static inline void parsec_lifo_chain( parsec_lifo_t* lifo,
                                     parsec_list_item_t* ring)
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)ring % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    PARSEC_ITEMS_ATTACH(lifo, ring);

    /* item free acts as a mini lock to avoid ABA problems */
    ring->aba_key = 1;
    parsec_list_item_t* tail = (parsec_list_item_t*)ring->list_prev;

     do {
        parsec_list_item_t *next = (parsec_list_item_t *) lifo->lifo_head.data.item;
        tail->list_next = next;
        parsec_atomic_wmb();
         if( parsec_atomic_cas_ptr(&lifo->lifo_head.data.item, next, ring) ) {
            parsec_atomic_wmb();
            /* now safe to pop this item */
            ring->aba_key = 0;
            return;
        }
        /* DO some kind of pause to release the bus */
    } while (1);
}

#if defined(PARSEC_HAVE_ATOMIC_LLSC_PTR)

static inline void _parsec_lifo_release_cpu (void)
{
    /* there are many ways to cause the current thread to be suspended. This one
     * should work well in most cases. Another approach would be to use poll (NULL, 0, ) but
     * the interval will be forced to be in ms (instead of ns or us). Note that there
     * is a performance improvement for the lifo test when this call is made on detection
     * of contention but it may not translate into actually MPI or application performance
     * improvements. */
    static struct timespec interval = { .tv_sec = 0, .tv_nsec = 100 };
    nanosleep (&interval, NULL);
}

/* Retrieve one element from the LIFO. If we reach the ghost element then the LIFO
 * is empty so we return NULL.
 */
static inline parsec_list_item_t *parsec_lifo_pop(parsec_lifo_t* lifo)
{
    parsec_list_item_t *item, *next;
    int attempt = 0;

    do {
        if (++attempt == 5) {
            /* deliberatly suspend this thread to allow other threads to run. this should
             * only occur during periods of contention on the lifo. */
            _parsec_lifo_release_cpu ();
            attempt = 0;
        }

        item = (parsec_list_item_t *) parsec_atomic_ll_ptr((long*)&(lifo->lifo_head.data.item));
        if (lifo->lifo_ghost == item) {
            return NULL;
        }

        next = (parsec_list_item_t *) item->list_next;
    } while (!parsec_atomic_sc_ptr((long*)&lifo->lifo_head.data.item, (intptr_t)next));

    parsec_atomic_wmb();

    item->list_next = NULL;
    PARSEC_ITEM_DETACH(item);
    return item;
}

static inline parsec_list_item_t* parsec_lifo_try_pop( parsec_lifo_t* lifo )
{
    parsec_list_item_t *item, *next;
    int attempt = 0;

    item = (parsec_list_item_t *) parsec_atomic_ll_ptr((long*)&lifo->lifo_head.data.item);
    if (lifo->lifo_ghost == item) {
        return NULL;
    }

    next = (parsec_list_item_t *) item->list_next;
    if( !parsec_atomic_sc_ptr((long*)&lifo->lifo_head.data.item, (intptr_t)next) )
        return NULL;

    parsec_atomic_wmb();

    item->list_next = NULL;
    PARSEC_ITEM_DETACH(item);
    return item;
}
#else

/* Retrieve one element from the LIFO. If we reach the ghost element then the LIFO
 * is empty so we return NULL.
 */
static inline parsec_list_item_t *parsec_lifo_pop(parsec_lifo_t* lifo)
{
    parsec_list_item_t *item;
    while ((item = lifo->lifo_head.data.item) != lifo->lifo_ghost) {
        /* ensure it is safe to pop the head */
        if (parsec_atomic_cas_64b(&item->aba_key, 0UL, 1UL)) {
            continue;
        }

        parsec_atomic_wmb ();

        /* try to swap out the head pointer */
        if( parsec_atomic_cas_ptr(&lifo->lifo_head.data.item,
                                  item, (void*)item->list_next) ) {
            break;
        }

        /* NTH: don't need another atomic here */
        item->aba_key = 0;

        /* Do some kind of pause to release the bus */
    }

    if (item == lifo->lifo_ghost) {
        return NULL;
    }

    parsec_atomic_wmb ();

    item->list_next = NULL;
    PARSEC_ITEM_DETACH(item);
    return item;
}

static inline parsec_list_item_t* parsec_lifo_try_pop( parsec_lifo_t* lifo )
{
    parsec_list_item_t *item;
    if( (item = lifo->lifo_head.data.item) != lifo->lifo_ghost ) {
        /* ensure it is safe to pop the head */
        if (parsec_atomic_cas_64b(&item->aba_key, 0UL, 1UL)) {
            return NULL;
        }

        parsec_atomic_wmb ();

        /* try to swap out the head pointer */
        if( parsec_atomic_cas_ptr(&lifo->lifo_head.data.item,
                                  item, (void *)item->list_next) ) {
            return NULL;
        }

        /* NTH: don't need another atomic here */
        item->aba_key = 0;

        /* Do some kind of pause to release the bus */
    }

    if (item == lifo->lifo_ghost) {
        return NULL;
    }

    parsec_atomic_wmb ();

    item->list_next = NULL;
    PARSEC_ITEM_DETACH(item);
    return item;
}
#endif /* defined(PARSEC_HAVE_ATOMIC_LLSC_PTR) */


#endif  /* defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_128B)) */

static inline void parsec_lifo_nolock_push( parsec_lifo_t* lifo,
                                           parsec_list_item_t* item )
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)item % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    PARSEC_ITEM_ATTACH(lifo, item);

    item->list_next = lifo->lifo_head.data.item;
    lifo->lifo_head.data.item = item;
}

static inline void parsec_lifo_nolock_chain( parsec_lifo_t* lifo,
                                            parsec_list_item_t* items )
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)items % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    PARSEC_ITEMS_ATTACH(lifo, items);

    parsec_list_item_t* tail = (parsec_list_item_t*)items->list_prev;

    tail->list_next = lifo->lifo_head.data.item;
    lifo->lifo_head.data.item = items;
}

static inline parsec_list_item_t* parsec_lifo_nolock_pop( parsec_lifo_t* lifo )
{
    parsec_list_item_t* item = lifo->lifo_head.data.item;
    lifo->lifo_head.data.item = (parsec_list_item_t*)item->list_next;
    PARSEC_ITEM_DETACH(item);
    return item;
}

/*
 * http://stackoverflow.com/questions/10528280/why-is-the-below-code-giving-dereferencing-type-punned-pointer-will-break-stric
 *
 * void * converts to any pointer type, and any pointer type converts
 * to void *, but void ** does not convert to a pointer to some other
 * type of pointer, nor do pointers to other pointer types convert to
 * void **.
 */
#define PARSEC_LIFO_ITEM_ALLOC( LIFO, elt, truesize ) ({         \
    void *_elt = NULL;                                          \
    int _rc;                                                    \
    _rc = posix_memalign(&_elt,                                 \
                         PARSEC_LIFO_ALIGNMENT(LIFO), (truesize));\
    assert( 0 == _rc && NULL != _elt ); (void)_rc;              \
    OBJ_CONSTRUCT(_elt, parsec_list_item_t);                     \
    (elt) = (__typeof__(elt))_elt;                              \
  })
#define PARSEC_LIFO_ITEM_FREE( elt ) do { OBJ_DESTRUCT( elt ); free(elt); } while (0)

#endif  /* LIFO_H_HAS_BEEN_INCLUDED */
