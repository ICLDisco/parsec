/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef LIFO_H_HAS_BEEN_INCLUDED
#define LIFO_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "dague/class/list_item.h"

typedef struct dague_lifo_s dague_lifo_t;
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_lifo_t);

static inline int
dague_lifo_is_empty( dague_lifo_t* lifo );
static inline int
dague_lifo_nolock_is_empty( dague_lifo_t* lifo);
#define dague_ulifo_is_empty(lifo) dague_lifo_nolock_is_empty(lifo);

static inline void
dague_lifo_push(dague_lifo_t* lifo, dague_list_item_t* item);
static inline void
dague_lifo_nolock_push(dague_lifo_t* lifo, dague_list_item_t* item);
#define dague_ulifo_push(lifo, item) dague_lifo_nolock_push(lifo, item)

static inline void
dague_lifo_chain(dague_lifo_t* lifo, dague_list_item_t* items);
static inline void
dague_lifo_nolock_chain(dague_lifo_t* lifo, dague_list_item_t* items);
#define dague_ulifo_chain(lifo, items) dague_lifo_nolock_chain(lifo, items)

static inline dague_list_item_t*
dague_lifo_pop(dague_lifo_t* lifo);
static inline dague_list_item_t*
dague_lifo_try_pop(dague_lifo_t* lifo);
static inline dague_list_item_t*
dague_lifo_nolock_pop(dague_lifo_t* lifo);
#define dague_ulifo_pop(lifo) dague_lifo_nolock_pop(lifo)


/***********************************************************************
 * Interface is defined. Everything else is private thereafter */

/**
 * By default all LIFO will handle elements aligned to DAGUE_LIFO_ALIGNMENT_DEFAULT
 * bits. If a different type of alignment is needed, the trick is to manually allocate
 * the lifo and set the alignment by hand before calling OBJ_CONSTRUCT on it.
 */
#if !defined(DAGUE_LIFO_ALIGNMENT_DEFAULT)
#define DAGUE_LIFO_ALIGNMENT_DEFAULT 3
#endif  /* !defined(DAGUE_LIFO_ALIGNMENT_DEFAULT) */

#define DAGUE_LIFO_ALIGNMENT_BITS(LIFO)  ((LIFO)->alignment)
#define DAGUE_LIFO_ALIGNMENT(LIFO)       (( ( ((uintptr_t)1 << DAGUE_LIFO_ALIGNMENT_BITS(LIFO) ) < sizeof(void*) ) ? \
                                            ( sizeof(void*) ) :         \
                                            ( (uintptr_t)1 << DAGUE_LIFO_ALIGNMENT_BITS(LIFO) ) ))

#include <stdlib.h>
#include <dague/sys/atomic.h>

/**
 * This code is imported from Open MPI.
 */
/**
 * Counted pointer to avoid the ABA problem.
 */
typedef union dague_counted_pointer_u {
    struct {
        /** update counter used when cmpset_128 is available */
        uint64_t counter;
        /** list item pointer */
        dague_list_item_t *item;
    } data;
#if defined(DAGUE_ATOMIC_HAS_ATOMIC_CAS_128B)
    /** used for atomics when there is a cmpset that can operate on
     * two 64-bit values */
    __uint128_t value;
#endif  /* defined(DAGUE_ATOMIC_HAS_ATOMIC_CAS_128B) */
} dague_counted_pointer_t;

struct dague_lifo_s {
    dague_object_t           super;
    uint8_t                  alignment;
    dague_list_item_t       *lifo_ghost;
    dague_counted_pointer_t  lifo_head;
};

/* The ghost pointer will never change. The head will change via an
 * atomic compare-and-swap. On most architectures the reading of a
 * pointer is an atomic operation so we don't have to protect it. */
static inline int dague_lifo_is_empty( dague_lifo_t* lifo )
{
    return ((dague_list_item_t *)lifo->lifo_head.data.item == lifo->lifo_ghost);
}
#define dague_lifo_nolock_is_empty dague_lifo_is_empty

#if defined(DAGUE_ATOMIC_HAS_ATOMIC_CAS_128B)
/* Add one element to the FIFO. We will return the last head of the list
 * to allow the upper level to detect if this element is the first one in the
 * list (if the list was empty before this operation).
 */
static inline bool
dague_update_counted_pointer(volatile dague_counted_pointer_t *addr, dague_counted_pointer_t old,
                             dague_list_item_t *item)
{
    dague_counted_pointer_t elem = {.data = {.item = item, .counter = old.data.counter + 1}};
    return dague_atomic_cas_128b(&addr->value, old.value, elem.value);
}

static inline void dague_lifo_push( dague_lifo_t* lifo,
                                    dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG_PARANOID)
    assert( (uintptr_t)item % DAGUE_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    DAGUE_ITEM_ATTACH(lifo, item);

    do {
        dague_list_item_t *next = (dague_list_item_t *) lifo->lifo_head.data.item;

        item->list_next = next;
        dague_atomic_wmb ();

        /* to protect against ABA issues it is sufficient to only update the counter in pop */
        if (dague_atomic_cas_64b((uint64_t*)&lifo->lifo_head.data.item, (uint64_t)next, (uint64_t)item)) {
            return;
        }
        /* DO some kind of pause to release the bus */
    } while (1);
}
static inline void dague_lifo_chain( dague_lifo_t* lifo,
                                     dague_list_item_t* ring)
{
#if defined(DAGUE_DEBUG_PARANOID)
    assert( (uintptr_t)ring % DAGUE_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    DAGUE_ITEMS_ATTACH(lifo, ring);

    dague_list_item_t* tail = (dague_list_item_t*)ring->list_prev;

    do {
        dague_list_item_t *next = (dague_list_item_t *) lifo->lifo_head.data.item;

        tail->list_next = next;
        dague_atomic_wmb ();

        /* to protect against ABA issues it is sufficient to only update the counter in pop */
        if (dague_atomic_cas_64b((uint64_t*)&lifo->lifo_head.data.item, (uint64_t)next, (uint64_t)ring)) {
            return;
        }
        /* DO some kind of pause to release the bus */
    } while (1);
}

static inline dague_list_item_t* dague_lifo_pop( dague_lifo_t* lifo )
{
    dague_list_item_t *item;

    do {
        dague_counted_pointer_t old_head;

        old_head.data.counter = lifo->lifo_head.data.counter;
        dague_atomic_rmb ();
        item = old_head.data.item = lifo->lifo_head.data.item;

        if (item == lifo->lifo_ghost) {
            return NULL;
        }

        if (dague_update_counted_pointer (&lifo->lifo_head, old_head,
                                         (dague_list_item_t *) item->list_next)) {
            dague_atomic_wmb ();
            item->list_next = NULL;
            return item;
        }
    } while (1);
}

static inline dague_list_item_t* dague_lifo_try_pop( dague_lifo_t* lifo )
{
    dague_counted_pointer_t old_head;
    dague_list_item_t *item;

    old_head.data.counter = lifo->lifo_head.data.counter;
    dague_atomic_rmb();
    item = old_head.data.item = lifo->lifo_head.data.item;

    if (item == lifo->lifo_ghost) {
        return NULL;
    }

    if (dague_update_counted_pointer (&lifo->lifo_head, old_head,
                                     (dague_list_item_t *) item->list_next)) {
        dague_atomic_wmb();
        item->list_next = NULL;
        return item;
    }
    return NULL;
}

#else  /* !defined(DAGUE_ATOMIC_HAS_ATOMIC_CAS_128B) */

/* Add one element to the LIFO. We will return the last head of the list
 * to allow the upper level to detect if this element is the first one in the
 * list (if the list was empty before this operation).
 */
static inline void dague_lifo_push(dague_lifo_t *lifo,
                                   dague_list_item_t *item)
{
    /* item free acts as a mini lock to avoid ABA problems */
    item->aba_key = 1;
     do {
        dague_list_item_t *next = (dague_list_item_t *) lifo->lifo_head.data.item;
        item->list_next = next;
        dague_atomic_wmb();
         if( dague_atomic_cas(&lifo->lifo_head.data.item, next, item) ) {
            dague_atomic_wmb();
            /* now safe to pop this item */
            item->aba_key = 0;
            return;
        }
        /* DO some kind of pause to release the bus */
    } while (1);
}

static inline void dague_lifo_chain( dague_lifo_t* lifo,
                                     dague_list_item_t* ring)
{
#if defined(DAGUE_DEBUG_PARANOID)
    assert( (uintptr_t)ring % DAGUE_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    DAGUE_ITEMS_ATTACH(lifo, ring);

    /* item free acts as a mini lock to avoid ABA problems */
    ring->aba_key = 1;
    dague_list_item_t* tail = (dague_list_item_t*)ring->list_prev;

     do {
        dague_list_item_t *next = (dague_list_item_t *) lifo->lifo_head.data.item;
        tail->list_next = next;
        dague_atomic_wmb();
         if( dague_atomic_cas(&lifo->lifo_head.data.item, next, ring) ) {
            dague_atomic_wmb();
            /* now safe to pop this item */
            ring->aba_key = 0;
            return;
        }
        /* DO some kind of pause to release the bus */
    } while (1);
}

#if DAGUE_HAVE_ATOMIC_LLSC_PTR

static inline void _dague_lifo_release_cpu (void)
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
static inline dague_list_item_t *dague_lifo_pop(dague_lifo_t* lifo)
{
    dague_list_item_t *item, *next;
    int attempt = 0;

    do {
        if (++attempt == 5) {
            /* deliberatly suspend this thread to allow other threads to run. this should
             * only occur during periods of contention on the lifo. */
            _dague_lifo_release_cpu ();
            attempt = 0;
        }

        item = (dague_list_item_t *) dague_atomic_ll_ptr (&lifo->lifo_head.data.item);
        if (&lifo->lifo_ghost == item) {
            return NULL;
        }

        next = (dague_list_item_t *) item->list_next;
    } while (!dague_atomic_sc_ptr (&lifo->lifo_head.data.item, next));

    dague_atomic_wmb ();

    item->list_next = NULL;
    return item;
}

static inline dague_list_item_t* dague_lifo_try_pop( dague_lifo_t* lifo )
{
    dague_list_item_t *item, *next;
    int attempt = 0;

    item = (dague_list_item_t *) dague_atomic_ll_ptr (&lifo->lifo_head.data.item);
    if (&lifo->lifo_ghost == item) {
        return NULL;
    }

    next = (dague_list_item_t *) item->list_next;
    if( !dague_atomic_sc_ptr (&lifo->lifo_head.data.item, next) )
        return NULL;

    dague_atomic_wmb ();

    item->list_next = NULL;
    return item;
}
#else

/* Retrieve one element from the LIFO. If we reach the ghost element then the LIFO
 * is empty so we return NULL.
 */
static inline dague_list_item_t *dague_lifo_pop(dague_lifo_t* lifo)
{
    dague_list_item_t *item;
    while ((item = lifo->lifo_head.data.item) != lifo->lifo_ghost) {
        /* ensure it is safe to pop the head */
        if (dague_atomic_cas((volatile int32_t *) &item->aba_key, 0, 1)) {
            continue;
        }

        dague_atomic_wmb ();

        /* try to swap out the head pointer */
        if( dague_atomic_cas(&lifo->lifo_head.data.item, item,
                                   (void *) item->list_next) ) {
            break;
        }

        /* NTH: don't need another atomic here */
        item->aba_key = 0;

        /* Do some kind of pause to release the bus */
    }

    if (item == lifo->lifo_ghost) {
        return NULL;
    }

    dague_atomic_wmb ();

    item->list_next = NULL;
    return item;
}

static inline dague_list_item_t* dague_lifo_try_pop( dague_lifo_t* lifo )
{
    dague_list_item_t *item;
    if( (item = lifo->lifo_head.data.item) != lifo->lifo_ghost ) {
        /* ensure it is safe to pop the head */
        if (dague_atomic_cas((volatile int32_t *) &item->aba_key, 0, 1)) {
            return NULL;
        }

        dague_atomic_wmb ();

        /* try to swap out the head pointer */
        if( dague_atomic_cas(&lifo->lifo_head.data.item, item,
                                   (void *) item->list_next) ) {
            return NULL;
        }

        /* NTH: don't need another atomic here */
        item->aba_key = 0;

        /* Do some kind of pause to release the bus */
    }

    if (item == lifo->lifo_ghost) {
        return NULL;
    }

    dague_atomic_wmb ();

    item->list_next = NULL;
    return item;
}
#endif /* DAGUE_HAVE_ATOMIC_LLSC_PTR */


#endif  /* defined(DAGUE_ATOMIC_HAS_ATOMIC_CAS_128B)) */

static inline void dague_lifo_nolock_push( dague_lifo_t* lifo,
                                           dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG_PARANOID)
    assert( (uintptr_t)item % DAGUE_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    DAGUE_ITEM_ATTACH(lifo, item);

    item->list_next = lifo->lifo_head.data.item;
    lifo->lifo_head.data.item = item;
}

static inline void dague_lifo_nolock_chain( dague_lifo_t* lifo,
                                            dague_list_item_t* items )
{
#if defined(DAGUE_DEBUG_PARANOID)
    assert( (uintptr_t)items % DAGUE_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    DAGUE_ITEMS_ATTACH(lifo, items);

    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;

    tail->list_next = lifo->lifo_head.data.item;
    lifo->lifo_head.data.item = items;
}

static inline dague_list_item_t* dague_lifo_nolock_pop( dague_lifo_t* lifo )
{
    dague_list_item_t* item = lifo->lifo_head.data.item;
    lifo->lifo_head.data.item = (dague_list_item_t*)item->list_next;
    DAGUE_ITEM_DETACH(item);
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
#define DAGUE_LIFO_ITEM_ALLOC( LIFO, elt, truesize ) ({         \
    void *_elt = NULL;                                          \
    int _rc;                                                    \
    _rc = posix_memalign(&_elt,                                 \
                         DAGUE_LIFO_ALIGNMENT(LIFO), (truesize));\
    assert( 0 == _rc && NULL != _elt ); (void)_rc;              \
    OBJ_CONSTRUCT(_elt, dague_list_item_t);                     \
    (elt) = (__typeof__(elt))_elt;                              \
  })
#define DAGUE_LIFO_ITEM_FREE( elt ) do { OBJ_DESTRUCT( elt ); free(elt); } while (0)

#endif  /* LIFO_H_HAS_BEEN_INCLUDED */
