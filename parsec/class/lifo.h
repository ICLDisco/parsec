/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef LIFO_H_HAS_BEEN_INCLUDED
#define LIFO_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/class/list_item.h"
#include "parsec/sys/atomic.h"
#if defined(PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR)
#include <time.h>
#endif  /* defined(PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR) */

/**
 * @defgroup parsec_internal_classes_lifo Last In First Out
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief Last In First out parsec_list_item_t management functions
 *
 *  @details There are two interfaces for LIFO: an atomic-based
 *           lock-free implementation (this file), and a lock-based
 *           lists emulation (in list.h). If you need to use
 *           list-compatible access in the LIFO, use the list.h
 *           implementation; otherwise, use this implementation.
 */

#if !defined(BUILDING_PARSEC)
#  include "parsec/class/lifo-external.h"
#else  /* !defined(BUILDING_PARSEC) */

#  if !defined(LIFO_STATIC_INLINE)
#    define LIFO_STATIC_INLINE static inline
#  endif  /* !defined(LIFO_STATIC_INLINE) */

BEGIN_C_DECLS

/**
 * @brief opaque structure to hold a LIFO
 */
typedef struct parsec_lifo_s parsec_lifo_t;
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_lifo_t);

/**
 * @brief check if the LIFO is empty
 *
 * @param[inout] lifo the LIFO to check
 * @return 0 if lifo is not empty, 1 otherwise
 *
 * @remark this function is thread safe
 */
LIFO_STATIC_INLINE int
parsec_lifo_is_empty( parsec_lifo_t* lifo );

/**
 * @brief check if the LIFO is empty, without forcing atomicity.
 *
 * @param[inout] lifo the LIFO to check
 * @return 0 if lifo is not empty, 1 otherwise
 *
 * @remark this function is not thread safe
 */
PARSEC_DECLSPEC int
parsec_nolock_lifo_is_empty( parsec_lifo_t* lifo );

/**
 * @brief Push an element in the LIFO
 *
 * @details push an element at the front of the LIFO
 *
 * @param[inout] lifo the LIFO into which to push the element
 * @param[inout] item the element to push in lifo
 *
 * @remark this function is thread safe
 */
LIFO_STATIC_INLINE void
parsec_lifo_push(parsec_lifo_t* lifo, parsec_list_item_t* item);

/**
 * @brief Push an element in the LIFO, without forcing atomicity.
 *
 * @details push an element at the front of the LIFO
 *
 * @param[inout] lifo the LIFO into which to push the element
 * @param[inout] item the element to push in lifo
 *
 * @remark this function is not thread safe
 */
LIFO_STATIC_INLINE void
parsec_lifo_nolock_push(parsec_lifo_t* lifo, parsec_list_item_t* item);

/**
 * @brief Chain a ring of elements in front of a LIFO
 *
 * @details Take a ring of elements (items->prev points to the last
 *          element in items), and push all the elements of items in
 *          front of the LIFO, preserving the order in items.
 *
 * @param[inout] lifo the LIFO into which to push the elements
 * @param[inout] items the elements ring to push in front
 *
 * @remark this function is thread safe
 */
LIFO_STATIC_INLINE void
parsec_lifo_chain(parsec_lifo_t* lifo, parsec_list_item_t* items);

/**
 * @brief Chain a ring of elements in front of a LIFO, without
 *        forcing atomicity.
 *
 * @details Take a ring of elements (items->prev points to the last
 *          element in items), and push all the elements of items in
 *          front of the LIFO, preserving the order in items.
 *
 * @param[inout] lifo the LIFO into which to push the elements
 * @param[inout] items the elements ring to push in front
 *
 * @remark this function is not thread safe
 */
LIFO_STATIC_INLINE void
parsec_lifo_nolock_chain(parsec_lifo_t* lifo, parsec_list_item_t* items);

/**
 * @brief Pop an element from the LIFO
 *
 * @details Pop the first element in the LIFO
 *
 * @param[inout] lifo the LIFO from which to pop the element
 * @return the element that was removed from the LIFO (NULL if
 *         the LIFO was empty)
 *
 * @remark this function is thread safe
 */
LIFO_STATIC_INLINE parsec_list_item_t*
parsec_lifo_pop(parsec_lifo_t* lifo);

/**
 * @brief Try popping an element from the LIFO
 *
 * @details Try popping the first element in the LIFO
 *
 * @param[inout] lifo the LIFO from which to pop the element
 * @return the element that was removed from the LIFO (NULL if
 *         the LIFO was empty)
 *
 * @remark this function is thread safe
 */
LIFO_STATIC_INLINE parsec_list_item_t*
parsec_lifo_try_pop(parsec_lifo_t* lifo);

/**
 * @brief Pop an element from the LIFO, without forcing atomicity.
 *
 * @details Pop the first element in the LIFO
 *
 * @param[inout] lifo the LIFO from which to pop the element
 * @return the element that was removed from the LIFO (NULL if
 *         the LIFO was empty)
 *
 * @remark this function is not thread safe
 */
LIFO_STATIC_INLINE parsec_list_item_t*
parsec_lifo_nolock_pop(parsec_lifo_t* lifo);

/**
 * @cond FALSE
 ***********************************************************************
 * Interface is defined. Everything else is private thereafter
 */

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

/**
 * This code is imported from Open MPI.
 */

/**
 * Counted pointer to avoid the ABA problem.
 */
typedef union parsec_counted_pointer_u {
    struct {
        union {
            /** update counter used when cmpset_128 is available */
            int64_t             counter;
            /* The lock used if 128-bit atomics are not available
             * and the item's aba-key is not used. */
            parsec_atomic_lock_t lock;
        } guard;
        /** list item pointer */
        parsec_list_item_t *item;
    } data;
#if defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128)
    /** used for atomics when there is a cmpset that can operate on
     * two 64-bit values */
    __int128_t value;
#endif  /* defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128) */
} parsec_counted_pointer_t;

struct parsec_lifo_s {
    parsec_object_t           super;
    uint8_t                   alignment;
    parsec_list_item_t       *lifo_ghost;
    parsec_counted_pointer_t  lifo_head;
};

/* The ghost pointer will never change. The head will change via an
 * atomic compare-and-swap. On most architectures the reading of a
 * pointer is an atomic operation so we don't have to protect it. */
LIFO_STATIC_INLINE int parsec_lifo_is_empty( parsec_lifo_t* lifo )
{
    return ((parsec_list_item_t *)lifo->lifo_head.data.item == lifo->lifo_ghost);
}

/* Same as above, we need an actual function in the external case */
LIFO_STATIC_INLINE int parsec_lifo_nolock_is_empty( parsec_lifo_t* lifo ) {
    return ((parsec_list_item_t *)lifo->lifo_head.data.item == lifo->lifo_ghost);
}

#if defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128)
/* Add one element to the FIFO. Returns true if successful, false otherwise.
 */
LIFO_STATIC_INLINE int
parsec_update_counted_pointer(volatile parsec_counted_pointer_t *addr, parsec_counted_pointer_t old,
                             parsec_list_item_t *item)
{
    parsec_counted_pointer_t elem = {.data = {.guard = {.counter = old.data.guard.counter + 1}, .item = item}};
    return parsec_atomic_cas_int128(&addr->value, old.value, elem.value);
}

LIFO_STATIC_INLINE void parsec_lifo_push( parsec_lifo_t* lifo,
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
LIFO_STATIC_INLINE void parsec_lifo_chain( parsec_lifo_t* lifo,
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

LIFO_STATIC_INLINE parsec_list_item_t* parsec_lifo_pop( parsec_lifo_t* lifo )
{
    parsec_list_item_t *item;
    parsec_counted_pointer_t old_head;

    do {

        old_head.data.guard.counter = lifo->lifo_head.data.guard.counter;
        parsec_atomic_rmb ();
        item = old_head.data.item = lifo->lifo_head.data.item;

        if (item == lifo->lifo_ghost) {
            return NULL;
        }

        if (parsec_update_counted_pointer(&lifo->lifo_head, old_head,
                                          (parsec_list_item_t *)item->list_next)) {
            parsec_atomic_wmb ();
            item->list_next = NULL;
            PARSEC_ITEM_DETACH(item);
            return item;
        }
    } while (1);
}

LIFO_STATIC_INLINE parsec_list_item_t* parsec_lifo_try_pop( parsec_lifo_t* lifo )
{
    parsec_counted_pointer_t old_head;
    parsec_list_item_t *item;

    old_head.data.guard.counter = lifo->lifo_head.data.guard.counter;
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

#elif defined(PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR)

LIFO_STATIC_INLINE void _parsec_lifo_release_cpu (void)
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

/* Add one element to the LIFO. We will return the last head of the list
 * to allow the upper level to detect if this element is the first one in the
 * list (if the list was empty before this operation).
 */
LIFO_STATIC_INLINE void parsec_lifo_push(parsec_lifo_t *lifo,
                                    parsec_list_item_t *item)
{
    int attempt = 0;
    PARSEC_ITEM_ATTACH(lifo, item);

    do {
        if( ++attempt == 5 ) {
            /* deliberatly suspend this thread to allow other threads to run. this should
             * only occur during periods of contention on the lifo. */
            _parsec_lifo_release_cpu ();
            attempt = 0;
        }
        parsec_list_item_t *next = (parsec_list_item_t *) parsec_atomic_ll_ptr((long*)&lifo->lifo_head.data.item);
        item->list_next = next;
        parsec_atomic_wmb();
    } while( !parsec_atomic_sc_ptr((long*)&lifo->lifo_head.data.item, (intptr_t)item) );
}

LIFO_STATIC_INLINE void parsec_lifo_chain( parsec_lifo_t* lifo,
                                     parsec_list_item_t* ring)
{
    int attempt = 0;
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)ring % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    PARSEC_ITEMS_ATTACH(lifo, ring);

    parsec_list_item_t* tail = (parsec_list_item_t*)ring->list_prev;

    do {
        if( ++attempt == 5 ) {
            /* deliberatly suspend this thread to allow other threads to run. this should
             * only occur during periods of contention on the lifo. */
            _parsec_lifo_release_cpu ();
            attempt = 0;
        }
        parsec_list_item_t *next = (parsec_list_item_t *) parsec_atomic_ll_ptr((long*)&lifo->lifo_head.data.item);
        tail->list_next = next;
        parsec_atomic_wmb();
    } while( !parsec_atomic_sc_ptr((long*)&lifo->lifo_head.data.item, (intptr_t)ring) );
}

/* Retrieve one element from the LIFO. If we reach the ghost element then the LIFO
 * is empty so we return NULL.
 */
LIFO_STATIC_INLINE parsec_list_item_t *parsec_lifo_pop(parsec_lifo_t* lifo)
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

LIFO_STATIC_INLINE parsec_list_item_t* parsec_lifo_try_pop( parsec_lifo_t* lifo )
{
    parsec_list_item_t *item, *next;

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

#elif defined(PARSEC_USE_64BIT_LOCKFREE_LIST)

/* Add one element to the LIFO. We will return the last head of the list
 * to allow the upper level to detect if this element is the first one in the
 * list (if the list was empty before this operation).
 */
LIFO_STATIC_INLINE void parsec_lifo_push(parsec_lifo_t *lifo,
                                    parsec_list_item_t *item)
{
    PARSEC_ITEM_ATTACH(lifo, item);
    /* item free acts as a mini lock to avoid ABA problems */
    item->aba_key = 1;
    do {
        parsec_list_item_t *next = (parsec_list_item_t *) lifo->lifo_head.data.item;
        item->list_next = next;
        parsec_atomic_wmb();
        if( parsec_atomic_cas_ptr(&lifo->lifo_head.data.item, next, item) ) {
            /* now safe to pop this item */
            item->aba_key = 0;
            return;
        }
        /* DO some kind of pause to release the bus */
    } while (1);
}

LIFO_STATIC_INLINE void parsec_lifo_chain(parsec_lifo_t* lifo,
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
            /* now safe to pop this item */
            ring->aba_key = 0;
            return;
        }
        /* DO some kind of pause to release the bus */
    } while (1);
}

/* Retrieve one element from the LIFO. If we reach the ghost element then the LIFO
 * is empty so we return NULL.
 */
LIFO_STATIC_INLINE parsec_list_item_t *parsec_lifo_pop(parsec_lifo_t* lifo)
{
    parsec_list_item_t *item;

    while ((item = lifo->lifo_head.data.item) != lifo->lifo_ghost) {
        /* ensure it is safe to pop the head */
        if (!parsec_atomic_cas_int32(&item->aba_key, 0UL, 1UL)) {
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

LIFO_STATIC_INLINE parsec_list_item_t* parsec_lifo_try_pop( parsec_lifo_t* lifo )
{
    parsec_list_item_t *item;

    if( (item = lifo->lifo_head.data.item) != lifo->lifo_ghost ) {
        /* ensure it is safe to pop the head */
        if (!parsec_atomic_cas_int32(&item->aba_key, 0UL, 1UL)) {
            return NULL;
        }

        parsec_atomic_wmb ();

        /* try to swap out the head pointer */
        if( !parsec_atomic_cas_ptr(&lifo->lifo_head.data.item,
                                   item, (void *)item->list_next) ) {
            item->aba_key = 0UL;
            return NULL;
        }

        /* NTH: don't need another atomic here */
        item->aba_key = 0UL;
        parsec_atomic_wmb ();

        item->list_next = NULL;
        PARSEC_ITEM_DETACH(item);
    }

    return item == lifo->lifo_ghost ? NULL : item;
}

#else /* defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128) || defined(PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR) || defined(PARSEC_USE_64BIT_LOCKFREE_LIST) */

/* Add one element to the LIFO. We will return the last head of the list
 * to allow the upper level to detect if this element is the first one in the
 * list (if the list was empty before this operation).
 */
LIFO_STATIC_INLINE void parsec_lifo_push(parsec_lifo_t *lifo,
                                    parsec_list_item_t *item)
{
    PARSEC_ITEM_ATTACH(lifo, item);
    parsec_atomic_lock(&lifo->lifo_head.data.guard.lock);
    parsec_list_item_t *next = (parsec_list_item_t *) lifo->lifo_head.data.item;
    item->list_next = next;
    lifo->lifo_head.data.item = item;
    parsec_atomic_unlock(&lifo->lifo_head.data.guard.lock);
}

LIFO_STATIC_INLINE void parsec_lifo_chain(parsec_lifo_t* lifo,
                                     parsec_list_item_t* ring)
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)ring % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    PARSEC_ITEMS_ATTACH(lifo, ring);

    parsec_atomic_lock(&lifo->lifo_head.data.guard.lock);
    parsec_list_item_t* tail = (parsec_list_item_t*) ring->list_prev;

    parsec_list_item_t *next = (parsec_list_item_t*) lifo->lifo_head.data.item;
    tail->list_next = next;
    lifo->lifo_head.data.item = ring;
    parsec_atomic_unlock(&lifo->lifo_head.data.guard.lock);
}

/* Retrieve one element from the LIFO. If we reach the ghost element then the LIFO
 * is empty so we return NULL.
 */
LIFO_STATIC_INLINE parsec_list_item_t *parsec_lifo_pop(parsec_lifo_t* lifo)
{
    parsec_list_item_t *item;

    /* Short-cut if empty to avoid lock-thrashing */
    if (lifo->lifo_head.data.item == lifo->lifo_ghost) {
        return NULL;
    }

    parsec_atomic_lock(&lifo->lifo_head.data.guard.lock);
    if ((item = lifo->lifo_head.data.item) != lifo->lifo_ghost) {
        lifo->lifo_head.data.item = (parsec_list_item_t*)item->list_next;
        item->list_next = NULL;
        PARSEC_ITEM_DETACH(item);
    } else {
        item = NULL;
    }
    parsec_atomic_unlock(&lifo->lifo_head.data.guard.lock);
    return NULL;
}

LIFO_STATIC_INLINE parsec_list_item_t *parsec_lifo_try_pop(parsec_lifo_t* lifo)
{
    parsec_list_item_t *item;

    /* Short-cut if empty to avoid lock-thrashing */
    if (lifo->lifo_head.data.item == lifo->lifo_ghost) {
        return NULL;
    }

    if (!parsec_atomic_trylock(&lifo->lifo_head.data.guard.lock)) {
        return NULL;
    }

    if ((item = lifo->lifo_head.data.item) != lifo->lifo_ghost) {
        lifo->lifo_head.data.item = (parsec_list_item_t*)item->list_next;
        item->list_next = NULL;
        PARSEC_ITEM_DETACH(item);
    } else {
        item = NULL;
    }
    parsec_atomic_unlock(&lifo->lifo_head.data.guard.lock);
    return NULL;

}

#endif  /* defined(PARSEC_ATOMIC_HAS_ATOMIC_CAS_INT128) || defined(PARSEC_ATOMIC_HAS_ATOMIC_LLSC_PTR) || defined(PARSEC_USE_64BIT_LOCKFREE_LIST) */

LIFO_STATIC_INLINE void parsec_lifo_nolock_push( parsec_lifo_t* lifo,
                                            parsec_list_item_t* item )
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (uintptr_t)item % PARSEC_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    PARSEC_ITEM_ATTACH(lifo, item);

    item->list_next = lifo->lifo_head.data.item;
    lifo->lifo_head.data.item = item;
}

LIFO_STATIC_INLINE void parsec_lifo_nolock_chain( parsec_lifo_t* lifo,
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

LIFO_STATIC_INLINE parsec_list_item_t* parsec_lifo_nolock_pop( parsec_lifo_t* lifo )
{
    parsec_list_item_t* item = lifo->lifo_head.data.item;
    lifo->lifo_head.data.item = (parsec_list_item_t*)item->list_next;
    PARSEC_ITEM_DETACH(item);
    return item;
}

/**
 * @brief Allocate a lifo item.
 *
 * @details Allocate an element that is correctly aligned to be 
 * used in the lifo. One may change the alignment of elements before
 * allocating the first item in the lifo by changing lifo->alignment.
 *
 * @param[in] lifo the LIFO the element will be used with.
 * @return The element that was allocated.
 */
LIFO_STATIC_INLINE parsec_list_item_t* parsec_lifo_item_alloc( parsec_lifo_t* lifo, size_t truesize) {
    void *elt = NULL;
    int rc;
    rc = posix_memalign(&elt,
                        PARSEC_LIFO_ALIGNMENT(lifo), (truesize));
    assert( 0 == rc && NULL != elt ); (void)rc;
    OBJ_CONSTRUCT(elt, parsec_list_item_t);
    return (parsec_list_item_t*) elt;
}

/**
 * @brief Free a lifo item.
 *
 * @details Free an item that was allocated by parsec_lifo_item_alloc.
 *
 * @param[inout] item the LIFO the element to free.
 *
 * @return none.
 *
 * @remarks The item must not be present in any lifo.
 */
LIFO_STATIC_INLINE void parsec_lifo_item_free(parsec_list_item_t* item) {
    OBJ_DESTRUCT( item );
    free(item);
}

/** @endcond */

END_C_DECLS

/**
 * @}
 */
#endif  /* !defined(BUILDING_PARSEC) */

#endif  /* LIFO_H_HAS_BEEN_INCLUDED */
