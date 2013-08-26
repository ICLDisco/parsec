/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef LIFO_H_HAS_BEEN_INCLUDED
#define LIFO_H_HAS_BEEN_INCLUDED

#include <dague_config.h>
#include <dague/class/dague_object.h>
#include "list_item.h"

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

#include <stdlib.h>
#include <dague/sys/atomic.h>

struct dague_lifo_s {
    dague_object_t     super;
    uint8_t            alignment;
    dague_list_item_t *lifo_ghost;
    dague_list_item_t *lifo_head;
};

/**
 * By default all LIFO will handle elements aligned to DAGUE_LIFO_ALIGNMENT_DEFAULT
 * bits. If a different type of alignment is needed, the trick is to manually allocate
 * the lifo and set the alignment by hand before calling OBJ_CONSTRUCT on it.
 */
#if !defined(DAGUE_LIFO_ALIGNMENT_DEFAULT)
#define DAGUE_LIFO_ALIGNMENT_DEFAULT 3
#endif  /* !defined(DAGUE_LIFO_ALIGNMENT_DEFAULT) */

#define DAGUE_LIFO_ALIGNMENT_BITS(LIFO)  ((LIFO)->alignment)
#define DAGUE_LIFO_ALIGNMENT(LIFO)       (1 << DAGUE_LIFO_ALIGNMENT_BITS(LIFO) )
#define DAGUE_LIFO_CNTMASK(LIFO)         (DAGUE_LIFO_ALIGNMENT(LIFO)-1)
#define DAGUE_LIFO_PTRMASK(LIFO)         (~(DAGUE_LIFO_CNTMASK(LIFO)))
#define DAGUE_LIFO_CNT(LIFO, v)          ((uintptr_t)((uintptr_t)(v) & DAGUE_LIFO_CNTMASK(LIFO)))
#define DAGUE_LIFO_PTR(LIFO, v)          ((dague_list_item_t *)((uintptr_t)(v) & DAGUE_LIFO_PTRMASK(LIFO)))
#define DAGUE_LIFO_VAL(LIFO, p, c)       ((dague_list_item_t *)(((uintptr_t)DAGUE_LIFO_PTR(LIFO, p)) | DAGUE_LIFO_CNT(LIFO, c)))

/*
 * http://stackoverflow.com/questions/10528280/why-is-the-below-code-giving-dereferencing-type-punned-pointer-will-break-stric
 *
 * void * converts to any pointer type, and any pointer type converts
 * to void *, but void ** does not convert to a pointer to some other
 * type of pointer, nor do pointers to other pointer types convert to
 * void **.
 */
#define DAGUE_LIFO_ITEM_ALLOC( LIFO, elt, truesize ) ({                 \
            void *_elt = NULL;                                          \
            if( 0 == posix_memalign(&_elt,                              \
                                    DAGUE_LIFO_ALIGNMENT(LIFO), (truesize)) ) { \
                assert( NULL != _elt );                                 \
                OBJ_CONSTRUCT(_elt, dague_list_item_t);                 \
            }                                                           \
            (elt) = (__typeof__(elt))_elt;                              \
        })
#define DAGUE_LIFO_ITEM_FREE( elt ) free(elt)


/* The ghost pointer will never change. The head will change via an
 * atomic compare-and-swap. On most architectures the reading of a
 * pointer is an atomic operation so we don't have to protect it. */
static inline int dague_lifo_is_empty( dague_lifo_t* lifo )
{
    return ( (DAGUE_LIFO_PTR(lifo, lifo->lifo_head) == lifo->lifo_ghost) ? 1 : 0);
}
static inline int dague_lifo_nolock_is_empty( dague_lifo_t* lifo )
{
    return dague_lifo_is_empty(lifo);
}

static inline void dague_lifo_push( dague_lifo_t* lifo,
                                    dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert( (uintptr_t)item % DAGUE_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    DAGUE_ITEM_ATTACH(lifo, item);

    dague_list_item_t* tp = DAGUE_LIFO_VAL(lifo, item, (item->keeper_of_the_seven_keys + 1));

    do {
        item->list_next = lifo->lifo_head;
        if( dague_atomic_cas(&(lifo->lifo_head),
                             (uintptr_t)item->list_next,
                             (uintptr_t)tp) ) {
            return;
        }
        /* DO some kind of pause to release the bus */
    } while( 1 );
}
static inline void dague_lifo_nolock_push( dague_lifo_t* lifo,
                                           dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert( (uintptr_t)item % DAGUE_LIFO_ALIGNMENT(LIFO) == 0 );
#endif
    DAGUE_ITEM_ATTACH(lifo, item);

    item->list_next = lifo->lifo_head;
    lifo->lifo_head = item;
}

static inline void dague_lifo_chain( dague_lifo_t* lifo,
                                     dague_list_item_t* items )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert( (uintptr_t)items % DAGUE_LIFO_ALIGNMENT(LIFO) == 0 );
#endif
    DAGUE_ITEMS_ATTACH(lifo, items);

    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
    dague_list_item_t* tp = DAGUE_LIFO_VAL(lifo, items, (items->keeper_of_the_seven_keys + 1));

    do {
        tail->list_next = lifo->lifo_head;
        if( dague_atomic_cas(&(lifo->lifo_head),
                             (uintptr_t) tail->list_next,
                             (uintptr_t) tp) ) {
            return;
        }
        /* DO some kind of pause to release the bus */
    } while( 1 );
}
static inline void dague_lifo_nolock_chain( dague_lifo_t* lifo,
                                            dague_list_item_t* items )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert( (uintptr_t)items % DAGUE_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    DAGUE_ITEMS_ATTACH(lifo, items);

    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;

    tail->list_next = lifo->lifo_head;
    lifo->lifo_head = items;
}

static inline dague_list_item_t* dague_lifo_pop( dague_lifo_t* lifo )
{
    dague_list_item_t *item, *save;

    item = lifo->lifo_head;
    while(DAGUE_LIFO_PTR(lifo, item) != lifo->lifo_ghost) {
        if( dague_atomic_cas(&(lifo->lifo_head),
                             (uintptr_t) item,
                             (uintptr_t) DAGUE_LIFO_PTR(lifo, item)->list_next ) )
            break;
        item = lifo->lifo_head;
        /* Do some kind of pause to release the bus */
    }
    save = item;
    item = DAGUE_LIFO_PTR(lifo, item);
    if( item == lifo->lifo_ghost ) return NULL;
    item->keeper_of_the_seven_keys = DAGUE_LIFO_CNT(lifo, save);
    DAGUE_ITEM_DETACH(item);
    return item;
}

static inline dague_list_item_t* dague_lifo_try_pop( dague_lifo_t* lifo )
{
    dague_list_item_t *item, *save;

    item = lifo->lifo_head;
    if( DAGUE_LIFO_PTR(lifo, item) == lifo->lifo_ghost )
        return NULL;

    if( dague_atomic_cas(&(lifo->lifo_head),
                         (uintptr_t) item,
                         (uintptr_t) DAGUE_LIFO_PTR(lifo, item)->list_next) )
    {
        save = item;
        item = DAGUE_LIFO_PTR(lifo, item);
        item->keeper_of_the_seven_keys = DAGUE_LIFO_CNT(lifo, save);
        DAGUE_ITEM_DETACH(item);
        return item;
    }
    return NULL;
}

static inline dague_list_item_t* dague_lifo_nolock_pop( dague_lifo_t* lifo )
{
    dague_list_item_t* item = lifo->lifo_head;
    lifo->lifo_head = (dague_list_item_t*)item->list_next;
    DAGUE_ITEM_DETACH(item);
    return item;
}

#endif  /* LIFO_H_HAS_BEEN_INCLUDED */
