/*
 * Copyright (c) 2009-2014 The University of Tennessee and The University
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

#if defined(DAGUE_ATOMIC_HAS_CAS128B)
typedef __uint128_t dague_lifo_head_t;
#define __dague_lifo_cas dague_atomic_cas_128b
#else
typedef dague_list_item_t* dague_lifo_head_t;
#define __dague_lifo_cas dague_atomic_cas
#endif /*defined(DAGUE_ATOMIC_HAS_CAS128B)*/

struct dague_lifo_s {
    dague_object_t     super;
    uint8_t            alignment;
    dague_list_item_t *lifo_ghost;
    dague_lifo_head_t  lifo_head;
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
#define DAGUE_LIFO_ALIGNMENT(LIFO)       (( ( ((uintptr_t)1 << DAGUE_LIFO_ALIGNMENT_BITS(LIFO) ) < sizeof(void*) ) ? \
                                            ( sizeof(void*) ) :         \
                                            ( (uintptr_t)1 << DAGUE_LIFO_ALIGNMENT_BITS(LIFO) ) ))
#if defined(DAGUE_ATOMIC_HAS_CAS128B)
#define DAGUE_LIFO_PTR(LIFO, v) ((dague_list_item_t*)(uintptr_t)(v))
#define DAGUE_LIFO_VAL(LIFO, v, k) ((((dague_lifo_head_t)((uint64_t)k))<<64)+((dague_lifo_head_t)(uintptr_t)v))
#else
#define DAGUE_LIFO_CNTMASK(LIFO)         (DAGUE_LIFO_ALIGNMENT(LIFO)-1)
#define DAGUE_LIFO_PTRMASK(LIFO)         (~(DAGUE_LIFO_CNTMASK(LIFO)))
#define DAGUE_LIFO_CNT(LIFO, v)          ((uintptr_t)((uintptr_t)(v) & DAGUE_LIFO_CNTMASK(LIFO)))
#define DAGUE_LIFO_PTR(LIFO, v)          ((dague_list_item_t *)((uintptr_t)(v) & DAGUE_LIFO_PTRMASK(LIFO)))
#define DAGUE_LIFO_VAL(LIFO, p, c)       ((dague_list_item_t *)(((uintptr_t)DAGUE_LIFO_PTR(LIFO, p)) | DAGUE_LIFO_CNT(LIFO, c)))
#endif /*defined(DAGUE_ATOMIC_HAS_CAS128B)*/

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
            int _rc;                                                    \
            _rc = posix_memalign(&_elt,                                 \
                                DAGUE_LIFO_ALIGNMENT(LIFO), (truesize));\
            assert( 0 == _rc && NULL != _elt ); (void)_rc;              \
            OBJ_CONSTRUCT(_elt, dague_list_item_t);                     \
            (elt) = (__typeof__(elt))_elt;                              \
        })
#define DAGUE_LIFO_ITEM_FREE( elt ) do { OBJ_DESTRUCT( elt ); free(elt); } while (0)

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

    dague_lifo_head_t nhead = DAGUE_LIFO_VAL(lifo, item, ++(item->aba_key));

    do {
        dague_lifo_head_t ohead = lifo->lifo_head;
        item->list_next = DAGUE_LIFO_PTR(lifo, ohead);
        if( __dague_lifo_cas(&(lifo->lifo_head),
                             ohead,
                             nhead) ) {
            return;
        }
        /* DO some kind of pause to release the bus */
    } while( 1 );
}
static inline void dague_lifo_nolock_push( dague_lifo_t* lifo,
                                           dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert( (uintptr_t)item % DAGUE_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    DAGUE_ITEM_ATTACH(lifo, item);

    item->list_next = DAGUE_LIFO_PTR(lifo, lifo->lifo_head);
    lifo->lifo_head = DAGUE_LIFO_VAL(lifo, item, 0);
}

static inline void dague_lifo_chain( dague_lifo_t* lifo,
                                     dague_list_item_t* items )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert( (uintptr_t)items % DAGUE_LIFO_ALIGNMENT(lifo) == 0 );
#endif
    DAGUE_ITEMS_ATTACH(lifo, items);

    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
    dague_lifo_head_t nhead = DAGUE_LIFO_VAL(lifo, items, ++(items->aba_key));

    do {
        dague_lifo_head_t ohead = lifo->lifo_head;
        tail->list_next = DAGUE_LIFO_PTR(lifo, ohead);

        if( __dague_lifo_cas(&(lifo->lifo_head),
                             ohead,
                             nhead) ) {
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

    tail->list_next = DAGUE_LIFO_PTR(lifo, lifo->lifo_head);
    lifo->lifo_head = DAGUE_LIFO_VAL(lifo, items, 0);
}

static inline dague_list_item_t* dague_lifo_pop( dague_lifo_t* lifo )
{
    dague_list_item_t *item, *nitem;
    dague_lifo_head_t ohead, nhead;

    ohead = lifo->lifo_head;
    item = DAGUE_LIFO_PTR(lifo, ohead);
    while(item != lifo->lifo_ghost) {
        nitem = DAGUE_LIST_ITEM_NEXT(item);
        nhead = DAGUE_LIFO_VAL(lifo, nitem, nitem->aba_key); /* if aba_key changed, ohead is not current anymore and nhead is discarded */
        if( __dague_lifo_cas(&(lifo->lifo_head),
                             ohead,
                             nhead ) )
            break;
        ohead = lifo->lifo_head;
        item = DAGUE_LIFO_PTR(lifo, ohead);
        /* Do some kind of pause to release the bus */
    }
    if( item == lifo->lifo_ghost ) return NULL;
    DAGUE_ITEM_DETACH(item);
    return item;
}

static inline dague_list_item_t* dague_lifo_try_pop( dague_lifo_t* lifo )
{
    dague_list_item_t *item, *nitem;
    dague_lifo_head_t ohead, nhead;

    ohead = lifo->lifo_head;
    item = DAGUE_LIFO_PTR(lifo, ohead);
    
    if( item == lifo->lifo_ghost )
        return NULL;
    
    nitem = DAGUE_LIST_ITEM_NEXT(item);
    nhead = DAGUE_LIFO_VAL(lifo, nitem, nitem->aba_key); /* if aba_key changed, ohead is not current anymore and nhead is discarded */
    if( __dague_lifo_cas(&(lifo->lifo_head),
                         ohead,
                         nhead ) ) { 
        DAGUE_ITEM_DETACH(item);
        return item;
    }
    return NULL;
}

static inline dague_list_item_t* dague_lifo_nolock_pop( dague_lifo_t* lifo )
{
    dague_list_item_t* item = DAGUE_LIFO_PTR(lifo, lifo->lifo_head);
    lifo->lifo_head = DAGUE_LIFO_VAL(lifo, item->list_next, 0);
    DAGUE_ITEM_DETACH(item);
    return item;
}

#endif  /* LIFO_H_HAS_BEEN_INCLUDED */
