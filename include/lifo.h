/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef LIFO_H_HAS_BEEN_INCLUDED
#define LIFO_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

#include "list_item.h"

typedef struct dague_lifo_t dague_lifo_t;

static inline void 
dague_lifo_construct( dague_lifo_t* lifo ); 
static inline void 
dague_lifo_destruct( dague_lifo_t* lifo );

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
 
#ifdef DAGUE_DEBUG_LIFO_USE_ATOMICS

#include <stdlib.h>
#include "atomic.h"

struct dague_lifo_t {
    dague_list_item_t *lifo_head;
    dague_list_item_t *lifo_ghost;
};

#define DAGUE_LIFO_ALIGNMENT_BITS  3
#define DAGUE_LIFO_ALIGNMENT      (1 << DAGUE_LIFO_ALIGNMENT_BITS )
#define DAGUE_LIFO_CNTMASK        (DAGUE_LIFO_ALIGNMENT-1)
#define DAGUE_LIFO_PTRMASK        (~(DAGUE_LIFO_CNTMASK))
#define DAGUE_LIFO_CNT( v )       ( (uintptr_t) ( (uintptr_t)(v) & DAGUE_LIFO_CNTMASK ) )
#define DAGUE_LIFO_PTR( v )       ( (dague_list_item_t *) ( (uintptr_t)(v) & DAGUE_LIFO_PTRMASK ) )
#define DAGUE_LIFO_VAL( p, c)     ( (dague_list_item_t *) ( ((uintptr_t)DAGUE_LIFO_PTR(p)) | DAGUE_LIFO_CNT(c) ) )

/*
 * http://stackoverflow.com/questions/10528280/why-is-the-below-code-giving-dereferencing-type-punned-pointer-will-break-stric
 * 
 * void * converts to any pointer type, and any pointer type converts
 * to void *, but void ** does not convert to a pointer to some other
 * type of pointer, nor do pointers to other pointer types convert to
 * void **.
 */
#define DAGUE_LIFO_ITEM_ALLOC( elt, truesize ) ({                   \
    void *_elt = NULL;                                            \
    if( 0 == posix_memalign(&_elt,                                  \
                            DAGUE_LIFO_ALIGNMENT, (truesize)) ) {   \
        assert( NULL != _elt );                                     \
        dague_list_item_construct((dague_list_item_t*)_elt);        \
    }                                                               \
    (elt) = (__typeof__(elt))_elt;                                  \
})
#define DAGUE_LIFO_ITEM_FREE( elt ) free(elt)


/* The ghost pointer will never change. The head will change via an
 * atomic compare-and-swap. On most architectures the reading of a
 * pointer is an atomic operation so we don't have to protect it. */
static inline int dague_lifo_is_empty( dague_lifo_t* lifo )
{
    return ( (DAGUE_LIFO_PTR(lifo->lifo_head) == lifo->lifo_ghost) ? 1 : 0);
}
static inline int dague_lifo_nolock_is_empty( dague_lifo_t* lifo )
{
    return dague_lifo_is_empty(lifo);
}

static inline void dague_lifo_push( dague_lifo_t* lifo, 
                                    dague_list_item_t* item )                                   
{
#if defined(DAGUE_DEBUG)
    assert( (uintptr_t)item % DAGUE_LIFO_ALIGNMENT == 0 );
#endif
    DAGUE_ITEM_ATTACH(lifo, item);

    dague_list_item_t* tp = DAGUE_LIFO_VAL(item, (item->keeper_of_the_seven_keys + 1));
    
    do {
        item->list_next = lifo->lifo_head;
        if( dague_atomic_cas(&(lifo->lifo_head),
                             (uintptr_t) item->list_next,
                             (uintptr_t) tp) ) {
            return;
        }
        /* DO some kind of pause to release the bus */
    } while( 1 );
}
static inline void dague_lifo_nolock_push( dague_lifo_t* lifo, 
                                           dague_list_item_t* item )
{
 #if defined(DAGUE_DEBUG)
    assert( (uintptr_t)item % DAGUE_LIFO_ALIGNMENT == 0 );
#endif
    DAGUE_ITEM_ATTACH(lifo, item);
    
    item->list_next = lifo->lifo_head;
    lifo->lifo_head = item;
}

static inline void dague_lifo_chain( dague_lifo_t* lifo,
                                     dague_list_item_t* items )
{
#if defined(DAGUE_DEBUG)
    assert( (uintptr_t)items % DAGUE_LIFO_ALIGNMENT == 0 );
#endif
    DAGUE_ITEMS_ATTACH(lifo, items);

    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
    dague_list_item_t* tp = DAGUE_LIFO_VAL(items, (items->keeper_of_the_seven_keys + 1));
    
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
#if defined(DAGUE_DEBUG)
    assert( (uintptr_t)items % DAGUE_LIFO_ALIGNMENT == 0 );
#endif
    DAGUE_ITEMS_ATTACH(lifo, items);

    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
    
    tail->list_next = lifo->lifo_head;
    lifo->lifo_head = items;
}

static inline dague_list_item_t* dague_lifo_pop( dague_lifo_t* lifo )
{
    dague_list_item_t *item, *save;

    while(DAGUE_LIFO_PTR((item = lifo->lifo_head)) != lifo->lifo_ghost) {
        if( dague_atomic_cas(&(lifo->lifo_head),
                             (uintptr_t) item,
                             (uintptr_t) DAGUE_LIFO_PTR(item)->list_next ) )
            break;
        /* Do some kind of pause to release the bus */
    }
    save = item;
    item = DAGUE_LIFO_PTR(item);
    if( item == lifo->lifo_ghost ) return NULL;
    item->keeper_of_the_seven_keys = DAGUE_LIFO_CNT(save);
    DAGUE_ITEM_DETACH(item);
    return item;    
}

static inline dague_list_item_t* dague_lifo_try_pop( dague_lifo_t* lifo )
{
    dague_list_item_t *item, *save;

    item = lifo->lifo_head;
    if( DAGUE_LIFO_PTR(item) == lifo->lifo_ghost ) 
        return NULL;
        
    if( dague_atomic_cas(&(lifo->lifo_head),
                         (uintptr_t) item,
                         (uintptr_t) DAGUE_LIFO_PTR(item)->list_next) )
    {
        save = item;
        item = DAGUE_LIFO_PTR(item);
        item->keeper_of_the_seven_keys = DAGUE_LIFO_CNT(save);
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

static inline void dague_lifo_construct( dague_lifo_t* lifo )
{
    DAGUE_LIFO_ITEM_ALLOC(lifo->lifo_ghost, sizeof(dague_list_item_t));
    dague_list_item_construct(lifo->lifo_ghost);
    DAGUE_ITEM_ATTACH(lifo, lifo->lifo_ghost);
    lifo->lifo_head = lifo->lifo_ghost;
}

static inline void dague_lifo_destruct( dague_lifo_t *lifo )
{
    DAGUE_ITEM_DETACH(lifo->lifo_ghost);
    DAGUE_LIFO_ITEM_FREE(lifo->lifo_ghost);
}

#else /* LIFO_USE_ATOMICS */

#include "list.h"

struct dague_lifo_t {
    dague_list_t list;
};

#define DAGUE_LIFO_ITEM_ALLOC( elt, truesize ) ({                       \
    (elt) = (__typeof__(elt)) malloc(truesize);                         \
    assert( NULL != elt ); \
    DAGUE_LIST_ITEM_CONSTRUCT(elt);                                     \
    (elt); })
#define DAGUE_LIFO_ITEM_FREE( elt ) do {                                \
    DAGUE_LIST_ITEM_DESTRUCT(elt);                                      \
    free(elt); } while(0)

static inline void 
dague_lifo_construct( dague_lifo_t* lifo ) {
    dague_list_construct((dague_list_t*)lifo);
}

static inline void
dague_lifo_destruct( dague_lifo_t* lifo ) {
    dague_list_destruct((dague_list_t*)lifo);
}

static inline int
dague_lifo_is_empty( dague_lifo_t* lifo ) {
    return dague_list_is_empty((dague_list_t*)lifo);
}

static inline int 
dague_lifo_nolock_is_empty( dague_lifo_t* lifo)
{
    return dague_list_nolock_is_empty((dague_list_t*)lifo);
}

static inline void
dague_lifo_push(dague_lifo_t* lifo, dague_list_item_t* item) {
    dague_list_push_front((dague_list_t*)lifo, item); 
}
static inline void
dague_lifo_nolock_push(dague_lifo_t* lifo, dague_list_item_t* item) { 
    dague_list_nolock_push_front((dague_list_t*)lifo, item); 
}

static inline void
dague_lifo_chain(dague_lifo_t* lifo, dague_list_item_t* items) {
    dague_list_chain_front((dague_list_t*)lifo, items);
}
static inline void
dague_lifo_nolock_chain(dague_lifo_t* lifo, dague_list_item_t* items) { 
    dague_list_nolock_chain_front((dague_list_t*)lifo, items);
}

static inline dague_list_item_t*
dague_lifo_pop(dague_lifo_t* lifo) {
    return dague_list_pop_front((dague_list_t*)lifo); 
}
static inline dague_list_item_t*
dague_lifo_try_pop(dague_lifo_t* lifo) {
    return dague_list_try_pop_front((dague_list_t*)lifo);
}
static inline dague_list_item_t* 
dague_lifo_nolock_pop(dague_lifo_t* lifo) { 
    return dague_list_nolock_pop_front((dague_list_t*)lifo); 
}

#endif /* LIFO_USE_ATOMICS */

#endif  /* LIFO_H_HAS_BEEN_INCLUDED */
