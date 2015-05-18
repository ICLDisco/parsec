/*
 * Copyright (c) 2010-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_LIST_ITEM_H_HAS_BEEN_INCLUDED
#define DAGUE_LIST_ITEM_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "dague/class/dague_object.h"
#include <stdlib.h>
#include <assert.h>

typedef struct dague_list_item_s {
    dague_object_t  super;
    volatile struct dague_list_item_s* list_next;
    /* This field is __very__ special and should be handled with extreme
     * care. It is used to avoid the ABA problem when atomic operations
     * are in use. It can deal with 2^DAGUE_LIFO_ALIGNMENT_BITS pops,
     * before running into the ABA. In all other cases, it is used to
     * separate the two volatile members of the struct to avoid
     * cacheline false sharing
     */
    uint64_t aba_key;
    volatile struct dague_list_item_s* list_prev;
#if defined(DAGUE_DEBUG_ENABLE)
    volatile int32_t refcount;
    volatile void* belong_to;
#endif  /* defined(DAGUE_DEBUG_ENABLE) */
} dague_list_item_t;

DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_list_item_t);

#define DAGUE_LIST_ITEM_NEXT(item) ((__typeof__(item))(((dague_list_item_t*)(item))->list_next))
#define DAGUE_LIST_ITEM_PREV(item) ((__typeof__(item))(((dague_list_item_t*)(item))->list_prev))

/** Make a well formed singleton ring with a list item @item.
 *   @return @item, a valid list item ring containing itself
 */
static inline dague_list_item_t*
dague_list_item_singleton( dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert(0 == item->refcount);
    item->belong_to = item;
#endif
    item->list_next = item;
    item->list_prev = item;
    return item;
}
#define DAGUE_LIST_ITEM_SINGLETON(item) dague_list_item_singleton((dague_list_item_t*) item)

/** Make a ring from a chain of items, starting with @first, ending with @last, @returns @first
 *    if first->last is not a valid chain of items, result is undetermined
 *    in DAGUE_DEBUG_ENABLE mode, attached items are detached, must be reattached if needed */
static inline dague_list_item_t*
dague_list_item_ring( dague_list_item_t* first, dague_list_item_t* last )
{
    first->list_prev = last;
    last->list_next = first;

#if defined(DAGUE_DEBUG_ENABLE)
    if( 1 == first->refcount )
    {   /* Pseudo detach the items if they had been attached */
        dague_list_item_t* item = first;
        do {
            assert( item->belong_to == first->belong_to );
            item->refcount--;
            assert( 0 == item->refcount );
            item = (dague_list_item_t*)item->list_next;
        } while(item != first);
    }
#endif

    return first;
}

/* Add an @item to the item ring @ring, preceding @ring (not thread safe)
 *   @return @ring, the list item representing the ring
 */
static inline dague_list_item_t*
dague_list_item_ring_push( dague_list_item_t* ring,
                           dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert( 0 == item->refcount );
    assert( (void*)0xdeadbeef != ring->list_next );
    assert( (void*)0xdeadbeef != ring->list_prev );
#endif
    item->list_next = ring;
    item->list_prev = ring->list_prev;
    ring->list_prev->list_next = item;
    ring->list_prev = item;
    return ring;
}

/* Merge @ring1 with @ring2 (not thread safe)
 *   @return @ring1
 */
static inline dague_list_item_t*
dague_list_item_ring_merge( dague_list_item_t* ring1,
                            dague_list_item_t* ring2 )
{
    volatile dague_list_item_t *tmp;
#if defined(DAGUE_DEBUG_ENABLE)
    assert( (void*)0xdeadbeef != ring1->list_next );
    assert( (void*)0xdeadbeef != ring1->list_prev );
    assert( (void*)0xdeadbeef != ring2->list_next );
    assert( (void*)0xdeadbeef != ring2->list_prev );
#endif
    ring2->list_prev->list_next = ring1;
    ring1->list_prev->list_next = ring2;
    tmp = ring1->list_prev;
    ring1->list_prev = ring2->list_prev;
    ring2->list_prev = tmp;

    return ring1;
}

/* Remove the current first item of the ring @item (not thread safe)
 *   @returns the ring starting at next item, or NULL if @item is a singleton
 */
static inline dague_list_item_t*
dague_list_item_ring_chop( dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert( (void*)0xdeadbeef != item->list_next );
    assert( (void*)0xdeadbeef != item->list_prev );
#endif
    dague_list_item_t* ring = (dague_list_item_t*)item->list_next;
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
#if defined(DAGUE_DEBUG_ENABLE)
    if(item->refcount) item->refcount--;
    item->list_prev = (void*)0xdeadbeef;
    item->list_next = (void*)0xdeadbeef;
#endif
    if(ring == item) return NULL;
    return ring;
}

#define _LIST_ITEM_ITERATOR(RING, LAST, ITEM, CODE)                     \
    ({                                                                  \
        dague_list_item_t* ITEM = (dague_list_item_t*)(RING);           \
        do {                                                            \
            {                                                           \
                CODE;                                                   \
            }                                                           \
            ITEM = (dague_list_item_t*)ITEM->list_next;                 \
        } while (ITEM != (LAST));                                       \
        ITEM;                                                           \
    })

static inline dague_list_item_t*
dague_list_item_ring_push_sorted( dague_list_item_t* ring,
                                  dague_list_item_t* item,
                                  size_t off )
{
    dague_list_item_t* position;
    int success = 0;

    dague_list_item_singleton(item);
    if( NULL == ring ) {
        return item;
    }
    position = _LIST_ITEM_ITERATOR(ring, ring, pos,
                                   {
                                       if( !A_LOWER_PRIORITY_THAN_B(item, pos, off) ) {
                                           success = 1;
                                           break;
                                       }
                                   });
    dague_list_item_ring_push(position, item);
    if( success && (ring == position) ) return item;
    return ring;
}

/* This is debu helpers for list items accounting */
#if defined(DAGUE_DEBUG_ENABLE)
#define DAGUE_ITEMS_VALIDATE(ITEMS)                                     \
    do {                                                                \
        dague_list_item_t *__end = (ITEMS);                             \
        int _number; dague_list_item_t *__item;                         \
        for(_number=0, __item = (dague_list_item_t*)__end->list_next;   \
            __item != __end;                                            \
            __item = (dague_list_item_t*)__item->list_next ) {          \
            assert( (__item->refcount == 0) || (__item->refcount == 1) );\
            assert( __end->refcount == __item->refcount );              \
            if( __item->refcount == 1 )                                 \
                assert(__item->belong_to == __end->belong_to);          \
            if( ++_number > 1000 ) assert(0);                           \
        }                                                               \
    } while(0)

#define DAGUE_ITEM_ATTACH(LIST, ITEM)                                   \
    do {                                                                \
        dague_list_item_t *_item_ = (ITEM);                             \
        _item_->refcount++;                                             \
        assert( 1 == _item_->refcount );                                \
        _item_->belong_to = (LIST);                                     \
    } while(0)

#define DAGUE_ITEMS_ATTACH(LIST, ITEMS)                                 \
    do {                                                                \
        dague_list_item_t *_item = (ITEMS);                             \
        assert( (void*)0xdeadbeef != _item->list_next );                \
        assert( (void*)0xdeadbeef != _item->list_prev );                \
        dague_list_item_t *_end = (dague_list_item_t *)_item->list_prev; \
        do {                                                            \
            DAGUE_ITEM_ATTACH(LIST, _item);                             \
            _item = (dague_list_item_t*)_item->list_next;               \
        } while(_item != _end->list_next);                              \
    } while(0)

#define DAGUE_ITEM_DETACH(ITEM)                                         \
    do {                                                                \
        dague_list_item_t *_item = (ITEM);                              \
        /* check for not poping the ghost element, doesn't work for atomic_lifo */\
        assert( _item->belong_to != (void*)_item );                     \
        _item->list_prev = (void*)0xdeadbeef;                           \
        _item->list_next = (void*)0xdeadbeef;                           \
        _item->refcount--;                                              \
        assert( 0 == _item->refcount );                                 \
    } while (0)
#else
#define DAGUE_ITEMS_VALIDATE_ELEMS(ITEMS) do { (void)(ITEMS); } while(0)
#define DAGUE_ITEM_ATTACH(LIST, ITEM) do { (void)(LIST); (void)(ITEM); } while(0)
#define DAGUE_ITEMS_ATTACH(LIST, ITEMS) do { (void)(LIST); (void)(ITEMS); } while(0)
#define DAGUE_ITEM_DETACH(ITEM) do { (void)(ITEM); } while(0)
#endif  /* DAGUE_DEBUG_ENABLE */

#endif

