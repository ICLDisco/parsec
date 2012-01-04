/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_LIST_ITEM_H_HAS_BEEN_INCLUDED
#define DAGUE_LIST_ITEM_H_HAS_BEEN_INCLUDED

#include "atomic.h"

typedef struct dague_list_item_t {
    volatile struct dague_list_item_t* list_next;
    /**
     * This field is __very__ special and should be handled with extreme
     * care. It is used to avoid the ABA problem when atomic operations
     * are in use. It can deal with 2^DAGUE_LIFO_ALIGNMENT_BITS pops,
     * before running into the ABA. In all other cases, it is used to
     * separate the two volatile members of the struct to avoid
     * cacheline false sharing
     */
    uint64_t keeper_of_the_seven_keys;
    volatile struct dague_list_item_t* list_prev;
#if defined(DAGUE_DEBUG)
    volatile int32_t refcount;
    volatile struct dague_list_t* belong_to_list;
#endif  /* defined(DAGUE_DEBUG) */
} dague_list_item_t;

static inline void 
dague_list_item_construct( dague_list_item_t* item )
{
    item->list_prev = item;
    item->list_next = item;
    item->keeper_of_the_seven_keys = 0;
#if defined(DAGUE_DEBUG)
    item->refcount = 0;
    item->belong_to_list = (void*)0xdeadbeef;
#endif
}

/* Make a well formed singleton ring with a list item @item.
 *   @return @item, a valid list item ring containing itself
 */
static inline dague_list_item_t* 
dague_list_item_singleton( dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG)
    assert(0 == item->refcount);
#endif
    item->list_next = item;
    item->list_prev = item;
    return item;
}
#define DAGUE_LIST_ITEM_SINGLETON(item) dague_list_item_singleton((dague_list_item_t*) item)

/* Add an @item to the item ring @ring, preceding @ring (not thread safe)
 *   @return @ring, the list item representing the ring
 */
static inline dague_list_item_t* 
dague_list_item_ring_push_back( dague_list_item_t* ring,
                                dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG)
    assert( 0 == ring->refcount );
    assert( (void*)0xdeadbeef != ring->list_next );
    assert( (void*)0xdeadbeef != ring->list_prev );
    assert( 0 == item->refcount );
#endif
    item->list_next = ring;
    item->list_prev = ring->list_prev;
    ring->list_prev->list_next = item;
    ring->list_prev = item;
    return ring;
}
#define dague_list_item_ring_pushb(ring, item) dague_list_item_ring_push_back(ring, item)
#define dague_list_item_ring_push(ring, item) dague_list_item_ring_push_back(ring, item)

/* Add an @item to the item ring @ring, preceding @ring (not thread safe)
 *   @return @ring, the list item representing the ring
 */
static inline dague_list_item_t* 
dague_list_item_ring_push_front( dague_list_item_t* ring,
                                 dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG)
    assert( 0 == ring->refcount );
    assert( (void*)0xdeadbeef != ring->list_next );
    assert( (void*)0xdeadbeef != ring->list_prev );
#endif
    dague_list_item_ring_push_back(ring, item);
    return item;
}
#define dague_list_item_ring_pushf(ring, item) dague_list_item_ring_pushf(ring, item)

/* Remove the current first item of the ring @item (not thread safe)
 *   @returns the ring starting at next item, or NULL if @item is a singleton
 */
static inline dague_list_item_t* 
dague_list_item_ring_remove_item( dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG)
    assert( 0 == item->refcount );
    assert( (void*)0xdeadbeef != item->list_next );
    assert( (void*)0xdeadbeef != item->list_prev );
#endif
    dague_list_item_t* ring = (dague_list_item_t*)item->list_next;
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
#if defined(DAGUE_DEBUG)
    item->list_prev = (void*)0xdeadbeef;
    item->list_next = (void*)0xdeadbeef;
#endif
    if(ring == item) return NULL;
    return ring;
}


/* This is debug helpers for list items accounting */
#if defined(DAGUE_DEBUG)
#define DAGUE_ITEMS_VALIDATE(ITEMS)                                     \
    do {                                                                \
        dague_list_item_t *__end = (ITEMS);                             \
        int _number; dague_list_item_t *__item;                          \
        for(_number=0, __item = (dague_list_item_t*)__end->list_next;    \
            __item != __end;                                            \
            __item = (dague_list_item_t*)__item->list_next ) {          \
            assert((__item->refcount == 0) || (__item->refcount == 1)); \
            assert(__end->refcount == __item->refcount);                \
            if( __item->refcount == 1 )                                 \
                assert(__item->belong_to_list == __end->belong_to_list);\
            if( ++_number > 1000 ) assert(0);                           \
        }                                                               \
    } while(0)

#define DAGUE_ITEM_ATTACH(LIST, ITEM)                                   \
    do {                                                                \
        dague_list_item_t *_item_ = (ITEM);                             \
        _item_->refcount++;                                             \
        assert(_item_->refcount == 1);                                  \
        _item_->belong_to_list = (struct dague_list_t*)(LIST);          \
    } while(0)

#define DAGUE_ITEMS_ATTACH(LIST, ITEMS)                                 \
    do {                                                                \
        dague_list_item_t *_item = (ITEMS);                             \
        assert( (void*)0xdeadbeef != _item->list_next );                \
        assert( (void*)0xdeadbeef != _item->list_prev );                \
        DAGUE_ITEMS_VALIDATE(_item);                                    \
        dague_list_item_t *_end = (dague_list_item_t *)_item->list_prev; \
        do {                                                            \
            DAGUE_ITEM_ATTACH(LIST, _item);                             \
            _item = (dague_list_item_t*)_item->list_next;               \
        } while (_item != _end);                                        \
    } while(0)

#define DAGUE_ITEM_DETACH(ITEM)            \
    do {                                         \
        dague_list_item_t *_item = (ITEM);       \
        /* check for not poping the ghost element, doesn't work for atomic_lifo */\
        assert( _item->belong_to_list != (void*)_item ); \
        _item->list_prev = (void*)0xdeadbeef;           \
        _item->list_next = (void*)0xdeadbeef;           \
        _item->refcount--;                       \
    } while (0)
#else
#define DAGUE_ITEMS_VALIDATE_ELEMS(ITEMS) do { (void)(ITEMS); } while(0)
#define DAGUE_ITEM_ATTACH(LIST, ITEM) do { (void)(LIST); (void)(ITEM); } while(0)
#define DAGUE_ITEMS_ATTACH(LIST, ITEMS) do { (void)(LIST); (void)(ITEMS); } while(0)
#define DAGUE_ITEM_DETACH(ITEM) do { (void)(ITEM); } while(0)
#endif  /* DAGUE_DEBUG */

#endif

