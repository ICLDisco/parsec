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

static inline void dague_list_item_construct( dague_list_item_t *item )
{
    item->list_prev = item;
    item->list_next = item;
    item->keeper_of_the_seven_keys = 0;
#if defined(DAGUE_DEBUG)
    item->refcount = 0;
    item->belong_to_list = (void*)0xdeadbeef;
#endif
}

/* Make a well formed singleton list with a list item so that it can be 
 * chained.
 */
#define DAGUE_LIST_ITEM_SINGLETON(item) dague_list_item_singleton((dague_list_item_t*) item)
static inline dague_list_item_t* dague_list_item_singleton(dague_list_item_t* item)
{
    item->list_next = item;
    item->list_prev = item;
    return item;
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
        assert(_item->list_next != (void*)0xdeadbeef);                  \
        assert(_item->list_prev != (void*)0xdeadbeef);                  \
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
#define DAGUE_ITEMS_DETACH(ITEM) do { (void)(ITEM); } while(0)
#endif  /* DAGUE_DEBUG */

#endif

