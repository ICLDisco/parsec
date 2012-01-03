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
    item->belong_to_list = 0xdeadbeef;
#endif
}

/* Make a well formed singleton list with a list item so that it can be 
 * pushed. 
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
#define DAGUE_VALIDATE_ELEMS(ITEMS)                                     \
    do {                                                                \
        dague_list_item_t *__end = (ITEMS);                             \
        dague_list_item_t *__item = (dague_list_item_t*)__end->list_next; \
        int _number = 0;                                                \
        for(; __item != __end;                                          \
            __item = (dague_list_item_t*)__item->list_next ) {          \
            if( ++_number > 1000 ) assert(0);                           \
        }                                                               \
    } while(0)

#define DAGUE_ATTACH_ELEM(LIST, ITEM)                                   \
    do {                                                                \
        dague_list_item_t *_item_ = (ITEM);                             \
        _item_->refcount++;                                             \
        _item_->belong_to_list = (struct dague_list_t*)(LIST);          \
    } while(0)

#define DAGUE_ATTACH_ELEMS(LIST, ITEMS)                                 \
    do {                                                                \
        dague_list_item_t *_item = (ITEMS);                             \
        dague_list_item_t *_end = (dague_list_item_t *)_item->list_prev; \
        do {                                                            \
            DAGUE_ATTACH_ELEM(LIST, _item);                             \
            _item = (dague_list_item_t*)_item->list_next;               \
        } while (_item != _end);                                        \
        DAGUE_VALIDATE_ELEMS(_item);                                    \
    } while(0)

#define DAGUE_DETACH_ELEM(ITEM)                  \
    do {                                         \
        dague_list_item_t *_item = (ITEM);       \
        _item->refcount--;                       \
        _item->belong_to_list = 0xdeadbeef;      \
    } while (0)
#else
#define DAGUE_VALIDATE_ELEMS(ITEMS)
#define DAGUE_ATTACH_ELEM(LIST, ITEM)
#define DAGUE_ATTACH_ELEMS(LIST, ITEMS)         DAGUE_VALIDATE_ELEMS(ITEMS)
#define DAGUE_DETACH_ELEM(ITEM)
#endif  /* DAGUE_DEBUG */

#endif

