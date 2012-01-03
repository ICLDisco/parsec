/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_LINKED_LIST_H_HAS_BEEN_INCLUDED
#define DAGUE_LINKED_LIST_H_HAS_BEEN_INCLUDED

#include "atomic.h"

typedef struct dague_list_item_t {
    volatile struct dague_list_item_t* list_next;
    /**
     * This field is __very__ special and should be handled with extreme
     * care. It is used to avoid the ABA problem when atomic operations
     * are in use. It can deal with 2^DAGUE_LIFO_ALIGNMENT_BITS pops,
     * before running into the ABA. In all other cases, it is used to
     * separate the two volatile members of the struct by a safe margin.
     */
    uint64_t keeper_of_the_seven_keys;
    volatile struct dague_list_item_t* list_prev;
#if defined(DAGUE_DEBUG)
    volatile int32_t refcount;
    volatile struct dague_list_t* belong_to_list;
#endif  /* defined(DAGUE_DEBUG) */
} dague_list_item_t;

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
        _item->belong_to_list = NULL;            \
    } while (0)
#else
#define DAGUE_VALIDATE_ELEMS(ITEMS)
#define DAGUE_ATTACH_ELEMS(LIST, ITEMS)         DAGUE_VALIDATE_ELEMS(ITEMS)
#define DAGUE_DETACH_ELEM(ITEM)
#endif  /* DAGUE_DEBUG */

/* Make a well formed singleton list with a list item so that it can be 
 * puhsed. 
 */
#define DAGUE_LIST_ITEM_SINGLETON(item) dague_list_item_singleton((dague_list_item_t*) item)
static inline dague_list_item_t* dague_list_item_singleton(dague_list_item_t* item)
{
    item->list_next = item;
    item->list_prev = item;
    return item;
}






typedef struct dague_list_t {
    dague_list_item_t  ghost_element;
    uint32_t atomic_lock;
} dague_list_t;

static inline void dague_list_construct( dague_list_t* linked_list )
{
    linked_list->ghost_element.list_next = &(linked_list->ghost_element);
    linked_list->ghost_element.list_prev = &(linked_list->ghost_element);
    linked_list->atomic_lock = 0;
}

static inline void dague_list_item_construct( dague_list_item_t *item )
{
    item->list_prev = item;
    item->list_next = item;
}

static inline int dague_list_is_empty( dague_list_t * linked_list )
{
    return linked_list->ghost_element.list_next != &(linked_list->ghost_element);
}

static inline void 
dague_list_add_head( dague_list_t * linked_list,
                            dague_list_item_t *item )
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    item->list_prev = &(linked_list->ghost_element);
    item->list_next = linked_list->ghost_element.list_next;
    linked_list->ghost_element.list_next->list_prev = item;
    linked_list->ghost_element.list_next = item;
    DAGUE_ATTACH_ELEMS(linked_list, item);
    dague_atomic_unlock(&(linked_list->atomic_lock));
}

static inline void 
dague_list_add_tail( dague_list_t * linked_list,
                            dague_list_item_t *item )
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    item->list_next = &(linked_list->ghost_element);
    item->list_prev = linked_list->ghost_element.list_prev;
    linked_list->ghost_element.list_prev->list_next = item;
    linked_list->ghost_element.list_prev = item;
    DAGUE_ATTACH_ELEMS(linked_list, item);
    dague_atomic_unlock(&(linked_list->atomic_lock));
}

static inline dague_list_item_t*
dague_list_remove_item( dague_list_t * linked_list,
                               dague_list_item_t* item)
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
    item->list_next = item;
    item->list_prev = item;
    dague_atomic_unlock(&(linked_list->atomic_lock));
    DAGUE_DETACH_ELEM(item);
    return item;
}

static inline dague_list_item_t*
dague_list_remove_head( dague_list_t * linked_list )
{
    dague_list_item_t* item;

    dague_atomic_lock(&(linked_list->atomic_lock));
    item = (dague_list_item_t*)linked_list->ghost_element.list_next;
    linked_list->ghost_element.list_next = item->list_next;
    item->list_next->list_prev = &linked_list->ghost_element;
    item->list_prev = item;
    item->list_next = item;
    dague_atomic_unlock(&(linked_list->atomic_lock));
    if( &(linked_list->ghost_element) != item ) {
        DAGUE_DETACH_ELEM(item);
        return item;
    }
    return NULL;
}

static inline dague_list_item_t*
dague_list_remove_tail( dague_list_t * linked_list )
{
    dague_list_item_t* item;

    dague_atomic_lock(&(linked_list->atomic_lock));
    item = (dague_list_item_t*)linked_list->ghost_element.list_prev;
    linked_list->ghost_element.list_prev = item->list_prev;
    item->list_prev->list_next = &linked_list->ghost_element;
    item->list_prev = item;
    item->list_next = item;
    dague_atomic_unlock(&(linked_list->atomic_lock));
    if( &(linked_list->ghost_element) != item ) {
        DAGUE_DETACH_ELEM(item);
        return item;
    }
    return NULL;
}

#endif  /* DAGUE_LINKED_LIST_H_HAS_BEEN_INCLUDED */
