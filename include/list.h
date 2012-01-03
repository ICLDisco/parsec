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


/****************************************************************/
/* Done with list_item definitions. Proceed with list functions */

typedef struct dague_list_t {
    dague_list_item_t  ghost_element;
    uint32_t atomic_lock;
} dague_list_t;

static inline void 
dague_list_construct( dague_list_t* linked_list )
{
    linked_list->ghost_element.list_next = &(linked_list->ghost_element);
    linked_list->ghost_element.list_prev = &(linked_list->ghost_element);
    linked_list->atomic_lock = 0;
}

static inline int 
dague_list_is_empty( dague_list_t * linked_list )
{
    return linked_list->ghost_element.list_next != &(linked_list->ghost_element);
}


/* Add/remove items from a list. nolock version do not lock the list */
static inline void
dague_list_nolock_add_head( dague_list_t* linked_list, 
                            dague_list_item_t* item )
{
    item->list_prev = &(linked_list->ghost_element);
    item->list_next = linked_list->ghost_element.list_next;
    linked_list->ghost_element.list_next->list_prev = item;
    linked_list->ghost_element.list_next = item;
    DAGUE_ATTACH_ELEM(linked_list, item);                                
}

static inline void 
dague_list_add_head( dague_list_t * linked_list,
                     dague_list_item_t *item )
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    dague_list_nolock_add_head(linked_list, item);
    dague_atomic_unlock(&(linked_list->atomic_lock));
}


static inline void 
dague_list_nolock_add_tail( dague_list_t * linked_list,
                            dague_list_item_t *item )
{
    item->list_next = &(linked_list->ghost_element);
    item->list_prev = linked_list->ghost_element.list_prev;
    linked_list->ghost_element.list_prev->list_next = item;
    linked_list->ghost_element.list_prev = item;
    DAGUE_ATTACH_ELEM(linked_list, item);
}

static inline void 
dague_list_add_tail( dague_list_t * linked_list,
                     dague_list_item_t *item )
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    dague_list_nolock_add_tail(linked_list, item);
    dague_atomic_unlock(&(linked_list->atomic_lock));
}


static inline dague_list_item_t*
dague_list_nolock_remove_head( dague_list_t * linked_list )
{
    dague_list_item_t* item;

    item = (dague_list_item_t*)linked_list->ghost_element.list_next;
    linked_list->ghost_element.list_next = item->list_next;
    item->list_next->list_prev = &linked_list->ghost_element;
    item->list_prev = item;
    item->list_next = item;
    if( &(linked_list->ghost_element) != item ) {
        DAGUE_DETACH_ELEM(item);
        return item;
    }
    return NULL;
}

static inline dague_list_item_t*
dague_list_remove_head( dague_list_t * linked_list )
{
    dague_list_item_t* item;
    dague_atomic_lock(&(linked_list->atomic_lock));
    item = dague_list_nolock_remove_head(linked_list);
    dague_atomic_unlock(&(linked_list->atomic_lock));
    return item;
}


static inline dague_list_item_t*
dague_list_nolock_remove_tail( dague_list_t * linked_list )
{
    dague_list_item_t* item;

    item = (dague_list_item_t*)linked_list->ghost_element.list_prev;
    linked_list->ghost_element.list_prev = item->list_prev;
    item->list_prev->list_next = &linked_list->ghost_element;
    item->list_prev = item;
    item->list_next = item;
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
    item = dague_list_nolock_remove_tail(linked_list);
    dague_atomic_unlock(&(linked_list->atomic_lock));
    return item;
}


static inline dague_list_item_t*
dague_list_nolock_remove_item( dague_list_t * linked_list,
                               dague_list_item_t* item)
{
#if defined(DAGUE_DEBUG)
    assert(item->belong_to_list == linked_list);
#else
    (void)linked_list;
#endif
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
    item->list_next = item;
    item->list_prev = item;
    DAGUE_DETACH_ELEM(item);
    return item;
}

static inline dague_list_item_t*
dague_list_remove_item( dague_list_t * linked_list,
                        dague_list_item_t* item)
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    item = dague_list_nolock_remove_item(linked_list, item);
    dague_atomic_unlock(&(linked_list->atomic_lock));
    return item;
}

/* define some convenience function shortnames */
/* uop versions are not locked versions of op */
#define dague_list_upush(list, item) dague_list_nolock_add_head(list, item)
#define dague_list_push(list, item) dague_list_add_head(list, item)
#define dague_list_upop(list) dague_list_nolock_remove_head(list)
#define dague_list_pop(list) dague_list_remove_head(list)


/* Iterate functions permits traversing the list. The considered items 
 *   remain in the list. Hence, the iterate functions are not thread 
 *   safe. If the list or the list items are modified (even with 
 *   locked remove/add), status is undetermined.
 * Typical use: 
 *   for( item = iterate_head(list); 
 *        item != NULL; 
 *        item = iterate_next(list, item) ) { use_item(item); }
 *  "use_item()"" should not change item->list_next, item->list_prev
 */
static inline dague_list_item_t*
dague_list_iterate_head( dague_list_t* linked_list )
{
    if(linked_list->ghost_element.list_next == &(linked_list->ghost_element))
        return NULL;
    return (dague_list_item_t*) linked_list->ghost_element.list_next;
}

static inline dague_list_item_t*
dague_list_iterate_tail( dague_list_t* linked_list )
{
    if(linked_list->ghost_element.list_prev == &(linked_list->ghost_element))
        return NULL;
    return (dague_list_item_t*) linked_list->ghost_element.list_prev;
}

static inline dague_list_item_t*
dague_list_iterate_next( dague_list_t* linked_list, 
                         dague_list_item_t* item )
{
    if(item->list_next == &(linked_list->ghost_element)) return NULL;
    else return (dague_list_item_t*) item->list_next;
}

static inline dague_list_item_t*
dague_list_iterate_prev( dague_list_t* linked_list, 
                         dague_list_item_t* item )
{
    if(item->list_prev == &(linked_list->ghost_element)) return NULL;
    else return (dague_list_item_t*) item->list_prev;
}

/* Remove current item, and returns the next */
static inline dague_list_item_t*
dague_list_iterate_remove_and_next( dague_list_t* linked_list, 
                                    dague_list_item_t* item )
{
    dague_list_item_t* next = dague_list_iterate_next(linked_list, item);
    dague_list_nolock_remove_item(linked_list, item);
    return next;
}

static inline dague_list_item_t*
dague_list_iterate_remove_and_prev( dague_list_t* linked_list,
                                    dague_list_item_t* item )
{
    dague_list_item_t* prev = dague_list_iterate_prev(linked_list, item);
    dague_list_nolock_remove_item(linked_list, item);
    return prev;
}

#define dague_list_iterate_remove(L,I) dague_list_iterate_remove_and_next(L,I)

/* Lock the list while it is used, to prevent concurent add/remove */
static inline void 
dague_list_iterate_lock( dague_list_t* linked_list )
{
    dague_atomic_lock(&(linked_list->atomic_lock));
}

static inline void
dague_list_iterate_unlock( dague_list_t* linked_list )
{
    dague_atomic_unlock(&(linked_list->atomic_lock));
}



#endif  /* DAGUE_LINKED_LIST_H_HAS_BEEN_INCLUDED */
