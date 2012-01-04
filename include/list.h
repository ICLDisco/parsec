/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* This file contains functions to access simple lists and stacks, 
 * When thread safe locking performance is critical, one could prefer 
 * atomic lifo (see lifo.h)
 */

#ifndef DAGUE_LIST_H_HAS_BEEN_INCLUDED
#define DAGUE_LIST_H_HAS_BEEN_INCLUDED

#include "atomic.h"
#include "list_item.h"

typedef struct dague_list_t {
    dague_list_item_t  ghost_element;
    uint32_t atomic_lock;
} dague_list_t;


static inline void dague_list_construct( dague_list_t* list );
static inline void dague_list_destruct( dague_list_t* list );

/** check if list is empty (mutex protected) */
static inline int dague_list_is_empty( dague_list_t* list );
/** check if list is empty (not thread safe) */
static inline int dague_list_nolock_is_empty( dague_list_t* list );

/* define some convenience function shortnames 
 *   OPf does OP on the head of the list, OPb on the tail
 *   tOP tries to do OP, but returns immediatly if the list is locked
 *   ulist_OP versions are unsafe (not locked) versions of list_OP
 * if you need a FIFO or a LIFO, consider fifo.h and lifo.h
 */
#define dague_list_pushf(list, item) dague_list_push_front(list, item)
#define dague_list_pushb(list, item) dague_list_push_back(list, item)

#define dague_list_chainf(list, items) dague_list_chain_front(list, items)
#define dague_list_chainb(list, items) dague_list_chain_back(list, items)

#define dague_list_popf(list) dague_list_pop_front(list)
#define dague_list_popb(list) dague_list_pop_back(list)
#define dague_list_tpopf(list) dague_list_try_pop_front(list)
#define dague_list_tpopb(list) dague_list_try_pop_back(list)

#define dague_ulist_is_empty(list) dague_list_nolock_is_empty(list)

#define dague_ulist_pushf(list, item) dague_list_nolock_push_front(list, item)
#define dague_ulist_pushb(list, item) dague_list_nolock_push_back(list, item)

#define dague_ulist_chainf(list, items) dague_list_nolock_chain_front(list, items)
#define dague_ulist_chainb(list, items) dague_list_nolock_chain_back(list, items)

#define dague_ulist_popf(list) dague_list_nolock_pop_front(list)
#define dague_ulist_popb(list) dague_list_nolock_pop_back(list)

#define dague_ulist_remove_item(list, item) dague_list_nolock_remove_item(list, item)

/* Iterate functions permits traversing the list. The considered items
 *   remain in the list. Hence, the iterate functions are not thread 
 *   safe. If the list or the list items are modified (even with 
 *   locked remove/add), status is undetermined. One can lock the list
 *   with iterate_lock/unlock, but as soon as iterate_unlock has been 
 *   called, regardless of the current locked status, calls to
 *   iterate_next/prev may have unspecified results.
 * Typical use: 
 *   iterate_lock(list);
 *   for( item = iterate_head(list); 
 *        item != iterate_end(list); 
 *        item = iterate_next(list, item) ) { use_item(item); }
 *   iterate_unlock(list);
 */

/** obtain the first item of @list, or the Ghost if @list is empty, 
 *    the returned item remains in the list (not thread safe) */
static inline dague_list_item_t*
dague_list_iterate_first( dague_list_t* list );
/** obtain the last item of @list, or the Ghost if @list is empty,
 *    the returned item remains in the list (not thread safe) */
static inline dague_list_item_t*
dague_list_iterate_last( dague_list_t* list );
/** obtain the successor of @item, or the Ghost if @item is the tail of @list
 *    the returned item remains in the list (not thread safe) */
static inline dague_list_item_t*
dague_list_iterate_next( dague_list_t* list, 
                         dague_list_item_t* item );
/** obtain the predecessor of @item, or the Ghost if @item is the head of @list
 *    the returned item remains in the list (not thread safe) */
static inline dague_list_item_t*
dague_list_iterate_prev( dague_list_t* list, 
                         dague_list_item_t* item );
/** add the @new item before the @position item in @list (not thread safe)
 *    @position item is obtained by one of the iterate functions, if 
 *    @position is the Ghost, @item is added back
 *    both @position and @new may be used in further iterate functions */
static inline void
dague_list_iterate_add_before( dague_list_t* list,
                               dague_list_item_t* position,
                               dague_list_item_t* new );
/** add the @new item before the @position item in @list (not thread safe)
 *    @position item is obtained by one of the iterate functions, if 
 *    @position is the Ghost, @item is added front
 *    both @position and @new may be used in further iterate functions */
static inline void
dague_list_iterate_add_after( dague_list_t* list,
                              dague_list_item_t* position,
                              dague_list_item_t* item );
/** remove current item, and returns the next (not thread safe) */
static inline dague_list_item_t*
dague_list_iterate_remove_and_next( dague_list_t* list, 
                                    dague_list_item_t* item ); 
/** remove current item, and returns the prev (not thread safe) */
static inline dague_list_item_t*
dague_list_iterate_remove_and_prev( dague_list_t* list,
                                    dague_list_item_t* item );
/* synonym to remove_and_next */
#define dague_list_iterate_remove(L,I) dague_list_iterate_remove_and_next(L,I)
/** lock the list */
static inline void 
dague_list_iterate_lock( dague_list_t* list );
/** unlock the list, any further calls to iterate_next/iterate_prev may
 *    have unspecified results, even if the list is locked again later */
static inline void
dague_list_iterate_unlock( dague_list_t* list );



/** push item first in the list (mutex protected) */ 
static inline void 
dague_list_push_front( dague_list_t* list, 
                       dague_list_item_t* item );
/** push item last in the list (mutex protected) */
static inline void 
dague_list_push_back( dague_list_t* list, 
                      dague_list_item_t* item );

/** chains the collection of items first in the list (mutex protected)
 *    items->prev must point to the tail of the items collection */
static inline void 
dague_list_chain_front( dague_list_t* list, 
                        dague_list_item_t* items );
/** chains the collection of items last in the list (mutex protected)
 *    items->prev must point to the tail of the items collection */
static inline void 
dague_list_chain_back( dague_list_t* list, 
                       dague_list_item_t* items );
                       
/** pop the first item of the list (mutex protected)
 *    if the list is empty, NULL is returned */
static inline dague_list_item_t*  
dague_list_pop_front( dague_list_t* list );
/** pop the last item of the list (mutex protected)
 *    if the list is empty, NULL is returned */
static inline dague_list_item_t*
dague_list_pop_back( dague_list_t* list );
/** try to pop the first item of the list (mutex protected)
 *    if the list is empty or currently locked, NULL is returned */
static inline dague_list_item_t*
dague_list_try_pop_front( dague_list_t* list );
/** try to pop the last item of the list (mutex protected)
 *    if the list is empty or currently locked, NULL is returned */
static inline dague_list_item_t*
dague_list_try_pop_back( dague_list_t* list );

/** remove a specific item from the list (not thread safe)
 *    item must be in the list, compared pointerwise */
static inline dague_list_item_t*
dague_list_remove_item( dague_list_t* list,
                        dague_list_item_t* item);


/* SAME AS ABOVE, FOR SINGLE THREAD USE */

/** push item first in the list (not thread safe) */ 
static inline void
dague_list_nolock_push_front( dague_list_t* list, 
                              dague_list_item_t* item );
/** push item last in the list (not thread safe) */ 
static inline void
dague_list_nolock_push_back( dague_list_t* list, 
                             dague_list_item_t* item );

/** chains the collection of items first in the list (not thread safe)
 *    items->prev must point to the tail of the items collection */
static inline void 
dague_list_nolock_chain_front( dague_list_t* list, 
                               dague_list_item_t* items );
/** chains the collection of items last in the list (not thread safe)
 *    items->prev must point to the tail of the items collection */
static inline void 
dague_list_nolock_chain_back( dague_list_t* list, 
                              dague_list_item_t* items );
                       
/** pop the first item of the list (not thread safe)
 *    if the list is empty, NULL is returned */
static inline dague_list_item_t*  
dague_list_nolock_pop_front( dague_list_t* list );
/** pop the last item of the list (not thread safe)
 *    if the list is empty, NULL is returned */
static inline dague_list_item_t*
dague_list_nolock_pop_back( dague_list_t* list );

/** remove a specific item from the list (not thread safe)
 *    item must be in the list, compared pointerwise */
static inline dague_list_item_t*
dague_list_nolock_remove_item( dague_list_t* list,
                               dague_list_item_t* item);



/***********************************************************************/
/* Interface ends here, everything else is private                     */ 

#define _HEAD(LIST) ((LIST)->ghost_element.list_next)
#define _TAIL(LIST) ((LIST)->ghost_element.list_prev)
#define _GHOST(LIST) (&((list)->ghost_element))

static inline void 
dague_list_construct( dague_list_t* list )
{
    _HEAD(list) = _GHOST(list);
    _TAIL(list) = _GHOST(list);
    list->atomic_lock = 0;
}

static inline void
dague_list_destruct( dague_list_t* list )
{
    assert(dague_list_is_empty(list));
}


static inline int 
dague_list_nolock_is_empty( dague_list_t* list )
{
    assert( ((_HEAD(list) != _GHOST(list)) && (_TAIL(list) != _GHOST(list))) || 
            ((_HEAD(list) == _GHOST(list)) && (_TAIL(list) == _GHOST(list))) );
    return _HEAD(list) == _GHOST(list);
}

static inline int
dague_list_is_empty( dague_list_t* list )
{
    int rc;
    dague_atomic_lock(&list->atomic_lock);
    rc = dague_list_nolock_is_empty(list);
    dague_atomic_unlock(&list->atomic_lock);
    return rc;
}


static inline void
dague_list_nolock_push_front( dague_list_t* list, 
                              dague_list_item_t* item )
{
    DAGUE_ATTACH_ELEM(list, item);
    item->list_prev = _GHOST(list);
    item->list_next = _HEAD(list);
    _HEAD(list)->list_prev = item;
    _HEAD(list) = item;
}

static inline void 
dague_list_push_front( dague_list_t* list,
                       dague_list_item_t *item )
{
    DAGUE_ATTACH_ELEM(list, item);
    item->list_prev = _GHOST(list);
    dague_atomic_lock(&list->atomic_lock);
    item->list_next = _HEAD(list);
    _HEAD(list)->list_prev = item;
    _HEAD(list) = item;
    dague_atomic_unlock(&list->atomic_lock);
}

static inline void 
dague_list_nolock_chain_front( dague_list_t* list,
                               dague_list_item_t* items )
{
    DAGUE_ATTACH_ELEMS(list, items);
    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
    items->list_prev = _GHOST(list);
    tail->list_next = _HEAD(list);
    _HEAD(list)->list_prev = tail;
    _HEAD(list) = items;
}

static inline void 
dague_list_chain_front( dague_list_t* list,
                        dague_list_item_t* items )
{
    DAGUE_ATTACH_ELEMS(list, items);
    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
    items->list_prev = _GHOST(list);
    dague_atomic_lock(&list->atomic_lock);
    tail->list_next = _HEAD(list);
    _HEAD(list)->list_prev = tail;
    _HEAD(list) = items;
    dague_atomic_unlock(&list->atomic_lock);
}


static inline void 
dague_list_nolock_push_back( dague_list_t* list,
                             dague_list_item_t *item )
{
    DAGUE_ATTACH_ELEM(list, item);
    item->list_next = _GHOST(list);
    item->list_prev = _TAIL(list);
    _TAIL(list)->list_next = item;
    _TAIL(list) = item;
}

static inline void 
dague_list_push_back( dague_list_t* list,
                      dague_list_item_t *item )
{
    DAGUE_ATTACH_ELEM(list, item);
    item->list_next = _GHOST(list);
    dague_atomic_lock(&list->atomic_lock);
    item->list_prev = _TAIL(list);
    _TAIL(list)->list_next = item;
    _TAIL(list) = item;
    dague_atomic_unlock(&list->atomic_lock);
}

static inline void 
dague_list_nolock_chain_back( dague_list_t* list,
                              dague_list_item_t* items )
{
    DAGUE_ATTACH_ELEMS(list, items);
    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
    tail->list_next = _GHOST(list);
    items->list_prev = _TAIL(list);
    _TAIL(list)->list_next = items;
    _TAIL(list) = tail;
}

static inline void 
dague_list_chain_back( dague_list_t* list,
                       dague_list_item_t* items )
{
    DAGUE_ATTACH_ELEMS(list, items);
    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
    tail->list_next = _GHOST(list);
    dague_atomic_lock(&list->atomic_lock);
    items->list_prev = _TAIL(list);
    _TAIL(list)->list_next = items;
    _TAIL(list) = tail;
    dague_atomic_unlock(&list->atomic_lock);
}


#define _RET_NULL_GHOST(LIST, ITEM) do {                                \
    if( _GHOST(LIST) != (ITEM) ) {                                      \
        DAGUE_DETACH_ELEM(ITEM);                                        \
        return (ITEM);                                                  \
    }                                                                   \
    return NULL;                                                        \
} while(0)

static inline dague_list_item_t*
dague_list_nolock_pop_front( dague_list_t* list )
{
    dague_list_item_t* item = (dague_list_item_t*)_HEAD(list);
    _HEAD(list) = item->list_next;
    _HEAD(list)->list_prev = &list->ghost_element;
    _RET_NULL_GHOST(list, item);
}

static inline dague_list_item_t*
dague_list_pop_front( dague_list_t* list )
{
    dague_atomic_lock(&list->atomic_lock);
    dague_list_item_t* item = (dague_list_item_t*)_HEAD(list);
    _HEAD(list) = item->list_next;
    _HEAD(list)->list_prev = _GHOST(list);
    dague_atomic_unlock(&list->atomic_lock);
    _RET_NULL_GHOST(list, item);
}

static inline dague_list_item_t*
dague_list_try_pop_front( dague_list_t* list)
{
    if( !dague_atomic_trylock(&list->atomic_lock) ) {
        return NULL;
    }
    dague_list_item_t* item = (dague_list_item_t*)_HEAD(list);
    _HEAD(list) = item->list_next;
    _HEAD(list)->list_prev = _GHOST(list);
    dague_atomic_unlock(&list->atomic_lock);
    _RET_NULL_GHOST(list, item);
}


static inline dague_list_item_t*
dague_list_nolock_pop_back( dague_list_t* list )
{
    dague_list_item_t* item = (dague_list_item_t*)_TAIL(list);
    _TAIL(list) = item->list_prev;
    _TAIL(list)->list_next = _GHOST(list);
    _RET_NULL_GHOST(list, item);
}

static inline dague_list_item_t*
dague_list_pop_back( dague_list_t* list )
{
    dague_atomic_lock(&list->atomic_lock);
    dague_list_item_t* item = (dague_list_item_t*)_TAIL(list);
    _TAIL(list) = item->list_prev;
    _TAIL(list)->list_next = _GHOST(list);
    dague_atomic_unlock(&list->atomic_lock);
    _RET_NULL_GHOST(list, item);
}

static inline dague_list_item_t*
dague_list_try_pop_back( dague_list_t* list)
{
    if( !dague_atomic_trylock(&list->atomic_lock) ) {
        return NULL;
    }
    dague_list_item_t* item = (dague_list_item_t*)_TAIL(list);
    _TAIL(list) = item->list_prev;
    _TAIL(list)->list_next = _GHOST(list);
    dague_atomic_unlock(&list->atomic_lock);
    _RET_NULL_GHOST(list, item);
}

#undef _RET_NULL_GHOST


static inline dague_list_item_t*
dague_list_nolock_remove_item( dague_list_t* list,
                               dague_list_item_t* item)
{
#if defined(DAGUE_DEBUG)
    assert(item->belong_to_list == list);
#else
    (void)list;
#endif
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
    item->list_next = item;
    item->list_prev = item;
    DAGUE_DETACH_ELEM(item);
    return item;
}

static inline dague_list_item_t*
dague_list_remove_item( dague_list_t* list,
                        dague_list_item_t* item)
{
#if defined(DAGUE_DEBUG)
    assert( item->belong_to_list == list );
#else
    (void)list;
#endif
    dague_atomic_lock(&list->atomic_lock);
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
    dague_atomic_unlock(&list->atomic_lock);
    item->list_next = item;
    item->list_prev = item;
    DAGUE_DETACH_ELEM(item);
    return item;
}



static inline dague_list_item_t*
dague_list_iterate_first( dague_list_t* list )
{
    return (dague_list_item_t*) _HEAD(list);
}

static inline dague_list_item_t*
dague_list_iterate_last( dague_list_t* list )
{
    return (dague_list_item_t*) _TAIL(list);
}

static inline dague_list_item_t*
dague_list_iterate_end( dague_list_t* list )
{
    return _GHOST(list);
}

static inline dague_list_item_t*
dague_list_iterate_next( dague_list_t* list, 
                         dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG)
    assert( item->belong_to_list == list );
#else
    (void)list;
#endif
    return (dague_list_item_t*)item->list_next;
}

static inline dague_list_item_t*
dague_list_iterate_prev( dague_list_t* list, 
                         dague_list_item_t* item )
{
#if defined(DAGUE_DEBUG)
    assert( item->belong_to_list == list );
#else
    (void)list;
#endif
    return (dague_list_item_t*)item->list_prev;
}

static inline void
dague_list_iterate_add_before( dague_list_t* list,
                               dague_list_item_t* position,
                               dague_list_item_t* new )
{
#if defined(DAGUE_DEBUG)
    assert( position->belong_to_list == list );
#endif
    DAGUE_ATTACH_ELEM(list, new);
    new->list_prev = position->list_prev;
    new->list_next = position;
    position->list_prev->list_next = new;
    position->list_prev = new;
}

static inline void
dague_list_iterate_add_after( dague_list_t* list,
                              dague_list_item_t* position,
                              dague_list_item_t* new )
{
#if defined(DAGUE_DEBUG)
    assert( position->belong_to_list == list );
#endif
    DAGUE_ATTACH_ELEM(list, new);
    new->list_prev = position;
    new->list_next = position->list_next;
    position->list_next->list_prev = new;
    position->list_next = new;
}

static inline dague_list_item_t*
dague_list_iterate_remove_and_next( dague_list_t* list, 
                                    dague_list_item_t* item )
{
    dague_list_item_t* next = dague_list_iterate_next(list, item);
    dague_list_nolock_remove_item(list, item);
    return next;
}

static inline dague_list_item_t*
dague_list_iterate_remove_and_prev( dague_list_t* list,
                                    dague_list_item_t* item )
{
    dague_list_item_t* prev = dague_list_iterate_prev(list, item);
    dague_list_nolock_remove_item(list, item);
    return prev;
}

#define dague_list_iterate_remove(L,I) dague_list_iterate_remove_and_next(L,I)

static inline void 
dague_list_iterate_lock( dague_list_t* list )
{
    dague_atomic_lock(&list->atomic_lock);
}

static inline void
dague_list_iterate_unlock( dague_list_t* list )
{
    dague_atomic_unlock(&list->atomic_lock);
}

#undef _GHOST
#undef _HEAD
#undef _TAIL

#endif  /* DAGUE_LIST_H_HAS_BEEN_INCLUDED */
