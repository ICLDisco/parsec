/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* This file contains functions to access doubly linked lists.
 * dague_ulist/dague_list_nolock functions are not thread safe, and
 *   can be used only when the list is locked (by list_lock) or when
 *   thread safety is ensured by another mean.
 * When locking performance is critical, one could prefer atomic lifo (see lifo.h)
 */

#ifndef DAGUE_LIST_H_HAS_BEEN_INCLUDED
#define DAGUE_LIST_H_HAS_BEEN_INCLUDED

#include <dague/sys/atomic.h>
#include "list_item.h"

typedef struct dague_list_t {
    dague_object_t     super;
    dague_list_item_t  ghost_element;
    volatile uint32_t  atomic_lock;
} dague_list_t;

DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_list_t);

/** lock the @list mutex, that same mutex is used by all
 *    mutex protected list operations */
static inline void
dague_list_lock( dague_list_t* list );
/** unlock the @list mutex, that same mutex is used by all
 *    mutex protected list operations */
static inline void
dague_list_unlock( dague_list_t* list );

/** check if @list is empty (mutex protected) */
static inline int dague_list_is_empty( dague_list_t* list );
/** check if list is empty (not thread safe) */
static inline int dague_list_nolock_is_empty( dague_list_t* list );
#define dague_ulist_is_empty(list) dague_list_nolock_is_empty(list)

/** Paste code to iterate on all items in the @LIST (front to back) (mutex protected)
 *    the @CODE_BLOCK code is applied to each item, which can be refered
 *    to as @ITEM_NAME in @CODE_BLOCK
 *    the entire loop iteration takes the list mutex, hence
 *      @CODE_BLOCK must not jump outside the block; although, break
 *      and continue are legitimate in @CODE_BLOCK
 *  @return the last considered item */
#define DAGUE_LIST_ITERATOR(LIST, ITEM_NAME, CODE_BLOCK) _OPAQUE_LIST_ITERATOR_DEFINITION(LIST,ITEM_NAME,CODE_BLOCK)
/** Paste code to iterate on all items in the @LIST (front to back) (not thread safe)
 *    the @CODE_BLOCK code is applied to each item, which can be refered
 *    to as @ITEM_NAME in @CODE_BLOCK
 *  @return the last considered item */
#define DAGUE_LIST_NOLOCK_ITERATOR(LIST, ITEM_NAME, CODE_BLOCK) _OPAQUE_ULIST_ITERATOR_DEFINITION(LIST,ITEM_NAME,CODE_BLOCK)
#define DAGUE_ULIST_ITERATOR(LIST, ITEM_NAME, CODE_BLOCK) _OPAQUE_ULIST_ITERATOR_DEFINITION(LIST,ITEM_NAME,CODE_BLOCK)

/** Alternatively: start from FIRST, until END, using NEXT to
 *  get the next element.
 *  Does not lock the list; does not work on ulist.
 */
#define DAGUE_LIST_ITERATOR_FIRST(LIST)    _OPAQUE_LIST_ITERATOR_FIRST_DEFINITION(LIST)
#define DAGUE_LIST_ITERATOR_END(LIST)      _OPAQUE_LIST_ITERATOR_END_DEFINITION(LIST)
#define DAGUE_LIST_ITERATOR_NEXT(ITEM)     _OPAQUE_LIST_ITERATOR_NEXT_DEFINITION(ITEM)

#define DAGUE_LIST_ITERATOR_LAST(LIST)     _OPAQUE_LIST_ITERATOR_LAST_DEFINITION(LIST)
#define DAGUE_LIST_ITERATOR_BEGIN(LIST)    _OPAQUE_LIST_ITERATOR_BEGIN_DEFINITION(LIST)
#define DAGUE_LIST_ITERATOR_PREV(ITEM)     _OPAQUE_LIST_ITERATOR_PREV_DEFINITION(ITEM)

/** add the @new item before the @position item in @list (not thread safe)
 *    @position item must be in @list
 *    if @position is the Ghost, @item is added back */
static inline void
dague_list_nolock_add_before( dague_list_t* list,
                       dague_list_item_t* position,
                       dague_list_item_t* new );
#define dague_ulist_add_before(list, pos, new) dague_list_nolock_add_before(list, pos, new)
/** convenience function, synonym to dague_list_nolock_add_before() */
#define dague_list_nolock_add(list, pos, new) dague_list_nolock_add_before(list, pos, new)
#define dague_ulist_add(list, pos, new) dague_list_nolock_add_before(list, pos, new)
/** add the @new item after the @position item in @list (not thread safe)
 *    @position item must be in @list
 *    if @position is the Ghost, @item is added front */
static inline void
dague_list_nolock_add_after( dague_list_t* list,
                      dague_list_item_t* position,
                      dague_list_item_t* item );
#define dague_ulist_add_after(list, pos, new) dague_list_nolock_add_after(list, pos, new)
/** remove a specific @item from the @list (not thread safe)
 *    @item must be in the @list
 *    @return predecessor of @item in @list */
static inline dague_list_item_t*
dague_list_nolock_remove( dague_list_t* list,
                          dague_list_item_t* item);
#define dague_ulist_remove(list, item) dague_list_nolock_remove(list, item)


/* SORTED LIST FUNCTIONS */

/** add the @item before the first element of @list that is strictly smaller" (mutex protected),
 *  according to the integer  value at @offset in items. That is, if the input @list is
 *  sorted (descending order), the resulting list is still sorted. */
static inline void
dague_list_push_sorted( dague_list_t* list,
                        dague_list_item_t* item,
                        size_t offset );
/** add the @item before the first element of @list that is striclty smaller (not thread safe),
 *  according to the integer  value at @offset in items. That is, if the input @list is
 *  sorted (descending order), the resulting list is still sorted. */
static inline void
dague_list_nolock_push_sorted( dague_list_t* list,
                               dague_list_item_t* item,
                               size_t offset );
#define dague_ulist_push_sorted(list, item, comparator) dague_list_nolock_push_sorted(list,item,comparator)


/** chain the unsorted @items (mutex protected), as if they had been
 *  inserted in a loop of dague_list_push_sorted(). That is, if the input
 * @list is sorted (descending order), the resulting list is still sorted. */
static inline void
dague_list_nolock_chain_sorted( dague_list_t* list,
                                dague_list_item_t* items,
                                size_t offset );
/** chain the unsorted @items (not thread safe), as if they had been
 *  inserted in a loop by dague_list_push_sorted(). That is, if the input
 * @list is sorted (descending order), the resulting list is still sorted. */
static inline void
dague_list_nolock_chain_sorted( dague_list_t* list,
                                dague_list_item_t* items,
                                size_t offset );
#define dague_ulist_chain_sorted(list, items, comp) dague_list_nolock_chain_sorted(list, items, comp)


/** sort @list according to the (descending) order defined by the integer
 * value at @offset in evey item (mutex protected) */
static inline void
dague_list_sort( dague_list_t* list,
                 size_t offset );
/** sort @list according to the (descending) order defined by the integer
 * value at @offset in evey item (not thread safe) */
static inline void
dague_list_nolock_sort( dague_list_t* list,
                        size_t offset );
#define dague_ulist_sort(list, comp) dague_list_nolock_sort(list,comp)

/* DEQUEUE EMULATION FUNCTIONS */

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

/** push item first in the list (mutex protected) */
static inline void
dague_list_push_front( dague_list_t* list,
                       dague_list_item_t* item );
#define dague_list_prepend dague_list_push_front

/** push item last in the list (mutex protected) */
static inline void
dague_list_push_back( dague_list_t* list,
                      dague_list_item_t* item );
#define dague_list_append dague_list_push_back

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

/** unchain the entire collection of items from the list (mutex protected)
 *    the return is a list_item ring */
static inline dague_list_item_t*
dague_list_unchain( dague_list_t* list );

/** pop the first item of the list (not thread safe)
 *    if the list is empty, NULL is returned */
static inline dague_list_item_t*
dague_list_nolock_pop_front( dague_list_t* list );
#define dague_ulist_pop_front(list) dague_list_nolock_pop_front(list)
/** pop the last item of the list (not thread safe)
 *    if the list is empty, NULL is returned */
static inline dague_list_item_t*
dague_list_nolock_pop_back( dague_list_t* list );
#define dague_ulist_pop_back(list) dague_list_nolock_pop_back(list)

/** push item first in the list (not thread safe) */
static inline void
dague_list_nolock_push_front( dague_list_t* list,
                              dague_list_item_t* item );
#define dague_ulist_push_front(list, item) dague_list_nolock_push_front(list, item)
/** push item last in the list (not thread safe) */
static inline void
dague_list_nolock_push_back( dague_list_t* list,
                             dague_list_item_t* item );
#define dague_ulist_push_back(list, item) dague_list_nolock_push_back(list, item)

/** chains the ring of @items first in the @list (not thread safe)
 *    items->prev must point to the tail of the items collection */
static inline void
dague_list_nolock_chain_front( dague_list_t* list,
                               dague_list_item_t* items );
#define dague_ulist_chain_front(list, items) dague_list_nolock_chain_front(list, items)
/** chains the ring of @items last in the @list (not thread safe)
 *    items->prev must point to the tail of the items collection */
static inline void
dague_list_nolock_chain_back( dague_list_t* list,
                              dague_list_item_t* items );
#define dague_ulist_chain_back(list, items) dague_list_nolock_chain_back(list, items)

/** unchain the entire collection of items from the list (not thread safe)
 *    the return is a list_item ring */
static inline dague_list_item_t*
dague_list_nolock_unchain( dague_list_t* list );
#define dague_ulist_unchain(list) dague_list_nolock_unchain(list)

/* FIFO EMULATION FUNCTIONS */

/** Convenience function, same as dague_list_pop_front() */
static inline dague_list_item_t*
dague_list_fifo_pop( dague_list_t* list ) {
    return dague_list_pop_front(list); }
/** Convenience function, same as dague_list_push_back() */
static inline void
dague_list_fifo_push( dague_list_t* list, dague_list_item_t* item ) {
    dague_list_push_back(list, item); }
/** Convenience function, same as dague_list_chain_back() */
static inline void
dague_list_fifo_chain( dague_list_t* list, dague_list_item_t* items ) {
    dague_list_chain_back(list, items); }

/** Convenience function, same as dague_list_nolock_pop_front() */
static inline dague_list_item_t*
dague_list_nolock_fifo_pop( dague_list_t* list ) {
    return dague_list_nolock_pop_front(list); }
#define dague_ulist_fifo_pop(list) dague_list_nolock_fifo_pop(list)
/** Convenience function, same as dague_list_nolock_push_back() */
static inline void
dague_list_nolock_fifo_push( dague_list_t* list, dague_list_item_t* item ) {
    dague_list_nolock_push_back(list, item); }
#define dague_ulist_fifo_push(list, item) dague_list_nolock_fifo_push(list, item)
/** Convenience function, same as dague_list_nolock_chain_back() */
static inline void
dague_list_nolock_fifo_chain( dague_list_t* list, dague_list_item_t* items ) {
    dague_list_nolock_chain_back(list, items); }
#define dague_ulist_fifo_chain(list, items) dague_list_nolock_fifo_chain(list, items)


/* LIFO EMULATION FUNCTIONS */

/** Convenience function, same as dague_list_pop_front() */
static inline dague_list_item_t*
dague_list_lifo_pop( dague_list_t* list ) {
    return dague_list_pop_front(list); }
/** Convenience function, same as dague_list_push_front() */
static inline void
dague_list_lifo_push( dague_list_t* list, dague_list_item_t* item ) {
    dague_list_push_front(list, item); }
/** Convenience function, same as dague_list_chain_front() */
static inline void
dague_list_lifo_chain( dague_list_t* list, dague_list_item_t* items ) {
    dague_list_chain_front(list, items); }

/** Convenience function, same as dague_list_nolock_pop_front() */
static inline dague_list_item_t*
dague_list_nolock_lifo_pop( dague_list_t* list ) {
    return dague_list_nolock_pop_front(list); }
#define dague_ulist_lifo_pop(list) dague_list_nolock_lifo_pop(list)
/** Convenience function, same as dague_list_nolock_push_front() */
static inline void
dague_list_nolock_lifo_push( dague_list_t* list, dague_list_item_t* item ) {
    dague_list_nolock_push_front(list, item); }
#define dague_ulist_lifo_push(list, item) dague_list_nolock_lifo_push(list, item)
/** Convenience function, same as dague_list_nolock_chain_front() */
static inline void
dague_list_nolock_lifo_chain( dague_list_t* list, dague_list_item_t* items ) {
    dague_list_nolock_chain_front(list, items); }
#define dague_ulist_lifo_chain(list, items) dague_list_nolock_lifo_chain(list, items)


/***********************************************************************/
/* Interface ends here, everything else is private                     */

#define _HEAD(LIST) ((LIST)->ghost_element.list_next)
#define _TAIL(LIST) ((LIST)->ghost_element.list_prev)
#define _GHOST(LIST) (&((list)->ghost_element))

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
dague_list_lock( dague_list_t* list )
{
    dague_atomic_lock(&list->atomic_lock);
}

static inline void
dague_list_unlock( dague_list_t* list )
{
    dague_atomic_unlock(&list->atomic_lock);
}

#define _OPAQUE_LIST_ITERATOR_FIRST_DEFINITION(list) ((dague_list_item_t*)(list)->ghost_element.list_next)
#define _OPAQUE_LIST_ITERATOR_END_DEFINITION(list)   (&((list)->ghost_element))
#define _OPAQUE_LIST_ITERATOR_NEXT_DEFINITION(ITEM)  ((dague_list_item_t*)ITEM->list_next)

#define _OPAQUE_LIST_ITERATOR_LAST_DEFINITION(list)  ((dague_list_item_t*)(list)->ghost_element.list_prev)
#define _OPAQUE_LIST_ITERATOR_BEGIN_DEFINITION(list) (&((list)->ghost_element))
#define _OPAQUE_LIST_ITERATOR_PREV_DEFINITION(ITEM)  ((dague_list_item_t*)ITEM->list_prev)

#define _OPAQUE_LIST_ITERATOR_DEFINITION(list,ITEM,CODE)                \
do {                                                                    \
    dague_list_item_t* ITEM;                                            \
    dague_list_lock(list);                                              \
    for(ITEM = (dague_list_item_t*)(list)->ghost_element.list_next;     \
        ITEM != &((list)->ghost_element);                               \
        ITEM = (dague_list_item_t*)ITEM->list_next)                     \
    { CODE; }                                                           \
    dague_list_unlock(list);                                            \
} while(0)

#define _OPAQUE_ULIST_ITERATOR_DEFINITION(list,ITEM,CODE)               \
do {                                                                    \
    dague_list_item_t* ITEM;                                            \
    for(ITEM = (dague_list_item_t*)(list)->ghost_element.list_next;     \
        ITEM != &((list)->ghost_element);                               \
        ITEM = (dague_list_item_t*)ITEM->list_next)                     \
    { CODE; }                                                           \
} while(0)

static inline void
dague_list_nolock_add_before( dague_list_t* list,
                              dague_list_item_t* position,
                              dague_list_item_t* new )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert( position->belong_to == list );
#endif
    DAGUE_ITEM_ATTACH(list, new);
    new->list_prev = position->list_prev;
    new->list_next = position;
    position->list_prev->list_next = new;
    position->list_prev = new;
}

static inline void
dague_list_nolock_add_after( dague_list_t* list,
                             dague_list_item_t* position,
                             dague_list_item_t* new )
{
#if defined(DAGUE_DEBUG_ENABLE)
    assert( position->belong_to == list );
#endif
    DAGUE_ITEM_ATTACH(list, new);
    new->list_prev = position;
    new->list_next = position->list_next;
    position->list_next->list_prev = new;
    position->list_next = new;
}


static inline dague_list_item_t*
dague_list_nolock_remove( dague_list_t* list,
                          dague_list_item_t* item)
{
    assert( &list->ghost_element != item );
#if defined(DAGUE_DEBUG_ENABLE)
    assert( list == item->belong_to );
#else
    (void)list;
#endif
    dague_list_item_t* prev = (dague_list_item_t*)item->list_prev;
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
    DAGUE_ITEM_DETACH(item);
    return prev;
}

static inline void
dague_list_push_sorted( dague_list_t* list,
                        dague_list_item_t* item,
                        size_t off )
{
    dague_list_lock(list);
    dague_list_nolock_push_sorted(list, item, off);
    dague_list_unlock(list);
}

static inline void
dague_list_nolock_push_sorted( dague_list_t* list,
                               dague_list_item_t* new,
                               size_t off )
{
    dague_list_item_t* position = NULL;
    DAGUE_ULIST_ITERATOR(list, pos, {
        position = pos;
        if( A_LOWER_PRIORITY_THAN_B(new, pos, off) ) {
            break;
        }
    });
    assert(NULL != position);
    dague_ulist_add_before(list, position, new);
}

static inline void
dague_list_chain_sorted( dague_list_t* list,
                         dague_list_item_t* items,
                         size_t off )
{
    dague_list_lock(list);
    dague_list_nolock_chain_sorted(list, items, off);
    dague_list_unlock(list);
}

/* Insertion sort, but do in-place merge if sequential items are monotonic
 * random complexity is O(ln*in), but is reduced to O(ln+in)) if items
 * are already sorted; average case should be O(k*(ln+in)) for
 * scheduling k ranges of dependencies by priority*/
static inline void
dague_list_nolock_chain_sorted( dague_list_t* list,
                                dague_list_item_t* items,
                                size_t off )
{
    dague_list_item_t* new;
    dague_list_item_t* pos;
    if( NULL == items ) return;
    if( dague_list_nolock_is_empty(list) )
    {   /* the list must contain the pos element in next loop */
        new = items;
        items = dague_list_item_ring_chop(items);
        dague_list_nolock_add(list, _GHOST(list), new);
    }
    pos = (dague_list_item_t*)_TAIL(list);

    for(new = items;
        NULL != new;
        new = items)
    {
        items = dague_list_item_ring_chop(items);
        if( A_HIGHER_PRIORITY_THAN_B(new, pos, off) )
        {   /* this new item is larger than the last insert,
             * reboot and insert from the beginning */
             pos = (dague_list_item_t*)_HEAD(list);
        }
        /* search the first strictly (for stability) smaller element,
         * from the current start position, then insert before it */
        for(; pos != _GHOST(list); pos = (dague_list_item_t*)pos->list_next)
        {
            if( A_HIGHER_PRIORITY_THAN_B(new, pos, off) )
                break;
        }
        dague_list_nolock_add_before(list, pos, new);
        pos = new;
    }
}


static inline void
dague_list_sort( dague_list_t* list,
                 size_t off )
{
    dague_list_lock(list);
    dague_list_nolock_sort(list, off);
    dague_list_unlock(list);
}

static inline void
dague_list_nolock_sort( dague_list_t* list,
                        size_t off )
{
    if(dague_list_nolock_is_empty(list)) return;

    /* remove the items from the list, then chain_sort the items */
    dague_list_item_t* items;
    items = dague_list_item_ring((dague_list_item_t*)_HEAD(list),
                                 (dague_list_item_t*)_TAIL(list));
    _HEAD(list) = _GHOST(list);
    _TAIL(list) = _GHOST(list);
    dague_list_nolock_chain_sorted(list, items, off);
}

static inline void
dague_list_nolock_push_front( dague_list_t* list,
                              dague_list_item_t* item )
{
    DAGUE_ITEM_ATTACH(list, item);
    item->list_prev = _GHOST(list);
    item->list_next = _HEAD(list);
    _HEAD(list)->list_prev = item;
    _HEAD(list) = item;
}

static inline void
dague_list_push_front( dague_list_t* list,
                       dague_list_item_t *item )
{
    DAGUE_ITEM_ATTACH(list, item);
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
    DAGUE_ITEMS_ATTACH(list, items);
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
    DAGUE_ITEMS_ATTACH(list, items);
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
    DAGUE_ITEM_ATTACH(list, item);
    item->list_next = _GHOST(list);
    item->list_prev = _TAIL(list);
    _TAIL(list)->list_next = item;
    _TAIL(list) = item;
}

static inline void
dague_list_push_back( dague_list_t* list,
                      dague_list_item_t *item )
{
    DAGUE_ITEM_ATTACH(list, item);
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
    DAGUE_ITEMS_ATTACH(list, items);
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
    DAGUE_ITEMS_ATTACH(list, items);
    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
    tail->list_next = _GHOST(list);
    dague_atomic_lock(&list->atomic_lock);
    items->list_prev = _TAIL(list);
    _TAIL(list)->list_next = items;
    _TAIL(list) = tail;
    dague_atomic_unlock(&list->atomic_lock);
}

static inline dague_list_item_t*
dague_list_nolock_unchain( dague_list_t* list )
{
    dague_list_item_t* head;
    dague_list_item_t* tail;
    if( dague_list_nolock_is_empty(list) )
        return NULL;
    head = (dague_list_item_t*)_HEAD(list);
    tail = (dague_list_item_t*)_TAIL(list);
    _HEAD(list) = _GHOST(list);
    _TAIL(list) = _GHOST(list);
    dague_list_item_ring(head, tail);
    return head;
}

static inline dague_list_item_t* 
dague_list_unchain( dague_list_t* list )
{
    dague_list_item_t* head;
    dague_list_item_t* tail;
    dague_atomic_lock(&list->atomic_lock);
    if( dague_list_nolock_is_empty(list) )
        return NULL;
    head = (dague_list_item_t*)_HEAD(list);
    tail = (dague_list_item_t*)_TAIL(list);
    _HEAD(list) = _GHOST(list);
    _TAIL(list) = _GHOST(list);
    dague_atomic_unlock(&list->atomic_lock);
    dague_list_item_ring(head, tail);
    return head;
}


#define _RET_NULL_GHOST(LIST, ITEM) do {                                \
    if( _GHOST(LIST) != (ITEM) ) {                                      \
        DAGUE_ITEM_DETACH(ITEM);                                        \
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

#undef _GHOST
#undef _HEAD
#undef _TAIL

#endif  /* DAGUE_LIST_H_HAS_BEEN_INCLUDED */
