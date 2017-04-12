/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* This file contains functions to access doubly linked lists.
 * parsec_list_nolock functions are not thread safe, and
 *   can be used only when the list is locked (by list_lock) or when
 *   thread safety is ensured by another mean.
 * When locking performance is critical, one could prefer atomic lifo (see lifo.h)
 */

#ifndef PARSEC_LIST_H_HAS_BEEN_INCLUDED
#define PARSEC_LIST_H_HAS_BEEN_INCLUDED

#include <parsec/sys/atomic.h>
#include "parsec/class/list_item.h"

typedef struct parsec_list_t {
    parsec_object_t      super;
    parsec_list_item_t   ghost_element;
    parsec_atomic_lock_t atomic_lock;
} parsec_list_t;

PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_list_t);

/** lock the @list mutex, that same mutex is used by all
 *    mutex protected list operations */
static inline void
parsec_list_lock( parsec_list_t* list );
/** unlock the @list mutex, that same mutex is used by all
 *    mutex protected list operations */
static inline void
parsec_list_unlock( parsec_list_t* list );

/** check if @list is empty (mutex protected) */
static inline int parsec_list_is_empty( parsec_list_t* list );
/** check if list is empty (not thread safe) */
static inline int parsec_list_nolock_is_empty( parsec_list_t* list );

/** check if @list contains @item (not thread safe) */
static inline int parsec_list_nolock_contains( parsec_list_t *list, parsec_list_item_t *item );

/** Paste code to iterate on all items in the @LIST (front to back) (mutex protected)
 *    the @CODE_BLOCK code is applied to each item, which can be refered
 *    to as @ITEM_NAME in @CODE_BLOCK
 *    the entire loop iteration takes the list mutex, hence
 *      @CODE_BLOCK must not jump outside the block; although, break
 *      and continue are legitimate in @CODE_BLOCK
 *  @return the last considered item */
#define PARSEC_LIST_ITERATOR(LIST, ITEM_NAME, CODE_BLOCK) _OPAQUE_LIST_ITERATOR_DEFINITION(LIST,ITEM_NAME,CODE_BLOCK)
/** Paste code to iterate on all items in the @LIST (front to back) (not thread safe)
 *    the @CODE_BLOCK code is applied to each item, which can be refered
 *    to as @ITEM_NAME in @CODE_BLOCK
 *  @return the last considered item */
#define PARSEC_LIST_NOLOCK_ITERATOR(LIST, ITEM_NAME, CODE_BLOCK) _OPAQUE_LIST_NOLOCK_ITERATOR_DEFINITION(LIST,ITEM_NAME,CODE_BLOCK)

/** Alternatively: start from FIRST, until END, using NEXT to
 *  get the next element.
 *  Does not lock the list
 */
#define PARSEC_LIST_ITERATOR_FIRST(LIST)    _OPAQUE_LIST_ITERATOR_FIRST_DEFINITION(LIST)
#define PARSEC_LIST_ITERATOR_END(LIST)      _OPAQUE_LIST_ITERATOR_END_DEFINITION(LIST)
#define PARSEC_LIST_ITERATOR_NEXT(ITEM)     _OPAQUE_LIST_ITERATOR_NEXT_DEFINITION(ITEM)

#define PARSEC_LIST_ITERATOR_LAST(LIST)     _OPAQUE_LIST_ITERATOR_LAST_DEFINITION(LIST)
#define PARSEC_LIST_ITERATOR_BEGIN(LIST)    _OPAQUE_LIST_ITERATOR_BEGIN_DEFINITION(LIST)
#define PARSEC_LIST_ITERATOR_PREV(ITEM)     _OPAQUE_LIST_ITERATOR_PREV_DEFINITION(ITEM)

/** add the @newel item before the @position item in @list (not thread safe)
 *    @position item must be in @list
 *    if @position is the Ghost, @item is added back */
static inline void
parsec_list_nolock_add_before( parsec_list_t* list,
                       parsec_list_item_t* position,
                       parsec_list_item_t* newel );
/** convenience function, synonym to parsec_list_nolock_add_before() */
#define parsec_list_nolock_add(list, pos, newel) parsec_list_nolock_add_before(list, pos, newel)
/** add the @newel item after the @position item in @list (not thread safe)
 *    @position item must be in @list
 *    if @position is the Ghost, @item is added front */
static inline void
parsec_list_nolock_add_after( parsec_list_t* list,
                      parsec_list_item_t* position,
                      parsec_list_item_t* item );
/** remove a specific @item from the @list (not thread safe)
 *    @item must be in the @list
 *    @return predecessor of @item in @list */
static inline parsec_list_item_t*
parsec_list_nolock_remove( parsec_list_t* list,
                          parsec_list_item_t* item);


/* SORTED LIST FUNCTIONS */

/** add the @item before the first element of @list that is strictly smaller (mutex protected),
 *  according to the integer value at @offset in items. That is, if the input @list is
 *  sorted (descending order), the resulting list is still sorted. */
static inline void
parsec_list_push_sorted( parsec_list_t* list,
                        parsec_list_item_t* item,
                        size_t offset );
/** add the @item before the first element of @list that is striclty smaller (not thread safe),
 *  according to the integer value at @offset in items. That is, if the input @list is
 *  sorted (descending order), the resulting list is still sorted. */
static inline void
parsec_list_nolock_push_sorted( parsec_list_t* list,
                               parsec_list_item_t* item,
                               size_t offset );


/** chain the unsorted @items (mutex protected), as if they had been
 *  inserted in a loop of parsec_list_push_sorted(). That is, if the input
 * @list is sorted (descending order), the resulting list is still sorted. */
static inline void
parsec_list_chain_sorted( parsec_list_t* list,
                         parsec_list_item_t* items,
                         size_t offset );
/** chain the unsorted @items (not thread safe), as if they had been
 *  inserted in a loop by parsec_list_push_sorted(). That is, if the input
 * @list is sorted (descending order), the resulting list is still sorted. */
static inline void
parsec_list_nolock_chain_sorted( parsec_list_t* list,
                                parsec_list_item_t* items,
                                size_t offset );


/** sort @list according to the (descending) order defined by the integer
 * value at @offset in evey item (mutex protected) */
static inline void
parsec_list_sort( parsec_list_t* list,
                 size_t offset );
/** sort @list according to the (descending) order defined by the integer
 * value at @offset in evey item (not thread safe) */
static inline void
parsec_list_nolock_sort( parsec_list_t* list,
                        size_t offset );

/* DEQUEUE EMULATION FUNCTIONS */

/** pop the first item of the list (mutex protected)
 *    if the list is empty, NULL is returned */
static inline parsec_list_item_t*
parsec_list_pop_front( parsec_list_t* list );
/** pop the last item of the list (mutex protected)
 *    if the list is empty, NULL is returned */
static inline parsec_list_item_t*
parsec_list_pop_back( parsec_list_t* list );
/** try to pop the first item of the list (mutex protected)
 *    if the list is empty or currently locked, NULL is returned */
static inline parsec_list_item_t*
parsec_list_try_pop_front( parsec_list_t* list );
/** try to pop the last item of the list (mutex protected)
 *    if the list is empty or currently locked, NULL is returned */
static inline parsec_list_item_t*
parsec_list_try_pop_back( parsec_list_t* list );

/** push item first in the list (mutex protected) */
static inline void
parsec_list_push_front( parsec_list_t* list,
                       parsec_list_item_t* item );
#define parsec_list_prepend parsec_list_push_front

/** push item last in the list (mutex protected) */
static inline void
parsec_list_push_back( parsec_list_t* list,
                      parsec_list_item_t* item );
#define parsec_list_append parsec_list_push_back

/** chains the collection of items first in the list (mutex protected)
 *    items->prev must point to the tail of the items collection */
static inline void
parsec_list_chain_front( parsec_list_t* list,
                        parsec_list_item_t* items );
/** chains the collection of items last in the list (mutex protected)
 *    items->prev must point to the tail of the items collection */
static inline void
parsec_list_chain_back( parsec_list_t* list,
                       parsec_list_item_t* items );

/** unchain the entire collection of items from the list (mutex protected)
 *    the return is a list_item ring */
static inline parsec_list_item_t*
parsec_list_unchain( parsec_list_t* list );

/** pop the first item of the list (not thread safe)
 *    if the list is empty, NULL is returned */
static inline parsec_list_item_t*
parsec_list_nolock_pop_front( parsec_list_t* list );

/** pop the last item of the list (not thread safe)
 *    if the list is empty, NULL is returned */
static inline parsec_list_item_t*
parsec_list_nolock_pop_back( parsec_list_t* list );


/** push item first in the list (not thread safe) */
static inline void
parsec_list_nolock_push_front( parsec_list_t* list,
                              parsec_list_item_t* item );

/** push item last in the list (not thread safe) */
static inline void
parsec_list_nolock_push_back( parsec_list_t* list,
                             parsec_list_item_t* item );

/** chains the ring of @items first in the @list (not thread safe)
 *    items->prev must point to the tail of the items collection */
static inline void
parsec_list_nolock_chain_front( parsec_list_t* list,
                               parsec_list_item_t* items );

/** chains the ring of @items last in the @list (not thread safe)
 *    items->prev must point to the tail of the items collection */
static inline void
parsec_list_nolock_chain_back( parsec_list_t* list,
                              parsec_list_item_t* items );

/** unchain the entire collection of items from the list (not thread safe)
 *    the return is a list_item ring */
static inline parsec_list_item_t*
parsec_list_nolock_unchain( parsec_list_t* list );

/* FIFO EMULATION FUNCTIONS */

/** Convenience function, same as parsec_list_pop_front() */
static inline parsec_list_item_t*
parsec_list_fifo_pop( parsec_list_t* list ) {
    return parsec_list_pop_front(list); }
/** Convenience function, same as parsec_list_push_back() */
static inline void
parsec_list_fifo_push( parsec_list_t* list, parsec_list_item_t* item ) {
    parsec_list_push_back(list, item); }
/** Convenience function, same as parsec_list_chain_back() */
static inline void
parsec_list_fifo_chain( parsec_list_t* list, parsec_list_item_t* items ) {
    parsec_list_chain_back(list, items); }

/** Convenience function, same as parsec_list_nolock_pop_front() */
static inline parsec_list_item_t*
parsec_list_nolock_fifo_pop( parsec_list_t* list ) {
    return parsec_list_nolock_pop_front(list); }

/** Convenience function, same as parsec_list_nolock_push_back() */
static inline void
parsec_list_nolock_fifo_push( parsec_list_t* list, parsec_list_item_t* item ) {
    parsec_list_nolock_push_back(list, item); }

/** Convenience function, same as parsec_list_nolock_chain_back() */
static inline void
parsec_list_nolock_fifo_chain( parsec_list_t* list, parsec_list_item_t* items ) {
    parsec_list_nolock_chain_back(list, items); }


/* LIFO EMULATION FUNCTIONS */

/** Convenience function, same as parsec_list_pop_front() */
static inline parsec_list_item_t*
parsec_list_lifo_pop( parsec_list_t* list ) {
    return parsec_list_pop_front(list); }
/** Convenience function, same as parsec_list_push_front() */
static inline void
parsec_list_lifo_push( parsec_list_t* list, parsec_list_item_t* item ) {
    parsec_list_push_front(list, item); }
/** Convenience function, same as parsec_list_chain_front() */
static inline void
parsec_list_lifo_chain( parsec_list_t* list, parsec_list_item_t* items ) {
    parsec_list_chain_front(list, items); }

/** Convenience function, same as parsec_list_nolock_pop_front() */
static inline parsec_list_item_t*
parsec_list_nolock_lifo_pop( parsec_list_t* list ) {
    return parsec_list_nolock_pop_front(list); }

/** Convenience function, same as parsec_list_nolock_push_front() */
static inline void
parsec_list_nolock_lifo_push( parsec_list_t* list, parsec_list_item_t* item ) {
    parsec_list_nolock_push_front(list, item); }

/** Convenience function, same as parsec_list_nolock_chain_front() */
static inline void
parsec_list_nolock_lifo_chain( parsec_list_t* list, parsec_list_item_t* items ) {
    parsec_list_nolock_chain_front(list, items); }


/***********************************************************************/
/* Interface ends here, everything else is private                     */

#define _HEAD(LIST) ((LIST)->ghost_element.list_next)
#define _TAIL(LIST) ((LIST)->ghost_element.list_prev)
#define _GHOST(LIST) (&((list)->ghost_element))

static inline int
parsec_list_nolock_is_empty( parsec_list_t* list )
{
    assert( ((_HEAD(list) != _GHOST(list)) && (_TAIL(list) != _GHOST(list))) ||
            ((_HEAD(list) == _GHOST(list)) && (_TAIL(list) == _GHOST(list))) );
    return _HEAD(list) == _GHOST(list);
}

static inline int
parsec_list_is_empty( parsec_list_t* list )
{
    int rc;
    parsec_atomic_lock(&list->atomic_lock);
    rc = parsec_list_nolock_is_empty(list);
    parsec_atomic_unlock(&list->atomic_lock);
    return rc;
}

static inline void
parsec_list_lock( parsec_list_t* list )
{
    parsec_atomic_lock(&list->atomic_lock);
}

static inline void
parsec_list_unlock( parsec_list_t* list )
{
    parsec_atomic_unlock(&list->atomic_lock);
}

#define _OPAQUE_LIST_ITERATOR_FIRST_DEFINITION(list) ((parsec_list_item_t*)(list)->ghost_element.list_next)
#define _OPAQUE_LIST_ITERATOR_END_DEFINITION(list)   (&((list)->ghost_element))
#define _OPAQUE_LIST_ITERATOR_NEXT_DEFINITION(ITEM)  ((parsec_list_item_t*)((ITEM)->list_next))

#define _OPAQUE_LIST_ITERATOR_LAST_DEFINITION(list)  ((parsec_list_item_t*)(list)->ghost_element.list_prev)
#define _OPAQUE_LIST_ITERATOR_BEGIN_DEFINITION(list) (&((list)->ghost_element))
#define _OPAQUE_LIST_ITERATOR_PREV_DEFINITION(ITEM)  ((parsec_list_item_t*)((ITEM)->list_prev))

#define _OPAQUE_LIST_ITERATOR_DEFINITION(list,ITEM,CODE) ({             \
    parsec_list_item_t* ITEM;                                            \
    parsec_list_lock(list);                                              \
    for(ITEM = (parsec_list_item_t*)(list)->ghost_element.list_next;     \
        ITEM != &((list)->ghost_element);                               \
        ITEM = (parsec_list_item_t*)ITEM->list_next)                     \
    { CODE; }                                                           \
    parsec_list_unlock(list);                                            \
    ITEM;                                                               \
})

#define _OPAQUE_LIST_NOLOCK_ITERATOR_DEFINITION(list,ITEM,CODE) ({      \
    parsec_list_item_t* ITEM;                                            \
    for(ITEM = (parsec_list_item_t*)(list)->ghost_element.list_next;     \
        ITEM != &((list)->ghost_element);                               \
        ITEM = (parsec_list_item_t*)ITEM->list_next)                     \
    {                                                                   \
        CODE;                                                           \
    }                                                                   \
    ITEM;                                                               \
})

static inline int 
parsec_list_nolock_contains( parsec_list_t *list, parsec_list_item_t *item )
{
    parsec_list_item_t* litem;
    litem = PARSEC_LIST_NOLOCK_ITERATOR(list, ITEM,
        {
            if( item == ITEM )
                break;
        });
    return item == litem;
}

static inline void
parsec_list_nolock_add_before( parsec_list_t* list,
                              parsec_list_item_t* position,
                              parsec_list_item_t* newel )
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( position->belong_to == list );
#endif
    PARSEC_ITEM_ATTACH(list, newel);
    newel->list_prev = position->list_prev;
    newel->list_next = position;
    position->list_prev->list_next = newel;
    position->list_prev = newel;
}

static inline void
parsec_list_nolock_add_after( parsec_list_t* list,
                             parsec_list_item_t* position,
                             parsec_list_item_t* newel )
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( position->belong_to == list );
#endif
    PARSEC_ITEM_ATTACH(list, newel);
    newel->list_prev = position;
    newel->list_next = position->list_next;
    position->list_next->list_prev = newel;
    position->list_next = newel;
}


static inline parsec_list_item_t*
parsec_list_nolock_remove( parsec_list_t* list,
                          parsec_list_item_t* item)
{
    assert( &list->ghost_element != item );
#if defined(PARSEC_DEBUG_PARANOID)
    assert( list == item->belong_to );
#endif
    (void)list;
    parsec_list_item_t* prev = (parsec_list_item_t*)item->list_prev;
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
    PARSEC_ITEM_DETACH(item);
    return prev;
}

static inline void
parsec_list_push_sorted( parsec_list_t* list,
                        parsec_list_item_t* item,
                        size_t off )
{
    parsec_list_lock(list);
    parsec_list_nolock_push_sorted(list, item, off);
    parsec_list_unlock(list);
}

static inline void
parsec_list_nolock_push_sorted( parsec_list_t* list,
                               parsec_list_item_t* newel,
                               size_t off )
{
    parsec_list_item_t* position = PARSEC_LIST_NOLOCK_ITERATOR(list, pos,
    {
        if( A_HIGHER_PRIORITY_THAN_B(newel, pos, off) )
            break;
    });
    parsec_list_nolock_add_before(list, position, newel);
}

static inline void
parsec_list_chain_sorted( parsec_list_t* list,
                         parsec_list_item_t* items,
                         size_t off )
{
    parsec_list_lock(list);
    parsec_list_nolock_chain_sorted(list, items, off);
    parsec_list_unlock(list);
}

/* Insertion sort, but do in-place merge if sequential items are monotonic
 * random complexity is O(ln*in), but is reduced to O(ln+in)) if items
 * are already sorted; average case should be O(k*(ln+in)) for
 * scheduling k ranges of dependencies by priority*/
static inline void
parsec_list_nolock_chain_sorted( parsec_list_t* list,
                                parsec_list_item_t* items,
                                size_t off )
{
    parsec_list_item_t* newel;
    parsec_list_item_t* pos;
    if( NULL == items ) return;
    if( parsec_list_nolock_is_empty(list) )
    {   /* the list must contain the pos element in next loop */
        newel = items;
        items = parsec_list_item_ring_chop(items);
        parsec_list_nolock_add(list, _GHOST(list), newel);
    }
    pos = (parsec_list_item_t*)_TAIL(list);

    for(newel = items;
        NULL != newel;
        newel = items)
    {
        items = parsec_list_item_ring_chop(items);
        if( A_HIGHER_PRIORITY_THAN_B(newel, pos, off) )
        {   /* this newel item is larger than the last insert,
             * reboot and insert from the beginning */
             pos = (parsec_list_item_t*)_HEAD(list);
        }
        /* search the first strictly (for stability) smaller element,
         * from the current start position, then insert before it */
        for(; pos != _GHOST(list); pos = (parsec_list_item_t*)pos->list_next)
        {
            if( A_HIGHER_PRIORITY_THAN_B(newel, pos, off) )
                break;
        }
        parsec_list_nolock_add_before(list, pos, newel);
        pos = newel;
    }
}

/*
 * http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
 *   by Simon Tatham
 *
 * A simple mergesort implementation on lists. Complexity O(N*log(N)).
 */
static inline void
parsec_list_nolock_chain_sort_mergesort(parsec_list_t *list,
                                       size_t off)
{
    parsec_list_item_t *items, *p, *q, *e, *tail, *oldhead;
    int insize, nmerges, psize, qsize, i;

    /* Remove the items from the list, and clean the list */
    items = parsec_list_item_ring((parsec_list_item_t*)_HEAD(list),
                                 (parsec_list_item_t*)_TAIL(list));
    _HEAD(list) = _GHOST(list);
    _TAIL(list) = _GHOST(list);

    insize = 1;

    while (1) {
        p = items;
        oldhead = items;            /* only used for circular linkage */
        items = NULL;
        tail = NULL;

        nmerges = 0;  /* count number of merges we do in this pass */

        while (p) {
            nmerges++;  /* there exists a merge to be done */
            /* step `insize' places along from p */
            q = p;
            psize = 0;
            for (i = 0; i < insize; i++) {
                psize++;
                q = (parsec_list_item_t*)(q->list_next == oldhead ? NULL : q->list_next);
                if (!q) break;
            }

            /* if q hasn't fallen off end, we have two lists to merge */
            qsize = insize;

            /* now we have two lists; merge them */
            while (psize > 0 || (qsize > 0 && q)) {

                /* decide whether next element of merge comes from p or q */
                if (psize == 0) {
                    /* p is empty; e must come from q. */
                    e = q; q = (parsec_list_item_t*)q->list_next; qsize--;
                    if (q == oldhead) q = NULL;
                } else if (qsize == 0 || !q) {
                    /* q is empty; e must come from p. */
                    e = p; p = (parsec_list_item_t*)p->list_next; psize--;
                    if (p == oldhead) p = NULL;
                } else if (A_LOWER_PRIORITY_THAN_B(p, q, off)) {
                    /* First element of p is lower (or same);
                     * e must come from p. */
                    e = p; p = (parsec_list_item_t*)p->list_next; psize--;
                    if (p == oldhead) p = NULL;
                } else {
                    /* First element of q is lower; e must come from q. */
                    e = q; q = (parsec_list_item_t*)q->list_next; qsize--;
                    if (q == oldhead) q = NULL;
                }

                /* add the next element to the merged list */
                if (tail) {
                    tail->list_next = e;
                } else {
                    items = e;
                }
                /* Maintain reverse pointers in a doubly linked list. */
                e->list_prev = tail;
                tail = e;
            }

            /* now p has stepped `insize' places along, and q has too */
            p = q;
        }
        tail->list_next = items;
        items->list_prev = tail;

        /* If we have done only one merge, we're finished. */
        if (nmerges <= 1)   /* allow for nmerges==0, the empty list case */
            break;

        /* Otherwise repeat, merging lists twice the size */
        insize *= 2;
    }
    parsec_list_nolock_chain_front(list, items);
}

static inline void
parsec_list_sort( parsec_list_t* list,
                 size_t off )
{
    parsec_list_lock(list);
    parsec_list_nolock_sort(list, off);
    parsec_list_unlock(list);
}

static inline void
parsec_list_nolock_sort( parsec_list_t* list,
                        size_t off )
{
    if(parsec_list_nolock_is_empty(list)) return;

#if 0
    /* remove the items from the list, then chain_sort the items */
    parsec_list_item_t* items;
    items = parsec_list_item_ring((parsec_list_item_t*)_HEAD(list),
                                 (parsec_list_item_t*)_TAIL(list));
    _HEAD(list) = _GHOST(list);
    _TAIL(list) = _GHOST(list);
    parsec_list_nolock_chain_sorted(list, items, off);
#else
    parsec_list_nolock_chain_sort_mergesort(list, off);
#endif
}

static inline void
parsec_list_nolock_push_front( parsec_list_t* list,
                              parsec_list_item_t* item )
{
    PARSEC_ITEM_ATTACH(list, item);
    item->list_prev = _GHOST(list);
    item->list_next = _HEAD(list);
    _HEAD(list)->list_prev = item;
    _HEAD(list) = item;
}

static inline void
parsec_list_push_front( parsec_list_t* list,
                       parsec_list_item_t *item )
{
    PARSEC_ITEM_ATTACH(list, item);
    item->list_prev = _GHOST(list);
    parsec_atomic_lock(&list->atomic_lock);
    item->list_next = _HEAD(list);
    _HEAD(list)->list_prev = item;
    _HEAD(list) = item;
    parsec_atomic_unlock(&list->atomic_lock);
}

static inline void
parsec_list_nolock_chain_front( parsec_list_t* list,
                               parsec_list_item_t* items )
{
    PARSEC_ITEMS_ATTACH(list, items);
    parsec_list_item_t* tail = (parsec_list_item_t*)items->list_prev;
    items->list_prev = _GHOST(list);
    tail->list_next = _HEAD(list);
    _HEAD(list)->list_prev = tail;
    _HEAD(list) = items;
}

static inline void
parsec_list_chain_front( parsec_list_t* list,
                        parsec_list_item_t* items )
{
    PARSEC_ITEMS_ATTACH(list, items);
    parsec_list_item_t* tail = (parsec_list_item_t*)items->list_prev;
    items->list_prev = _GHOST(list);
    parsec_atomic_lock(&list->atomic_lock);
    tail->list_next = _HEAD(list);
    _HEAD(list)->list_prev = tail;
    _HEAD(list) = items;
    parsec_atomic_unlock(&list->atomic_lock);
}


static inline void
parsec_list_nolock_push_back( parsec_list_t* list,
                             parsec_list_item_t *item )
{
    PARSEC_ITEM_ATTACH(list, item);
    item->list_next = _GHOST(list);
    item->list_prev = _TAIL(list);
    _TAIL(list)->list_next = item;
    _TAIL(list) = item;
}

static inline void
parsec_list_push_back( parsec_list_t* list,
                      parsec_list_item_t *item )
{
    PARSEC_ITEM_ATTACH(list, item);
    item->list_next = _GHOST(list);
    parsec_atomic_lock(&list->atomic_lock);
    item->list_prev = _TAIL(list);
    _TAIL(list)->list_next = item;
    _TAIL(list) = item;
    parsec_atomic_unlock(&list->atomic_lock);
}

static inline void
parsec_list_nolock_chain_back( parsec_list_t* list,
                              parsec_list_item_t* items )
{
    PARSEC_ITEMS_ATTACH(list, items);
    parsec_list_item_t* tail = (parsec_list_item_t*)items->list_prev;
    tail->list_next = _GHOST(list);
    items->list_prev = _TAIL(list);
    _TAIL(list)->list_next = items;
    _TAIL(list) = tail;
}

static inline void
parsec_list_chain_back( parsec_list_t* list,
                       parsec_list_item_t* items )
{
    PARSEC_ITEMS_ATTACH(list, items);
    parsec_list_item_t* tail = (parsec_list_item_t*)items->list_prev;
    tail->list_next = _GHOST(list);
    parsec_atomic_lock(&list->atomic_lock);
    items->list_prev = _TAIL(list);
    _TAIL(list)->list_next = items;
    _TAIL(list) = tail;
    parsec_atomic_unlock(&list->atomic_lock);
}

static inline parsec_list_item_t*
parsec_list_nolock_unchain( parsec_list_t* list )
{
    parsec_list_item_t* head;
    parsec_list_item_t* tail;
    if( parsec_list_nolock_is_empty(list) )
        return NULL;
    head = (parsec_list_item_t*)_HEAD(list);
    tail = (parsec_list_item_t*)_TAIL(list);
    _HEAD(list) = _GHOST(list);
    _TAIL(list) = _GHOST(list);
    parsec_list_item_ring(head, tail);
    return head;
}

static inline parsec_list_item_t*
parsec_list_unchain( parsec_list_t* list )
{
    parsec_list_item_t* head;

    parsec_atomic_lock(&list->atomic_lock);
    head = parsec_list_nolock_unchain(list);
    parsec_atomic_unlock(&list->atomic_lock);
    return head;
}


#define _RET_NULL_GHOST(LIST, ITEM) do {                                \
    if( _GHOST(LIST) != (ITEM) ) {                                      \
        PARSEC_ITEM_DETACH(ITEM);                                        \
        return (ITEM);                                                  \
    }                                                                   \
    return NULL;                                                        \
} while(0)

static inline parsec_list_item_t*
parsec_list_nolock_pop_front( parsec_list_t* list )
{
    parsec_list_item_t* item = (parsec_list_item_t*)_HEAD(list);
    _HEAD(list) = item->list_next;
    _HEAD(list)->list_prev = &list->ghost_element;
    _RET_NULL_GHOST(list, item);
}

static inline parsec_list_item_t*
parsec_list_pop_front( parsec_list_t* list )
{
    parsec_atomic_lock(&list->atomic_lock);
    parsec_list_item_t* item = parsec_list_nolock_pop_front(list);
    parsec_atomic_unlock(&list->atomic_lock);
    return item;
}

static inline parsec_list_item_t*
parsec_list_try_pop_front( parsec_list_t* list)
{
    if( !parsec_atomic_trylock(&list->atomic_lock) ) {
        return NULL;
    }
    parsec_list_item_t* item = parsec_list_nolock_pop_front(list);
    parsec_atomic_unlock(&list->atomic_lock);
    return item;
}


static inline parsec_list_item_t*
parsec_list_nolock_pop_back( parsec_list_t* list )
{
    parsec_list_item_t* item = (parsec_list_item_t*)_TAIL(list);
    _TAIL(list) = item->list_prev;
    _TAIL(list)->list_next = _GHOST(list);
    _RET_NULL_GHOST(list, item);
}

static inline parsec_list_item_t*
parsec_list_pop_back( parsec_list_t* list )
{
    parsec_atomic_lock(&list->atomic_lock);
    parsec_list_item_t* item = parsec_list_nolock_pop_back(list);
    parsec_atomic_unlock(&list->atomic_lock);
    return item;
}

static inline parsec_list_item_t*
parsec_list_try_pop_back( parsec_list_t* list)
{
    if( !parsec_atomic_trylock(&list->atomic_lock) ) {
        return NULL;
    }
    parsec_list_item_t* item = parsec_list_nolock_pop_back(list);
    parsec_atomic_unlock(&list->atomic_lock);
    return item;
}

#undef _RET_NULL_GHOST

#undef _GHOST
#undef _HEAD
#undef _TAIL

#endif  /* PARSEC_LIST_H_HAS_BEEN_INCLUDED */
