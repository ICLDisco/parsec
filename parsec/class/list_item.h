/*
 * Copyright (c) 2010-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_LIST_ITEM_H_HAS_BEEN_INCLUDED
#define PARSEC_LIST_ITEM_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/class/parsec_object.h"
#include <stdlib.h>
#include <assert.h>

/**
 * @defgroup parsec_internal_classes_listitem List Item
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief List Items for PaRSEC
 *
 *  @details List Items are used in various structures to chain
 *    elements one with the others. See @ref
 *    parsec_internal_classes_list, @ref parsec_internal_classes_lifo,
 *    @ref parsec_internal_classes_fifo, for examples of data structures
 *    that use list items.
 * 
 *    Functions and macros in this group are used to manipulate the
 *    list items.
 */

BEGIN_C_DECLS

/**
 * @brief List Item structure
 */
typedef struct parsec_list_item_s {
    parsec_object_t  super;                         /**< A list item is a @ref parsec_internal_classes_object */
    volatile struct parsec_list_item_s* list_next;  /**< Pointer to the next item */
    volatile struct parsec_list_item_s* list_prev;  /**< Pointer to the previous item */
    int32_t aba_key;                                /**< This field is __very__ special and should be handled with extreme
                                                     *   care. It is used to avoid the ABA problem when atomic operations
                                                     *   are in use and we do not have support for 128 bits atomics.
                                                     */
#if defined(PARSEC_DEBUG_PARANOID)
    volatile int32_t refcount;                      /**< Number of higher data structures (e.g. lists) that are still pointing to this item */
    volatile void* belong_to;                       /**< Higher data structure into which this item belongs */
#endif  /* defined(PARSEC_DEBUG_PARANOID) */
} parsec_list_item_t;

PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_list_item_t);

/**
 * @brief Convenience macro to access the next field without casting
 *        an object that derives a List Item
 */
#define PARSEC_LIST_ITEM_NEXT(item) ((__typeof__(item))(((parsec_list_item_t*)(item))->list_next))
/**
 * @brief Convenience macro to access the prev field without casting
 *        an object that derives a List Item
 */
#define PARSEC_LIST_ITEM_PREV(item) ((__typeof__(item))(((parsec_list_item_t*)(item))->list_prev))

/** 
 * @brief
 *  Make a well formed singleton ring with a list item.
 *
 * @details
 *   @param[inout] item the item to singleton
 *   @return a valid list item ring containing itself
 */
static inline parsec_list_item_t*
parsec_list_item_singleton( parsec_list_item_t* item )
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert(0 == item->refcount);
    item->belong_to = item;
#endif
    item->list_next = item;
    item->list_prev = item;
    return item;
}
/**
 * @brief Convenience macro to singleton an object that derives a List Item
 */
#define PARSEC_LIST_ITEM_SINGLETON(item) parsec_list_item_singleton((parsec_list_item_t*) item)

/**
 * @brief
 *   Make a ring from a chain of items
 *
 * @details
 *  Starting with first, ending with last, returns first.
 *    if first->last is not a valid chain of items, result is undetermined
 *    in PARSEC_DEBUG_PARANOID mode, attached items are detached, must be reattached if needed 
 * @param[inout] first the first item of the chain
 * @param[inout] last the last item of the chain
 * @return first after it has been chained to last to make a ring
 */
static inline parsec_list_item_t*
parsec_list_item_ring( parsec_list_item_t* first, parsec_list_item_t* last )
{
    first->list_prev = last;
    last->list_next = first;

#if defined(PARSEC_DEBUG_PARANOID)
    if( 1 == first->refcount )
    {   /* Pseudo detach the items if they had been attached */
        parsec_list_item_t* item = first;
        do {
            assert( item->belong_to == first->belong_to );
            item->refcount--;
            assert( 0 == item->refcount );
            item = (parsec_list_item_t*)item->list_next;
        } while(item != first);
    }
#endif

    return first;
}

/**
 * @brief
 *   Add an item to an item ring.
 *
 * @details
 *   item is added to the item ring ring, preceding ring
 *  @param[inout] ring the ring of items to which item should be added
 *  @param[inout] item the item to add to ring
 *  @return ring, the list item representing the ring
 * @remark
 *  This is not a thread safe function
 */
static inline parsec_list_item_t*
parsec_list_item_ring_push( parsec_list_item_t* ring,
                           parsec_list_item_t* item )
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( 0 == item->refcount );
    assert( (parsec_list_item_t*)(void*)0xdeadbeefL != ring->list_next );
    assert( (parsec_list_item_t*)(void*)0xdeadbeefL != ring->list_prev );
#endif
    item->list_next = ring;
    item->list_prev = ring->list_prev;
    ring->list_prev->list_next = item;
    ring->list_prev = item;
    return ring;
}

/**
 * @brief
 *   Merge to ring of items.
 *
 * @details
 *   ring2 is added to the item ring ring1, succeeding ring1
 *  @param[inout] ring1 the ring of items to which ring2 should be added
 *  @param[inout] ring2 the ring of items to add to ring1
 *  @return ring1, the list item representing the merged ring
 * @remark
 *  This is not a thread safe function
 */
static inline parsec_list_item_t*
parsec_list_item_ring_merge( parsec_list_item_t* ring1,
                            parsec_list_item_t* ring2 )
{
    volatile parsec_list_item_t *tmp;
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (parsec_list_item_t*)(void*)0xdeadbeefL != ring1->list_next );
    assert( (parsec_list_item_t*)(void*)0xdeadbeefL != ring1->list_prev );
    assert( (parsec_list_item_t*)(void*)0xdeadbeefL != ring2->list_next );
    assert( (parsec_list_item_t*)(void*)0xdeadbeefL != ring2->list_prev );
#endif
    ring2->list_prev->list_next = ring1;
    ring1->list_prev->list_next = ring2;
    tmp = ring1->list_prev;
    ring1->list_prev = ring2->list_prev;
    ring2->list_prev = tmp;

    return ring1;
}

/**
 * @brief
 *   Removes an item from a ring of items.
 *
 * @details
 *   item must belong to a ring. It is singletoned, and the ring without
 *   item is returned.
 *  @param[inout] item the item from the ring of items to be removed.
 *  @return the rest of the ring
 * @remark
 *  This is not a thread safe function
 */
static inline parsec_list_item_t*
parsec_list_item_ring_chop( parsec_list_item_t* item )
{
#if defined(PARSEC_DEBUG_PARANOID)
    assert( (parsec_list_item_t*)(void*)0xdeadbeefL != item->list_next );
    assert( (parsec_list_item_t*)(void*)0xdeadbeefL != item->list_prev );
#endif
    parsec_list_item_t* ring = (parsec_list_item_t*)item->list_next;
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
#if defined(PARSEC_DEBUG_PARANOID)
    if(item->refcount) item->refcount--;
    item->list_prev = (parsec_list_item_t*)(void*)0xdeadbeefL;
    item->list_next = (parsec_list_item_t*)(void*)0xdeadbeefL;
#endif
    if(ring == item) return NULL;
    return ring;
}

/**
 * @brief Convenience macro to apply CODE on the elements of a ring
 *        of items
 *
 * @details
 *  Paste a code that execute 'CODE', assigning all elements between
 *  RING (included) and LAST (excluded) to ITEM before.
 * @param RING the first element on which CODE should be applied
 * @param LAST the last element on which CODE should not be applied (e.g.
 *  if LAST == RING, CODE is applied on all the items of RING
 * @param ITEM the name of a variable to use in CODE and that will be assigned
 *  to each item between RING and LAST
 * @param CODE A piece of code to execute for each value of ITEM
 * @return the last item (LAST)
 */
#define _LIST_ITEM_ITERATOR(RING, LAST, ITEM, CODE)                     \
    ({                                                                  \
        parsec_list_item_t* ITEM = (parsec_list_item_t*)(RING);           \
        do {                                                            \
            {                                                           \
                CODE;                                                   \
            }                                                           \
            ITEM = (parsec_list_item_t*)ITEM->list_next;                 \
        } while (ITEM != (LAST));                                       \
        ITEM;                                                           \
    })

/**
 * @brief
 *   Insert an item into a sorted items ring, preserving the sorted property
 *
 * @details
 *   Assuming there is an integer off bytes after the beginning of each item,
 *   inserts item before the first p of ring such that
 *     A_LOWER_PRIORITY_THAN_B(item, p, off) is false
 * @param[inout] ring the sorted ring of items
 * @param[inout] item the item to add
 * @param[in] off the offset where the integer to use to sort items can be found
 * @return the newly formed ring of items (can be equal or different from ring)
 * @remark This function is not thread safe
 */
static inline parsec_list_item_t*
parsec_list_item_ring_push_sorted( parsec_list_item_t* ring,
                                   parsec_list_item_t* item,
                                   size_t off )
{
    parsec_list_item_t* position;
    int success = 0;

    parsec_list_item_singleton(item);
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
    parsec_list_item_ring_push(position, item);
    if( success && (ring == position) ) return item;
    return ring;
}

/* This is debug helpers for list items accounting */
/**
 * Don't include the implementation in the doxygen documentation
 * @cond FALSE
 */
#if defined(PARSEC_DEBUG_PARANOID)
#define PARSEC_ITEMS_VALIDATE(ITEMS)                                    \
    do {                                                                \
        parsec_list_item_t *__end = (ITEMS);                            \
        int _number; parsec_list_item_t *__item;                        \
        for(_number=0, __item = (parsec_list_item_t*)__end->list_next;  \
            __item != __end;                                            \
            __item = (parsec_list_item_t*)__item->list_next ) {         \
            assert( (__item->refcount == 0) || (__item->refcount == 1) ); \
            assert( __end->refcount == __item->refcount );              \
            if( __item->refcount == 1 )                                 \
                assert(__item->belong_to == __end->belong_to);          \
            if( ++_number > 1000 ) assert(0);                           \
        }                                                               \
    } while(0)

#define PARSEC_ITEM_ATTACH(LIST, ITEM)                                  \
    do {                                                                \
        parsec_list_item_t *_item_ = (ITEM);                            \
        _item_->refcount++;                                             \
        assert( 1 == _item_->refcount );                                \
        _item_->belong_to = (LIST);                                     \
    } while(0)

#define PARSEC_ITEMS_ATTACH(LIST, ITEMS)                                \
    do {                                                                \
        parsec_list_item_t *_item = (ITEMS);                            \
        assert( (parsec_list_item_t*)(void*)0xdeadbeefL != _item->list_next ); \
        assert( (parsec_list_item_t*)(void*)0xdeadbeefL != _item->list_prev ); \
        parsec_list_item_t *_end = (parsec_list_item_t *)_item->list_prev; \
        do {                                                            \
            PARSEC_ITEM_ATTACH(LIST, _item);                            \
            _item = (parsec_list_item_t*)_item->list_next;              \
        } while(_item != _end->list_next);                              \
    } while(0)

#define PARSEC_ITEM_DETACH(ITEM)                                        \
    do {                                                                \
        parsec_list_item_t *_item = (ITEM);                             \
        /* check for not poping the ghost element */\
        assert( _item->belong_to != (void*)_item );                     \
        _item->list_prev = (parsec_list_item_t*)(void*)0xdeadbeefL;     \
        _item->list_next = (parsec_list_item_t*)(void*)0xdeadbeefL;     \
        _item->refcount--;                                              \
        assert( 0 == _item->refcount );                                 \
    } while (0)
#else
/** @endcond */
/**
 * @brief Check that an item is well formed (it does belong to the structure it is supposed to)
 */
#define PARSEC_ITEMS_VALIDATE_ELEMS(ITEMS) do { (void)(ITEMS); } while(0)
/**
 * @brief Attach an item to a higher level structure
 */
#define PARSEC_ITEM_ATTACH(LIST, ITEM) do { (void)(LIST); (void)(ITEM); } while(0)
/**
 * @brief Attach a ring of items to a higher level structure
 */
#define PARSEC_ITEMS_ATTACH(LIST, ITEMS) do { (void)(LIST); (void)(ITEMS); } while(0)
/**
 * @brief Dettach an item from a higher level structure
 */
#define PARSEC_ITEM_DETACH(ITEM) do { (void)(ITEM); } while(0)
#endif  /* PARSEC_DEBUG_PARANOID */

END_C_DECLS

/** @} */

#endif

