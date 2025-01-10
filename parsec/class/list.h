/*
 * Copyright (c) 2010-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2024      Stony Brook University.  All rights reserved.
 */

/* This file contains functions to access doubly linked lists.
 * parsec_list_nolock functions are not thread safe, and
 *   can be used only when the list is locked (by list_lock) or when
 *   thread safety is ensured by another mean. All other functions are
 *   thread safe.
 * When locking performance is critical, one could prefer atomic lifo (see lifo.h)
 */

#ifndef PARSEC_LIST_H_HAS_BEEN_INCLUDED
#define PARSEC_LIST_H_HAS_BEEN_INCLUDED

#include <stdbool.h>

#include "parsec/parsec_config.h"
#include "parsec/class/list_item.h"
#include "parsec/sys/atomic.h"

/**
 * @defgroup parsec_internal_classes_list Linked Lists
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief Linked Lists management functions
 *
 *  @details This implements a doubly-linked list set of
 *      functions.
 */

/** @cond FALSE */
BEGIN_C_DECLS
/** @endcond */

/**
 * @brief List Head structure
 */
typedef struct parsec_list_t {
    parsec_object_t      super;          /**< A list head is a PaRSEC Object */
    parsec_list_item_t   ghost_element;  /**< Elements get chained to the ghost element; an empty list
                                          *   has only the ghost element */
    parsec_atomic_lock_t atomic_lock;    /**< The list is protected through this lock */
} parsec_list_t;

/** @cond FALSE */
PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_list_t);
/** @endcond */

/**
 * @brief locks the list mutex.
 *
 * @details that same mutex is used by all mutex protected list
 *    operations
 *
 * @param[inout] list the list to lock
 */
static inline void
parsec_list_lock( parsec_list_t* list );

/**
 * @brief unlocks the list mutex.
 *
 * @details that same mutex is used by all mutex protected list
 *    operations
 *
 * @param[inout] list the list to unlock
 */
static inline void
parsec_list_unlock( parsec_list_t* list );

/**
 * @brief check if list is empty
 *
 * @param[inout] list the list to check
 * @return 0 if the list is not empty, 1 otherwise
 *
 * @remark this function is thread safe
 */
static inline int parsec_list_is_empty( parsec_list_t* list );

/**
 * @brief check if list is empty, ignoring the lock
 *
 * @param[in] list the list to check
 * @return 0 if the list is not empty, 1 otherwise
 *
 * @remark this function is not thread safe
 */
static inline int parsec_list_nolock_is_empty( parsec_list_t* list );

/**
 * @brief check if an element belongs to the list, ignoring the lock
 *
 * @param[in] list the list in which item is searched
 * @param[in] item the item to check against
 * @return 1 if item (the pointer value) belongs to list, 0 otherwise.
 *
 * @remark this function is not thread safe
 * @remark item is compared to other points belonging to the list; this
 *  function does not check that the items are similar objects, it checks
 *  if a given item address is already in the list.
 */
static inline int parsec_list_nolock_contains( parsec_list_t *list, parsec_list_item_t *item );

/**
 * @brief List iterator macro
 *
 * @details Paste code to iterate on all items in the LIST (front to back)
 *    the CODE_BLOCK code is applied to each item, which can be referred
 *    to as ITEM_NAME in CODE_BLOCK
 *    the entire loop iteration takes the list mutex, hence
 *      CODE_BLOCK must not jump outside the block; although, break
 *      and continue are legitimate in CODE_BLOCK
 *
 * @param[inout] LIST the list on which to iterate
 * @param[inout] ITEM_NAME the variable to use as item name in CODE_BLOCK
 * @param[in] CODE_BLOCK a block of code to execute with each item (break
 *    and continue are allowed)
 * @return the last considered item
 *
 * @remark lock on LIST is taken from the beginning to the end of this loop
 */
#define PARSEC_LIST_ITERATOR(LIST, ITEM_NAME, CODE_BLOCK) _OPAQUE_LIST_ITERATOR_DEFINITION(LIST,ITEM_NAME,CODE_BLOCK)

/**
 * @brief List iterator macro without taking the lock on the list
 *
 * @details Paste code to iterate on all items in the LIST (front to back)
 *    the CODE_BLOCK code is applied to each item, which can be refered
 *    to as ITEM_NAME in CODE_BLOCK
 *    the entire loop iteration takes the list mutex, hence
 *      CODE_BLOCK must not jump outside the block; although, break
 *      and continue are legitimate in CODE_BLOCK
 *
 * @param[inout] LIST the list on which to iterate
 * @param[inout] ITEM_NAME the variable to use as item name in CODE_BLOCK
 * @param[in] CODE_BLOCK a block of code to execute with each item (break
 *    and continue are allowed)
 * @return the last considered item
 *
 * @remark this is not thread safe
 */
#define PARSEC_LIST_NOLOCK_ITERATOR(LIST, ITEM_NAME, CODE_BLOCK) _OPAQUE_LIST_NOLOCK_ITERATOR_DEFINITION(LIST,ITEM_NAME,CODE_BLOCK)

/**
 * @brief Reverse list iterator macro without taking the lock on the list
 *
 * @details Paste code to iterate on all items in the LIST (front to back)
 *    the CODE_BLOCK code is applied to each item, which can be refered
 *    to as ITEM_NAME in CODE_BLOCK
 *    the entire loop iteration takes the list mutex, hence
 *      CODE_BLOCK must not jump outside the block; although, break
 *      and continue are legitimate in CODE_BLOCK
 *
 * @param[inout] LIST the list on which to iterate
 * @param[inout] ITEM_NAME the variable to use as item name in CODE_BLOCK
 * @param[in] CODE_BLOCK a block of code to execute with each item (break
 *    and continue are allowed)
 * @return the last considered item
 *
 * @remark this is not thread safe
 */
#define PARSEC_LIST_NOLOCK_REV_ITERATOR(LIST, ITEM_NAME, CODE_BLOCK) _OPAQUE_LIST_NOLOCK_REV_ITERATOR_DEFINITION(LIST,ITEM_NAME,CODE_BLOCK)

/**
 * @brief iterator convenience macro: get the first element of a list
 *
 * @details if item == PARSEC_LIST_ITERATOR_END(list), then item is not an item in the
 *          list, it is a marker that the end of the loop has been reached
 */
#define PARSEC_LIST_ITERATOR_FIRST(LIST)    _OPAQUE_LIST_ITERATOR_FIRST_DEFINITION(LIST)

/**
 * @brief iterator convenience macro: get the end of a list
 *
 * @details if item == PARSEC_LIST_ITERATOR_END(list), then item is not an item in the
 *          list, it is a marker that the end of the loop has been reached
 */
#define PARSEC_LIST_ITERATOR_END(LIST)      _OPAQUE_LIST_ITERATOR_END_DEFINITION(LIST)

/**
 * @brief gets the next item from an item in a list
 *
 * @details PARSEC_LIST_ITERATOR_NEXT(item) does not necessarily return
 *          an element that belongs to the list. The returned value must
 *          be checked against PARSEC_LIST_ITERATOR_END(list)
 */
#define PARSEC_LIST_ITERATOR_NEXT(ITEM)     _OPAQUE_LIST_ITERATOR_NEXT_DEFINITION(ITEM)

/**
 * @brief iterator convenience macro: get the last element of a list
 *
 * @details if item == PARSEC_LIST_ITERATOR_BEGIN(list), then item is not an item in the
 *          list, it is a marker that the end of the loop has been reached
 */
#define PARSEC_LIST_ITERATOR_LAST(LIST)     _OPAQUE_LIST_ITERATOR_LAST_DEFINITION(LIST)

/**
 * @brief iterator convenience macro: get the beginning of a list
 *
 * @details if item == PARSEC_LIST_ITERATOR_BEGIN(list), then item is not an item in the
 *          list, it is a marker that the end of the loop has been reached
 */
#define PARSEC_LIST_ITERATOR_BEGIN(LIST)    _OPAQUE_LIST_ITERATOR_BEGIN_DEFINITION(LIST)

/**
 * @brief gets the previous item from an item in a list
 *
 * @details PARSEC_LIST_ITERATOR_PREV(item) does not necessarily return
 *          an element that belongs to the list. The returned value must
 *          be checked against PARSEC_LIST_ITERATOR_BEGIN(list)
 */
#define PARSEC_LIST_ITERATOR_PREV(ITEM)     _OPAQUE_LIST_ITERATOR_PREV_DEFINITION(ITEM)

/**
 * @brief inserts an element in a list before another element without
 *        checking if the list is locked
 *
 * @details add the newel item before the position item in list
 *
 * @param[inout] list the list in which position belongs and in which newel must
 *                be added
 * @param[inout] position the element that must succeed to newel
 * @param[inout] newel the element the add to list before position
 *
 * @remark position item must be in list
 * @remark this function is not thread safe
 * @remark if position is the Ghost Element, item is added back
 */
static inline void
parsec_list_nolock_add_before( parsec_list_t* list,
                       parsec_list_item_t* position,
                       parsec_list_item_t* newel );

/**
 * @brief convenience macro: default behavior is to add before pos
 */
#define parsec_list_nolock_add(list, pos, newel) parsec_list_nolock_add_before(list, pos, newel)

/**
 * @brief inserts an element in a list after another element without
 *        checking if the list is locked
 *
 * @details add the newel item after the position item in list
 *
 * @param[inout] list the list in which position belongs and in which newel must
 *                be added
 * @param[inout] position the element that must precede newel
 * @param[inout] item the element the add to list after position
 *
 * @remark position item must be in list
 * @remark this function is not thread safe
 * @remark if position is the Ghost Element, item is added front
 */
static inline void
parsec_list_nolock_add_after( parsec_list_t* list,
                      parsec_list_item_t* position,
                      parsec_list_item_t* item );
static inline void
parsec_list_add_after( parsec_list_t* list,
                      parsec_list_item_t* position,
                      parsec_list_item_t* item );

/**
 * @brief removes an element from a list without
 *        checking if the list is locked
 *
 * @param[inout] list the list in which item belongs and from which
 *               it must be removed
 * @param[inout] item the element the remove from list
 * @return the predecessor of item in list
 *
 * @remark item must be in list
 * @remark this function is not thread safe
 */
static inline parsec_list_item_t*
parsec_list_nolock_remove( parsec_list_t* list,
                          parsec_list_item_t* item);


/* SORTED LIST FUNCTIONS */

/**
 * @brief Insert an item into a sorted list, keeping it sorted
 *
 * @details add item before the first element of list that is strictly
 *  smaller (mutex protected), according to the integer value at
 *  offset in items. That is, if the input list is sorted (descending
 *  order), the resulting list is still sorted.
 *
 * @param[inout] list the sorted list in which item should be inserted
 * @param[inout] item an item to insert in list
 * @param[in] offset the offset (in bytes) from the beginning of item
 *            in which an integer (sizeof(int) bytes) can be found.
 *            All items in list are assumed to have an integer at the
 *            same offset. Natural order is used to sort the items.
 *
 * @remark this function is thread safe
 */
static inline void
parsec_list_push_sorted( parsec_list_t* list,
                        parsec_list_item_t* item,
                        size_t offset );

/**
 * @brief Insert an item into a sorted list, keeping it sorted (without
 *        locking the list)
 *
 * @details add item before the first element of list that is strictly
 *  smaller, according to the integer value at offset in items. That
 *  is, if the input list is sorted (descending order), the resulting
 *  list is still sorted. The list is not locked during this operation.
 *
 * @param[inout] list the sorted list in which item should be inserted
 * @param[inout] item an item to insert in list
 * @param[in] offset the offset (in bytes) from the beginning of item
 *            in which an integer (sizeof(int) bytes) can be found.
 *            All items in list are assumed to have an integer at the
 *            same offset. Natural order is used to sort the items.
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_list_nolock_push_sorted( parsec_list_t* list,
                               parsec_list_item_t* item,
                               size_t offset );


/**
 * @brief Insert a set of items into a sorted list, keeping it sorted
 *
 * @details add each item in items before the first element of list
 *  that is strictly smaller, according to the integer value at offset
 *  in items. That is, if the input list is sorted (descending order),
 *  the resulting list is still sorted. The list is locked during
 *  this entire operation. items do not need to be sorted.
 *
 * @param[inout] list the sorted list in which the items should be inserted
 * @param[inout] items a list (unsorted) of items to insert in list
 * @param[in] offset the offset (in bytes) from the beginning of each item
 *            in which an integer (sizeof(int) bytes) can be found.
 *            All items in list are assumed to have an integer at the
 *            same offset. Natural order is used to sort the items.
 *
 * @remark this function is thread safe
 */
static inline void
parsec_list_chain_sorted( parsec_list_t* list,
                         parsec_list_item_t* items,
                         size_t offset );

/**
 * @brief Insert a set of items into a sorted list, keeping it sorted.
 *        The list is not locked during this operation.
 *
 * @details add each item in items before the first element of list
 *  that is strictly smaller, according to the integer value at offset
 *  in items. That is, if the input list is sorted (descending order),
 *  the resulting list is still sorted. The list is not locked during
 *  this entire operation. items do not need to be sorted.
 *
 * @param[inout] list the sorted list in which the items should be inserted
 * @param[inout] items a list (unsorted) of items to insert in list
 * @param[in] offset the offset (in bytes) from the beginning of each item
 *            in which an integer (sizeof(int) bytes) can be found.
 *            All items in list are assumed to have an integer at the
 *            same offset. Natural order is used to sort the items.
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_list_nolock_chain_sorted( parsec_list_t* list,
                                parsec_list_item_t* items,
                                size_t offset );


/**
 * @brief sort the list
 *
 * @details  All items in list are assumed to have an integer at the
 *           same offset. Natural order is used to sort the items.
 *           The list is locked during this operation.
 *
 * @param[inout] list an (unsorted) list of items to sort
 * @param[in] offset offset the offset (in bytes) from the beginning of each item
 *            in which an integer (sizeof(int) bytes) can be found.
 *            All items in list are assumed to have an integer at the
 *            same offset. Natural order is used to sort the items.
 *
 * @remark this function is thread safe
 */
static inline void
parsec_list_sort( parsec_list_t* list,
                 size_t offset );

/**
 * @brief sort the list without taking the lock
 *
 * @details  All items in list are assumed to have an integer at the
 *           same offset. Natural order is used to sort the items.
 *           The list is not locked during this operation.
 *
 * @param[inout] list an (unsorted) list of items to sort
 * @param[in] offset offset the offset (in bytes) from the beginning of each item
 *            in which an integer (sizeof(int) bytes) can be found.
 *            All items in list are assumed to have an integer at the
 *            same offset. Natural order is used to sort the items.
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_list_nolock_sort( parsec_list_t* list,
                        size_t offset );

/* DEQUEUE EMULATION FUNCTIONS */

/**
 * @brief Pop the head of the dequeue
 *
 * @details consider the list as a dequeue, and pop the head of the queue
 *
 * @param[inout] list the dequeue from which to pop the front element
 * @return the element that was removed from the dequeue (NULL if
 *         the dequeue was empty)
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline parsec_list_item_t*
parsec_list_pop_front( parsec_list_t* list );

/**
 * @brief Pop the tail of the dequeue
 *
 * @details consider the list as a dequeue, and pop the tail of the queue
 *
 * @param[inout] list the dequeue from which to pop the tail element
 * @return the element that was removed from the dequeue (NULL if
 *         the dequeue was empty)
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline parsec_list_item_t*
parsec_list_pop_back( parsec_list_t* list );

/**
 * @brief Try popping the head of the dequeue
 *
 * @details consider the list as a dequeue, and try popping its head.
 *
 * @param[inout] list the dequeue from which to pop the front element
 * @return the element, if one was removed from the dequeue (NULL if
 *         the dequeue was empty, or if another thread is currently
 *         holding a lock on the dequeue)
 *
 * @remark this function is thread safe
 * @remark this function will not wait if another thread is accessing
 *         the dequeue
 */
static inline parsec_list_item_t*
parsec_list_try_pop_front( parsec_list_t* list );

/**
 * @brief Try popping the tail of the dequeue
 *
 * @details consider the list as a dequeue, and try popping its tail.
 *
 * @param[inout] list the dequeue from which to pop the tail element
 * @return the element, if one was removed from the dequeue (NULL if
 *         the dequeue was empty, or if another thread is currently
 *         holding a lock on the dequeue)
 *
 * @remark this function is thread safe
 * @remark this function will not wait if another thread is accessing
 *         the dequeue
 */
static inline parsec_list_item_t*
parsec_list_try_pop_back( parsec_list_t* list );

/**
 * @brief Push an element at the front of the dequeue
 *
 * @details consider the list as a dequeue, and push an element at its front
 *
 * @param[inout] list the dequeue into which to push the element
 * @param[inout] item the element to push in the front
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline void
parsec_list_push_front( parsec_list_t* list,
                       parsec_list_item_t* item );
/**
 * @brief alias to parsec_list_push_front
 */
#define parsec_list_prepend parsec_list_push_front

/**
 * @brief Push an element at the end of the dequeue
 *
 * @details consider the list as a dequeue, and push an element at its end
 *
 * @param[inout] list the dequeue into which to push the element
 * @param[inout] item the element to push in the end
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline void
parsec_list_push_back( parsec_list_t* list,
                      parsec_list_item_t* item );
/**
 * @brief alias to parsec_list_push_front
 */
#define parsec_list_append parsec_list_push_back

/**
 * @brief Chain a ring of elements in front of a dequeue
 *
 * @details consider the list as a dequeue. Take a ring of elements
 *          (items->prev points to the last element in items),
 *          and push all the elements of items in front of the dequeue,
 *          preserving the order in items.
 *
 * @param[inout] list the dequeue into which to push the elements
 * @param[inout] items the elements ring to push in front
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline void
parsec_list_chain_front( parsec_list_t* list,
                        parsec_list_item_t* items );

/**
 * @brief Chain a ring of elements in the end of a dequeue
 *
 * @details consider the list as a dequeue. Take a ring of elements
 *          (items->prev points to the last element in items),
 *          and push all the elements of items in the back of the dequeue,
 *          preserving the order in items.
 *
 * @param[inout] list the dequeue into which to push the elements
 * @param[inout] items the elements ring to push in the back
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline void
parsec_list_chain_back( parsec_list_t* list,
                       parsec_list_item_t* items );

/**
 * @brief Extracts all elements from a dequeue, giving them as a ring
 *
 * @details consider the list as a dequeue. This function creates a
 *          double-linked ring of elements (first->prev points to last,
 *          and last->next points to first), that holds all elements
 *          of the dequeue. The dequeue is empty after this operation.
 *
 * @param[inout] list the dequeue from which to extract the elements
 * @return the ring of elements
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline parsec_list_item_t*
parsec_list_unchain( parsec_list_t* list );

/**
 * @brief Pop the head of the dequeue, without locking it
 *
 * @details consider the list as a dequeue, and pop the head of the queue
 *
 * @param[inout] list the dequeue from which to pop the front element
 * @return the element that was removed from the dequeue (NULL if
 *         the dequeue was empty)
 *
 * @remark this function is not thread safe
 */
static inline parsec_list_item_t*
parsec_list_nolock_pop_front( parsec_list_t* list );

/**
 * @brief Pop the tail of the dequeue, without lokcing it
 *
 * @details consider the list as a dequeue, and pop its tail.
 *
 * @param[inout] list the dequeue from which to pop the tail element
 * @return the element, if one was removed from the dequeue (NULL if
 *         the dequeue was empty)
 *
 * @remark this function is not thread safe
 */
static inline parsec_list_item_t*
parsec_list_nolock_pop_back( parsec_list_t* list );


/**
 * @brief Push an element at the end of the dequeue, without locking it
 *
 * @details consider the list as a dequeue, and push an element at its end
 *
 * @param[inout] list the dequeue into which to push the element
 * @param[inout] item the element to push in the end
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_list_nolock_push_front( parsec_list_t* list,
                              parsec_list_item_t* item );

/**
 * @brief Push an element at the end of the dequeue, without locking it
 *
 * @details consider the list as a dequeue, and push an element at its end
 *
 * @param[inout] list the dequeue into which to push the element
 * @param[inout] item the element to push in the end
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_list_nolock_push_back( parsec_list_t* list,
                             parsec_list_item_t* item );

/**
 * @brief Chain a ring of elements in front of a dequeue,
 *        without locking the dequeue
 *
 * @details consider the list as a dequeue. Take a ring of elements
 *          (items->prev points to the last element in items),
 *          and push all the elements of items in front of the dequeue,
 *          preserving the order in items.
 *
 * @param[inout] list the dequeue into which to push the elements
 * @param[inout] items the elements ring to push in front
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_list_nolock_chain_front( parsec_list_t* list,
                               parsec_list_item_t* items );

/**
 * @brief Chain a ring of elements in the end of a dequeue,
 *        without locking the dequeue
 *
 * @details consider the list as a dequeue. Take a ring of elements
 *          (items->prev points to the last element in items),
 *          and push all the elements of items in the back of the dequeue,
 *          preserving the order in items.
 *
 * @param[inout] list the dequeue into which to push the elements
 * @param[inout] items the elements ring to push in the back
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_list_nolock_chain_back( parsec_list_t* list,
                              parsec_list_item_t* items );

/**
 * @brief Extracts all elements from a dequeue, giving them as a ring,
 *        without locking the list
 *
 * @details consider the list as a dequeue. This function creates a
 *          double-linked ring of elements (first->prev points to last,
 *          and last->next points to first), that holds all elements
 *          of the dequeue. The dequeue is empty after this operation.
 *
 * @param[inout] list the dequeue from which to extract the elements
 * @return the ring of elements
 *
 * @remark this function is not thread safe
 */
static inline parsec_list_item_t*
parsec_list_nolock_unchain( parsec_list_t* list );


/**
 * @cond FALSE
 *  Don't include the implementation part in the doxygen documentation
 *********************************************************************
 *  Interface ends here, everything else is private
 */

#define _HEAD(LIST) ((LIST)->ghost_element.list_next)
#define _TAIL(LIST) ((LIST)->ghost_element.list_prev)
#define _GHOST(LIST) (&((list)->ghost_element))

#if defined(PARSEC_DEBUG_PARANOID)
#define _LIST_CHECK_CONSISTENCY(list)                                                 \
    assert( ((_HEAD(list) != _GHOST(list)) && (_TAIL(list) != _GHOST(list))) || \
            ((_HEAD(list) == _GHOST(list)) && (_TAIL(list) == _GHOST(list))) )
#else
#define _LIST_CHECK_CONSISTENCY(list)
#endif  /* defined(PARSEC_DEBUG_PARANOID) */

static inline int
parsec_list_nolock_is_empty( parsec_list_t* list )
{
    return _HEAD(list) == _GHOST(list);
}

static inline void
parsec_list_lock( parsec_list_t* list )
{
    parsec_atomic_lock(&list->atomic_lock);
    _LIST_CHECK_CONSISTENCY(list);
}

static inline void
parsec_list_unlock( parsec_list_t* list )
{
    _LIST_CHECK_CONSISTENCY(list);
    parsec_atomic_unlock(&list->atomic_lock);
}

static inline int
parsec_list_is_empty( parsec_list_t* list )
{
    int rc;
    parsec_list_lock(list);
    rc = parsec_list_nolock_is_empty(list);
    parsec_list_unlock(list);
    return rc;
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

#define _OPAQUE_LIST_NOLOCK_REV_ITERATOR_DEFINITION(list,ITEM,CODE) ({   \
    parsec_list_item_t* ITEM;                                            \
    for(ITEM = (parsec_list_item_t*)(list)->ghost_element.list_prev;     \
        ITEM != &((list)->ghost_element);                                \
        ITEM = (parsec_list_item_t*)ITEM->list_prev)                     \
    {                                                                    \
        CODE;                                                            \
    }                                                                    \
    ITEM;                                                                \
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


static inline void
parsec_list_add_after( parsec_list_t* list,
                             parsec_list_item_t* position,
                             parsec_list_item_t* newel )
{
    parsec_list_lock(list);
    parsec_list_nolock_add_after(list, position, newel);
    parsec_list_unlock(list);
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
    if (_HEAD(list) == _GHOST(list)) {
        parsec_list_nolock_push_front(list, newel);
    } else {
        /* take the range of priorities and decide whether to iterate forward or backward */
        int tail_val = COMPARISON_VAL(_TAIL(list), off);
        int head_val = COMPARISON_VAL(_HEAD(list), off);
        int comp_val = COMPARISON_VAL(newel, off);
        /* compute the pivot without risking overflow
         * first, we compute the half point from the head and tail
         * second, we account for odd numbers by adding 1 if both were odd */
        int pivot = (head_val/2) + (tail_val/2) + (((head_val%2) + (tail_val%2))) == 2 ? 1 : 0;
        if (comp_val > pivot) {
            /* new element is in upper half of priority range */
            parsec_list_item_t* position = PARSEC_LIST_NOLOCK_ITERATOR(list, pos,
            {
                if( A_HIGHER_PRIORITY_THAN_B(newel, pos, off) )
                    break;
            });
            parsec_list_nolock_add_before(list, position, newel);
        } else {
            /* new element is in lower half of priority range */
            parsec_list_item_t* position = PARSEC_LIST_NOLOCK_REV_ITERATOR(list, pos,
            {
                if( !A_HIGHER_PRIORITY_THAN_B(newel, pos, off) )
                    break;
            });
            parsec_list_nolock_add_after(list, position, newel);
        }
    }
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
    parsec_list_lock(list);
    item->list_next = _HEAD(list);
    _HEAD(list)->list_prev = item;
    _HEAD(list) = item;
    parsec_list_unlock(list);
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
    parsec_list_lock(list);
    tail->list_next = _HEAD(list);
    _HEAD(list)->list_prev = tail;
    _HEAD(list) = items;
    parsec_list_unlock(list);
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
    parsec_list_lock(list);
    item->list_prev = _TAIL(list);
    _TAIL(list)->list_next = item;
    _TAIL(list) = item;
    parsec_list_unlock(list);
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
    parsec_list_lock(list);
    items->list_prev = _TAIL(list);
    _TAIL(list)->list_next = items;
    _TAIL(list) = tail;
    parsec_list_unlock(list);
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

    parsec_list_lock(list);
    head = parsec_list_nolock_unchain(list);
    parsec_list_unlock(list);
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
    if( parsec_list_nolock_is_empty(list) ) {
        return NULL;
    }
    parsec_list_lock(list);
    parsec_list_item_t* item = parsec_list_nolock_pop_front(list);
    parsec_list_unlock(list);
    return item;
}

static inline parsec_list_item_t*
parsec_list_try_pop_front( parsec_list_t* list)
{
    if( parsec_list_nolock_is_empty(list) ) {
        return NULL;
    }
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
    if( parsec_list_nolock_is_empty(list) ) {
        return NULL;
    }
    parsec_list_lock(list);
    parsec_list_item_t* item = parsec_list_nolock_pop_back(list);
    parsec_list_unlock(list);
    return item;
}

static inline parsec_list_item_t*
parsec_list_try_pop_back( parsec_list_t* list)
{
    if( parsec_list_nolock_is_empty(list) ) {
        return NULL;
    }
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

END_C_DECLS

/**
 * @endcond
 * @}
 */

#endif  /* PARSEC_LIST_H_HAS_BEEN_INCLUDED */
