/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DEQUEUE_H_HAS_BEEN_INCLUDED
#define DEQUEUE_H_HAS_BEEN_INCLUDED

/* DEQUEUE definition.
 */

#include "parsec/parsec_config.h"
#include "parsec/class/list.h"

/**
 * @defgroup parsec_internal_classes_dequeue Dequeue
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief dequeue parsec_list_item_t management functions
 *
 *  @details Although the current implementation is a pure remap to
 * the list (see @ref parsec_internal_classes_list "list.h"), it is
 * not garanteed as such. If one needs to use both DEQUEUE and non
 * DEQUEUE access, list.h contains convenience functions to emulate a
 * dequeue, that is garanteed to be compatible with list accessors.
 */

BEGIN_C_DECLS

/**
 * @brief An (opaque) dequeue object
 */
typedef parsec_list_t parsec_dequeue_t;
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_dequeue_t);

/**
 * @brief check if dequeue is empty
 *
 * @param[inout] dequeue the dequeue to check
 * @return 0 if dequeue is not empty, 1 otherwise
 *
 * @remark this function is thread safe
 */
static inline int
parsec_dequeue_is_empty( parsec_dequeue_t* dequeue ) {
    return parsec_list_is_empty((parsec_list_t*)dequeue);
}

/**
 * @brief Pop the tail of the dequeue
 *
 * @details consider the list as a dequeue, and pop the tail of the queue
 *
 * @param[inout] dequeue the dequeue from which to pop the tail element
 * @return the element that was removed from the dequeue (NULL if
 *         the dequeue was empty)
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline parsec_list_item_t*
parsec_dequeue_pop_back( parsec_dequeue_t* dequeue ) {
    return parsec_list_pop_back((parsec_list_t*)dequeue);
}

/**
 * @brief Try poping the tail of the dequeue
 *
 * @details consider the list as a dequeue, and try poping its tail.
 *
 * @param[inout] deqeue the dequeue from which to pop the tail element
 * @return the element, if one was removed from the dequeue (NULL if
 *         the dequeue was empty, or if another thread is currently
 *         holding a lock on the dequeue)
 *
 * @remark this function is thread safe
 * @remark this function will not wait if another thread is accessing
 *         the dequeue
 */
static inline parsec_list_item_t*
parsec_dequeue_try_pop_back( parsec_dequeue_t* dequeue ) {
    return parsec_list_try_pop_back((parsec_list_t*)dequeue);
}

/**
 * @brief Pop the head of the dequeue
 *
 * @details consider the list as a dequeue, and pop the head of the queue
 *
 * @param[inout] dequeue the dequeue from which to pop the front element
 * @return the element that was removed from the dequeue (NULL if
 *         the dequeue was empty)
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline parsec_list_item_t*
parsec_dequeue_pop_front( parsec_dequeue_t* dequeue ) {
    return parsec_list_pop_front((parsec_list_t*)dequeue);
}

/**
 * @brief Try poping the head of the dequeue
 *
 * @details consider the list as a dequeue, and try poping its head.
 *
 * @param[inout] deqeue the dequeue from which to pop the front element
 * @return the element, if one was removed from the dequeue (NULL if
 *         the dequeue was empty, or if another thread is currently
 *         holding a lock on the dequeue)
 *
 * @remark this function is thread safe
 * @remark this function will not wait if another thread is accessing
 *         the dequeue
 */
static inline parsec_list_item_t*
parsec_dequeue_try_pop_front( parsec_dequeue_t* dequeue ) {
    return parsec_list_try_pop_front((parsec_list_t*)dequeue);
}

/**
 * @brief Push an element at the end of the dequeue
 *
 * @details consider the list as a dequeue, and push an element at its end
 *
 * @param[inout] deqeue the dequeue into which to push the element
 * @param[inout] item the element to push in the end
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline void
parsec_dequeue_push_back( parsec_dequeue_t* dequeue, parsec_list_item_t* item ) {
    parsec_list_push_back((parsec_list_t*)dequeue, item);
}

/**
 * @brief Push an element at the front of the dequeue
 *
 * @details consider the list as a dequeue, and push an element at its front
 *
 * @param[inout] deqeue the dequeue into which to push the element
 * @param[inout] item the element to push in the front
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline void
parsec_dequeue_push_front( parsec_dequeue_t* dequeue, parsec_list_item_t* item ) {
    parsec_list_push_front((parsec_list_t*)dequeue, item);
}

/**
 * @brief Chain a ring of elements in front of a dequeue
 *
 * @details consider the list as a dequeue. Take a ring of elements
 *          (items->prev points to the last element in items),
 *          and push all the elements of items in front of the dequeue,
 *          preserving the order in items.
 *
 * @param[inout] deqeue the dequeue into which to push the elements
 * @param[inout] items the elements ring to push in front
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline void
parsec_dequeue_chain_front( parsec_dequeue_t* dequeue, parsec_list_item_t* items ) {
    parsec_list_chain_front((parsec_list_t*)dequeue, items);
}

/**
 * @brief Chain a ring of elements in the end of a dequeue
 *
 * @details consider the list as a dequeue. Take a ring of elements
 *          (items->prev points to the last element in items),
 *          and push all the elements of items in the back of the dequeue,
 *          preserving the order in items.
 *
 * @param[inout] deqeue the dequeue into which to push the elements
 * @param[inout] items the elements ring to push in the back
 *
 * @remark this function is thread safe
 * @remark this function might lock until no other thread manipulates the
 *         dequeue
 */
static inline void
parsec_dequeue_chain_back( parsec_dequeue_t* dequeue, parsec_list_item_t* items ) {
    parsec_list_chain_back((parsec_list_t*)dequeue, items);
}

/**
 * @brief check if list is empty, ignoring the lock
 *
 * @param[in] deqeue the deqeue to check
 * @return 0 if the deqeue is not empty, 1 otherwise
 *
 * @remark this function is not thread safe
 */
static inline int
parsec_dequeue_nolock_is_empty( parsec_dequeue_t* dequeue) {
    return parsec_list_nolock_is_empty((parsec_list_t*)dequeue);
}

/**
 * @brief alias to parsec_dequeue_nolock_is_empty
 */
#define parsec_udequeue_is_empty(dequeue) parsec_dequeue_nolock_is_empty(dequeue)

/**
 * @brief Pop the head of the dequeue, without locking it
 *
 * @details consider the list as a dequeue, and pop the head of the queue
 *
 * @param[inout] deqeue the dequeue from which to pop the front element
 * @return the element that was removed from the dequeue (NULL if
 *         the dequeue was empty)
 *
 * @remark this function is not thread safe
 */
static inline parsec_list_item_t*
parsec_dequeue_nolock_pop_front( parsec_dequeue_t* dequeue ) {
    return parsec_list_nolock_pop_front((parsec_list_t*)dequeue);
}

/**
 * @brief alias to parsec_dequeue_nolock_pop_front
 */
#define parsec_udequeue_pop_front(dequeue) parsec_dequeue_nolock_pop_front(dequeue)

/**
 * @brief Pop the tail of the dequeue, without lokcing it
 *
 * @details consider the list as a dequeue, and pop its tail.
 *
 * @param[inout] deqeue the dequeue from which to pop the tail element
 * @return the element, if one was removed from the dequeue (NULL if
 *         the dequeue was empty)
 *
 * @remark this function is not thread safe
 */
static inline parsec_list_item_t*
parsec_dequeue_nolock_pop_back( parsec_dequeue_t* dequeue ) {
    return parsec_list_nolock_pop_back((parsec_list_t*)dequeue);
}

/**
 * @brief alias to parsec_dequeue_nolock_pop_back
 */
#define parsec_udequeue_pop_back(dequeue) parsec_dequeue_nolock_pop_back(dequeue)

/**
 * @brief Push an element at the end of the dequeue, without locking it
 *
 * @details consider the list as a dequeue, and push an element at its end
 *
 * @param[inout] deqeue the dequeue into which to push the element
 * @param[inout] item the element to push in the end
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_dequeue_nolock_push_front( parsec_dequeue_t* dequeue, parsec_list_item_t* item ) {
    parsec_list_nolock_push_front((parsec_list_t*)dequeue, item);
}

/**
 * @brief alias to parsec_dequeue_nolock_push_front
 */
#define parsec_udequeue_push_front(dequeue, item) parsec_dequeue_nolock_push_front(dequeue, item)

/**
 * @brief Push an element at the end of the dequeue, without locking it
 *
 * @details consider the list as a dequeue, and push an element at its end
 *
 * @param[inout] deqeue the dequeue into which to push the element
 * @param[inout] item the element to push in the end
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_dequeue_nolock_push_back( parsec_dequeue_t* dequeue, parsec_list_item_t* item ) {
    parsec_list_nolock_push_back((parsec_list_t*)dequeue, item);
}

/**
 * @brief alias to parsec_dequeue_nolock_push_back
 */
#define parsec_udequeue_push_back(dequeue, item) parsec_dequeue_nolock_push_back(dequeue, item)

/**
 * @brief Chain a ring of elements in front of a dequeue,
 *        without locking the dequeue
 *
 * @details consider the list as a dequeue. Take a ring of elements
 *          (items->prev points to the last element in items),
 *          and push all the elements of items in front of the dequeue,
 *          preserving the order in items.
 *
 * @param[inout] deqeue the dequeue into which to push the elements
 * @param[inout] items the elements ring to push in front
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_dequeue_nolock_chain_front( parsec_dequeue_t* dequeue, parsec_list_item_t* items ) {
    parsec_list_nolock_chain_front((parsec_list_t*)dequeue, items);
}

/**
 * @brief alias to parsec_dequeue_nolock_chain_front
 */
#define parsec_udequeue_chainf(dequeue, items) parsec_dequeue_nolock_chain_front(dequeue, items)

/**
 * @brief Chain a ring of elements in the end of a dequeue,
 *        without locking the dequeue
 *
 * @details consider the list as a dequeue. Take a ring of elements
 *          (items->prev points to the last element in items),
 *          and push all the elements of items in the back of the dequeue,
 *          preserving the order in items.
 *
 * @param[inout] deqeue the dequeue into which to push the elements
 * @param[inout] items the elements ring to push in the back
 *
 * @remark this function is not thread safe
 */
static inline void
parsec_dequeue_nolock_chain_back( parsec_dequeue_t* dequeue, parsec_list_item_t* items ) {
    parsec_list_nolock_chain_back((parsec_list_t*)dequeue, items);
}

/**
 * @brief alias to parsec_dequeue_nolock_chain_back
 */
#define parsec_udequeue_chain_back(dequeue, items) parsec_dequeue_nolock_chain_back(dequeue, items)

END_C_DECLS

/** @} */

#endif  /* DEQUEUE_H_HAS_BEEN_INCLUDED */
