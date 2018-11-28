/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef FIFO_H_HAS_BEEN_INCLUDED
#define FIFO_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/class/list.h"

/**
 * @defgroup parsec_internal_classes_fifo First In First Out
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief First In First out parsec_list_item_t management functions
 *
 *  @details Although the current implementation is a pure remap to
 *     the list (see @ref parsec_internal_classes_list "list.h"), it
 *     is not garanteed as such. If you need to use both FIFO and non
 *     FIFO access, list.h contains convenience functions to emulate a
 *     fifo that is garanteed to be compatible with list accessors.
 */

BEGIN_C_DECLS

/**
 * @brief A fifo object
 */
typedef parsec_list_t parsec_fifo_t;
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_fifo_t);

/**
 * @brief tests if the FIFO is empty
 *
 * @param[inout] fifo the FIFO to test
 * @return 0 if fifo is not empty, 1 otherwise
 *
 * @remark this is a thread safe operation
 */
static inline int
parsec_fifo_is_empty( parsec_fifo_t* fifo ) {
    return parsec_list_is_empty((parsec_list_t*)fifo);
}

/**
 * @brief test if the FIFO is empty without taking the lock on it
 *
 * @param[in] fifo the FIFO to test
 * @return 0 if fifo is not empty, 1 otherwise
 *
 * @remark this is not a thread safe operation
 */
static inline int
parsec_fifo_nolock_is_empty( parsec_fifo_t* fifo)
{
    return parsec_list_nolock_is_empty((parsec_list_t*)fifo);
}

/**
 * @brief Push an element in the FIFO
 *
 * @param[inout] fifo the FIFO in which item should be pushed
 * @param[inout] item the element to add to fifo
 *
 * @remark this is a thread safe operation
 */
static inline void
parsec_fifo_push(parsec_fifo_t* fifo, parsec_list_item_t* item) {
    parsec_list_push_front((parsec_list_t*)fifo, item);
}

/**
 * @brief Push an element in the FIFO without checking the lock on it
 *
 * @param[inout] fifo the FIFO in which item should be pushed
 * @param[inout] item the element to add to fifo
 *
 * @remark this is not a thread safe operation
 */
static inline void
parsec_fifo_nolock_push(parsec_fifo_t* fifo, parsec_list_item_t* item) {
    parsec_list_nolock_push_front((parsec_list_t*)fifo, item);
}

/**
 * @brief Chain a ring of elements at the end of a FIFO
 *
 * @details items is a ring of elements, ordered. They are all pushed at
 *     the end of the FIFO, preserving the order between them (as if a
 *     call to push had been issued for each item in the ring).
 *
 * @param[inout] fifo the FIFO to which the ring of items should be pushed
 * @param[inout] items a ring of elements add to fifo
 *
 * @remark this is a thread safe operation
 */
static inline void
parsec_fifo_chain(parsec_fifo_t* fifo, parsec_list_item_t* items) {
    parsec_list_chain_front((parsec_list_t*)fifo, items);
}

/**
 * @brief Chain a ring of elements at the end of a FIFO without
 *        taking the lock
 *
 * @details items is a ring of elements, ordered. They are all pushed at
 *     the end of the FIFO, preserving the order between them (as if a
 *     call to push had been issued for each item in the ring).
 *
 * @param[inout] fifo the FIFO to which the ring of items should be pushed
 * @param[inout] items a ring of elements add to fifo
 *
 * @remark this is not a thread safe operation
 */
static inline void
parsec_fifo_nolock_chain(parsec_fifo_t* fifo, parsec_list_item_t* items) {
    parsec_list_nolock_chain_front((parsec_list_t*)fifo, items);
}

/**
 * @brief Extracts the first element in a FIFO
 *
 * @details this function may wait until other threads are done
 *    pushing or extracting elements from the FIFO
 *
 * @param[inout] fifo the FIFO from which to extract the first element
 * @return NULL if fifo is empty, the first element that was removed from
 *        fifo otherwise.
 *
 * @remark this is a thread safe operation
 */
static inline parsec_list_item_t*
parsec_fifo_pop(parsec_fifo_t* fifo) {
    return parsec_list_pop_front((parsec_list_t*)fifo);
}

/**
 * @brief Try to extract the first element in a FIFO
 *
 * @details this function will not wait if other threads are done
 *    pushing or extracting elements from the FIFO
 *
 * @param[inout] fifo the FIFO from which to try to extract the first element
 * @return NULL if fifo is empty, or if another thread is holding the lock
 *        on fifo. The first element that was removed from fifo otherwise.
 *
 * @remark this is a thread safe operation
 */
static inline parsec_list_item_t*
parsec_fifo_try_pop(parsec_fifo_t* fifo) {
    return parsec_list_try_pop_front((parsec_list_t*)fifo);
}

/**
 * @brief Extracts the first element in a FIFO
 *
 * @param[inout] fifo the FIFO from which to extract the first element
 * @return NULL if fifo is empty, the first element that was removed from
 *        fifo otherwise.
 *
 * @remark this is not a thread safe operation
 */
static inline parsec_list_item_t*
parsec_fifo_nolock_pop(parsec_fifo_t* fifo) {
    return parsec_list_nolock_pop_front((parsec_list_t*)fifo);
}

END_C_DECLS

/** @} */

#endif  /* FIFO_H_HAS_BEEN_INCLUDED */
