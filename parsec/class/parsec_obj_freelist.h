/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_OBJ_FREELIST_H
#define PARSEC_OBJ_FREELIST_H

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/class/parsec_object.h"
#include "parsec/class/lifo.h"

/**
 * @defgroup parsec_internal_classes_obj_freelist Object Free-Lists
 * @ingroup parsec_internal_classes
 * @{
 *
 *  @brief Recycle the storage of PaRSEC objects through a free-list.
 *
 *  @details A free-list is a parsec_obj_allocator_t that keeps the
 *  storage of released objects in a thread-safe LIFO and hands it out
 *  again on the next allocation, instead of going through malloc/free.
 *  Objects are constructed when allocated and destructed when released,
 *  so the usual object semantics are preserved.
 *  @code
 *    parsec_obj_freelist_t fl;
 *    PARSEC_OBJ_FREELIST_CONSTRUCT(&fl, sally_t);
 *
 *    sally_t *sally = PARSEC_OBJ_NEW(sally_t, &fl.super);
 *    PARSEC_OBJ_RETAIN(sally);
 *    PARSEC_OBJ_RELEASE(sally);
 *    PARSEC_OBJ_RELEASE(sally);  // storage goes back to fl
 *
 *    parsec_obj_freelist_destruct(&fl);
 *  @endcode
 *  Objects of any class whose size does not exceed the element size of
 *  the free-list may be allocated from it. Only objects allocated from
 *  the free-list can be returned to it: it cannot be passed to
 *  PARSEC_OBJ_CONSTRUCT for storage obtained elsewhere. All
 *  objects must have been released before the free-list is destructed.
 */

BEGIN_C_DECLS

typedef struct parsec_obj_freelist_s {
    parsec_obj_allocator_t super;     /**< allocator interface, must be first */
    parsec_lifo_t         *lifo;      /**< storage of released objects */
    size_t                 elt_size;  /**< size of each element */
    volatile int32_t       nb_elt;    /**< number of elements allocated (cached or in use) */
} parsec_obj_freelist_t;

/**
 * @brief Construct a free-list for objects of up to elt_size bytes.
 *
 * @param[out] freelist the free-list to construct
 * @param[in] elt_size the maximum size of objects allocated from this free-list
 * @return PARSEC_SUCCESS, or PARSEC_ERR_OUT_OF_RESOURCE if the free-list
 *         could not be set up
 */
PARSEC_DECLSPEC int parsec_obj_freelist_construct(parsec_obj_freelist_t *freelist, size_t elt_size);

/**
 * @brief Destruct a free-list and free all cached elements.
 *
 * @details All objects allocated from the free-list must have been
 * released before.
 *
 * @param[inout] freelist the free-list to destruct
 * @return the number of elements that were still in use (0 if all were released)
 */
PARSEC_DECLSPEC int32_t parsec_obj_freelist_destruct(parsec_obj_freelist_t *freelist);

/**
 * @brief Construct a free-list for objects of class type.
 */
#define PARSEC_OBJ_FREELIST_CONSTRUCT(freelist, type)                   \
    parsec_obj_freelist_construct((freelist), sizeof(type))

END_C_DECLS

/**
 * @}
 */

#endif  /* PARSEC_OBJ_FREELIST_H */
