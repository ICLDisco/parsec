/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef HBBUFFER_H_HAS_BEEN_INCLUDED
#define HBBUFFER_H_HAS_BEEN_INCLUDED

/** @addtogroup parsec_internal_scheduling
 *  @{ */

#include"parsec/parsec_config.h"
#include "parsec/class/list_item.h"

BEGIN_C_DECLS

typedef struct parsec_hbbuffer_s parsec_hbbuffer_t;

/**
 * Hierarchical Bounded Buffers:
 *
 *   bounded buffers with a parent storage, to store elements
 *   that will be ejected from the current buffer at push time.
 */

/**
 * Generic push function: takes a pointer to a store object, a pointer to a
 * ring of items (usually tasks), and a distance where to start inserting
 * the elements. The store object can be of any type, it will internally be
 * able to cast itself to the right type. The provided elements should be
 * stored before this functions returns.
 */
typedef void (*parsec_hbbuffer_parent_push_fct_t)(void *store,
                                                  parsec_list_item_t *elt,
                                                  int32_t distance);

struct parsec_hbbuffer_s {
    size_t size;       /**< the size of the buffer, in number of void* */
    size_t ideal_fill; /**< hint on the number of elements that should be there to increase parallelism */
    unsigned int assoc_core_num; // only exists for scheduler instrumentation
    void    *parent_store; /**< pointer to this buffer parent store */
    /** function to push element to the parent store */
    parsec_hbbuffer_parent_push_fct_t parent_push_fct;
    volatile parsec_list_item_t *items[1]; /**< array of elements */
};

parsec_hbbuffer_t*
parsec_hbbuffer_new(size_t size,  size_t ideal_fill,
                    parsec_hbbuffer_parent_push_fct_t parent_push_fct,
                    void *parent_store);

void parsec_hbbuffer_destruct(parsec_hbbuffer_t *b);

void
parsec_hbbuffer_push_all(parsec_hbbuffer_t *b,
                         parsec_list_item_t *elt,
                         int32_t distance);

void
parsec_hbbuffer_push_all_by_priority(parsec_hbbuffer_t *b,
                                     parsec_list_item_t *list,
                                     int32_t distance);

/* This code is unsafe, since another thread may be inserting new elements.
 * Use is_empty in safe-checking only
 */
static inline int
parsec_hbbuffer_is_empty(parsec_hbbuffer_t *b)
{
    unsigned int i;
    for(i = 0; i < b->size; i++)
        if( NULL != b->items[i] )
            return 0;
    return 1;
}

parsec_list_item_t*
parsec_hbbuffer_pop_best(parsec_hbbuffer_t *b, off_t priority_offset);

END_C_DECLS

/** @} */

#endif /* HBBUFFER_H_HAS_BEEN_INCLUDED */
