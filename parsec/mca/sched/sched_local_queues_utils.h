/*
 * Copyright (c) 2013-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

/**
 * @FILE
 *   Define helper functions and types useful for different local queues based schedulers
 */

#ifndef _SCHED_LOCAL_QUEUES_UTILS_H
#define _SCHED_LOCAL_QUEUES_UTILS_H

#include "parsec/parsec_config.h"

typedef struct {
    parsec_dequeue_t   *system_queue;
    parsec_hbbuffer_t  *task_queue;
    int                nb_hierarch_queues;
    parsec_hbbuffer_t **hierarch_queues;
} local_queues_scheduler_object_t;

#define LOCAL_QUEUES_OBJECT(eu_context) ((local_queues_scheduler_object_t*)(eu_context)->scheduler_object)

static inline void push_in_queue_wrapper(void *store, parsec_list_item_t *elt, int32_t distance)
{
    (void)distance;
    parsec_dequeue_chain_back( (parsec_dequeue_t*)store, elt );
}

#ifdef PARSEC_HAVE_HWLOC
/** In case of hierarchical bounded buffer, define
 *  the wrappers to functions
 */
static inline void push_in_buffer_wrapper(void *store, parsec_list_item_t *elt, int32_t distance)
{
    /* Store is a hbbbuffer */
    parsec_hbbuffer_push_all( (parsec_hbbuffer_t*)store, elt, distance );
}
#endif

#endif /* _SCHED_LOCAL_QUEUES_UTILS_H */
