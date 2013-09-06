/*
 * Copyright (c) 2013      The University of Tennessee and The University
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

#include "dague_config.h"
#include "dague_hwloc.h"

typedef struct {
    dague_dequeue_t   *system_queue;
    dague_hbbuffer_t  *task_queue;
    int                nb_hierarch_queues;
    dague_hbbuffer_t **hierarch_queues;
} local_queues_scheduler_object_t;

#define LOCAL_QUEUES_OBJECT(eu_context) ((local_queues_scheduler_object_t*)(eu_context)->scheduler_object)

static void push_in_queue_wrapper(void *store, dague_list_item_t *elt)
{
    dague_dequeue_chain_back( (dague_dequeue_t*)store, elt );
}

#ifdef HAVE_HWLOC
/** In case of hierarchical bounded buffer, define
 *  the wrappers to functions
 */
static void push_in_buffer_wrapper(void *store, dague_list_item_t *elt)
{
    /* Store is a hbbbuffer */
    dague_hbbuffer_push_all( (dague_hbbuffer_t*)store, elt );
}
#endif

#endif /* _SCHED_LOCAL_QUEUES_UTILS_H */
