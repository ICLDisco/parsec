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
 * @file
 *   Define helper functions and types useful for different local queues based schedulers
 */

#ifndef _SCHED_LOCAL_QUEUES_UTILS_H
#define _SCHED_LOCAL_QUEUES_UTILS_H

#include "parsec/parsec_config.h"
#include "parsec/hbbuffer.h"

typedef struct {
    parsec_dequeue_t   *system_queue;               /* The overflow queue itself. */
    int                 local_system_queue_balance; /* A local sum of how many elements have been pushed / poped
                                                     * out of the system queue -- used for lockfree statistics and
                                                     * maintained by each algorithm in push / pop */
    parsec_hbbuffer_t  *task_queue;                 /* The lowest level bounded buffer, local to this thread only */
    int                 nb_hierarch_queues;         /* The number of bounded buffers -- algorithm dependent */
    parsec_hbbuffer_t **hierarch_queues;            /* The entire set of bounded buffers */
} local_queues_scheduler_object_t;

#define LOCAL_QUEUES_OBJECT(eu_context) ((local_queues_scheduler_object_t*)(eu_context)->scheduler_object)

static inline void push_in_queue_wrapper(void *store, parsec_list_item_t *elt, int32_t distance)
{
    (void)distance;
    parsec_dequeue_chain_back( (parsec_dequeue_t*)store, elt );
}

static long long int parsec_system_queue_length( parsec_vp_t *vp ) {
    long long int sum = 0;
    parsec_execution_stream_t *es;
    int thid;

    for(thid = 0; thid < vp->nb_cores; thid++) {
        es = vp->execution_streams[thid];
        sum += LOCAL_QUEUES_OBJECT( es )->local_system_queue_balance;
    }

    return sum;
}
    
static inline void push_in_system_queue_wrapper(void *sobj, parsec_list_item_t *elt, int32_t distance)
{
    int len = 0;
    local_queues_scheduler_object_t *obj = (local_queues_scheduler_object_t*)sobj;
    (void)distance;
    _LIST_ITEM_ITERATOR(elt, elt, item, {len++; });
    obj->local_system_queue_balance += len;
    parsec_dequeue_chain_back( obj->system_queue, elt );
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
