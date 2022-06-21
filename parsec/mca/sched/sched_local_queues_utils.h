/*
 * Copyright (c) 2013-2019 The University of Tennessee and The University
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
#if defined(PARSEC_PAPI_SDE)
    int                 local_system_queue_balance; /* A local sum of how many elements have been pushed / poped
                                                     * out of the system queue -- used for lockfree statistics and
                                                     * maintained by each algorithm in push / pop */
#endif
    parsec_hbbuffer_t  *task_queue;                 /* The lowest level bounded buffer, local to this thread only */
    int                 nb_hierarch_queues;         /* The number of bounded buffers -- algorithm dependent */
    parsec_hbbuffer_t **hierarch_queues;            /* The entire set of bounded buffers */
} parsec_mca_sched_local_queues_scheduler_object_t;

#define PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(eu_context) ((parsec_mca_sched_local_queues_scheduler_object_t*)(eu_context)->scheduler_object)

static inline void parsec_mca_sched_push_in_queue_wrapper(void *store, parsec_list_item_t *elt, int32_t distance)
{
    (void)distance;
    parsec_dequeue_chain_back( (parsec_dequeue_t*)store, elt );
}

static inline long long int parsec_mca_sched_system_queue_length( parsec_vp_t *vp ) {
#if defined(PARSEC_PAPI_SDE)
    long long int sum = 0;
    parsec_execution_stream_t *es;
    int thid;

    for(thid = 0; thid < vp->nb_cores; thid++) {
        es = vp->execution_streams[thid];
        sum += PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT( es )->local_system_queue_balance;
    }

    return sum;
#else
    (void)vp;
    return -1;
#endif
}

static inline parsec_task_t *parsec_mca_sched_pop_from_system_queue_wrapper(parsec_mca_sched_local_queues_scheduler_object_t *sched_obj)
{
    parsec_task_t *task = (parsec_task_t*)parsec_dequeue_try_pop_front(sched_obj->system_queue);
#if defined(PARSEC_PAPI_SDE)
    if( task != NULL )
        sched_obj->local_system_queue_balance--;
#endif
    return task;
}

static inline void parsec_mca_sched_push_in_system_queue_wrapper(void *sobj, parsec_list_item_t *elt, int32_t distance)
{
    parsec_mca_sched_local_queues_scheduler_object_t *obj = (parsec_mca_sched_local_queues_scheduler_object_t*)sobj;
#if defined(PARSEC_PAPI_SDE)
    int len = 0;
    _LIST_ITEM_ITERATOR(elt, elt, item, {len++; });
    obj->local_system_queue_balance += len;
#endif
    parsec_dequeue_chain_back( obj->system_queue, elt );
    (void)distance;
}

#ifdef PARSEC_HAVE_HWLOC
/** In case of hierarchical bounded buffer, define
 *  the wrappers to functions
 */
static inline void parsec_mca_sched_push_in_buffer_wrapper(void *store, parsec_list_item_t *elt, int32_t distance)
{
    /* Store is a hbbbuffer */
    parsec_hbbuffer_push_all( (parsec_hbbuffer_t*)store, elt, distance );
}
#endif

/**
 * List with local counter
 *
 * @details
 *   The local counter is used to keep an approximate length of the
 * queue while avoiding any lock / atomic operation. Each thread is
 * supposed to create such an object, that point to the same (thread-safe
 * list), but the local_counter is kept private.
 *
 * local_counter counts the number of elements this thread inserted or
 * removed from the list. It can thus be positive when the list is empty
 * (if the elements were removed by other threads), or negative (if that
 * thread removed more elements than it added).
 *
 * There are two implementations: one when SDE is enabled, the other, 
 * that removes the counter and the additional dereference if SDE is 
 * disabled
 */

#if defined(PARSEC_PAPI_SDE)
typedef struct {
    parsec_list_t *list;
    int            local_counter;
} parsec_mca_sched_list_local_counter_t;

static inline parsec_mca_sched_list_local_counter_t *parsec_mca_sched_allocate_list_local_counter(parsec_mca_sched_list_local_counter_t *msl)
{
    parsec_mca_sched_list_local_counter_t *sl = (parsec_mca_sched_list_local_counter_t*)malloc(sizeof(parsec_mca_sched_list_local_counter_t));
    if( NULL == msl ) {
        sl->list = PARSEC_OBJ_NEW(parsec_list_t);
    } else {
        sl->list = msl->list;
        PARSEC_OBJ_RETAIN(sl->list);
    }
    sl->local_counter = 0;
    return sl;
}

static inline void parsec_mca_sched_free_list_local_counter(parsec_mca_sched_list_local_counter_t *sl)
{
    PARSEC_OBJ_RELEASE(sl->list);
    free(sl);
}

static inline void parsec_mca_sched_list_local_counter_chain_sorted(parsec_mca_sched_list_local_counter_t *sl, parsec_task_t *it, size_t offset)
{
    int len = 0;
    _LIST_ITEM_ITERATOR(it, &it->super, item, {len++; });
    parsec_list_chain_sorted(sl->list, &it->super, offset);
    sl->local_counter += len;
}

static inline void parsec_mca_sched_list_local_counter_chain_back(parsec_mca_sched_list_local_counter_t *sl, parsec_task_t *it)
{
    int len = 0;
    _LIST_ITEM_ITERATOR(it, &it->super, item, {len++; });
    parsec_list_chain_back(sl->list, &it->super);
    sl->local_counter += len;
}

static inline parsec_task_t *parsec_mca_sched_list_local_counter_pop_front(parsec_mca_sched_list_local_counter_t *sl)
{
    parsec_task_t * context =
        (parsec_task_t*)parsec_list_pop_front(sl->list);
    if(NULL != context)
        sl->local_counter--;
    return context;
}

static inline parsec_task_t *parsec_mca_sched_list_local_counter_pop_back(parsec_mca_sched_list_local_counter_t *sl)
{
    parsec_task_t * context =
        (parsec_task_t*)parsec_list_pop_back(sl->list);
    if(NULL != context)
        sl->local_counter--;
    return context;
}

static inline long long int parsec_mca_sched_list_local_counter_length( parsec_vp_t *vp )
{
    int thid;
    long long int sum = 0;
    for(thid = 0; thid < vp->nb_cores; thid++) {
        sum += ((parsec_mca_sched_list_local_counter_t*)(vp->execution_streams[thid]))->local_counter;
    }
    return sum;
}

#else /* !defined(PARSEC_PAPI_SDE) */

typedef parsec_list_t parsec_mca_sched_list_local_counter_t;

static inline parsec_mca_sched_list_local_counter_t *parsec_mca_sched_allocate_list_local_counter(parsec_mca_sched_list_local_counter_t *list)
{
    if( NULL == list )
        return PARSEC_OBJ_NEW(parsec_list_t);
    PARSEC_OBJ_RETAIN(list);
    return list;
}

static inline void parsec_mca_sched_free_list_local_counter(parsec_mca_sched_list_local_counter_t *sl)
{
    PARSEC_OBJ_RELEASE(sl);
}

static inline void parsec_mca_sched_list_local_counter_chain_sorted(parsec_mca_sched_list_local_counter_t *sl, parsec_task_t *it, size_t offset)
{
    parsec_list_chain_sorted(sl, &it->super, offset);
}

static inline void parsec_mca_sched_list_local_counter_chain_back(parsec_mca_sched_list_local_counter_t *sl, parsec_task_t *it)
{
    parsec_list_chain_back(sl, &it->super);
}

static inline parsec_task_t *parsec_mca_sched_list_local_counter_pop_front(parsec_mca_sched_list_local_counter_t *sl)
{
    return (parsec_task_t*)parsec_list_pop_front(sl);
}

static inline parsec_task_t *parsec_mca_sched_list_local_counter_pop_back(parsec_mca_sched_list_local_counter_t *sl)
{
    return (parsec_task_t*)parsec_list_pop_back(sl);
}

static inline long long int parsec_mca_sched_list_local_counter_length( parsec_vp_t *vp )
{
    (void)vp;
    return -1;
}

#endif /* defined(PARSEC_PAPI_SDE) */


#endif /* _SCHED_LOCAL_QUEUES_UTILS_H */
