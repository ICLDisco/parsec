/**
 * Copyright (c) 2017-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/spq/sched_spq.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"

/**
 * Module functions
 */
static int sched_spq_install(parsec_context_t* master);
static int sched_spq_schedule(parsec_execution_stream_t* es,
                             parsec_task_t* new_context,
                             int32_t distance);
static parsec_task_t*
sched_spq_select(parsec_execution_stream_t *es,
                int32_t* distance);
static int flow_spq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);
static void sched_spq_remove(parsec_context_t* master);
static void sched_spq_register_sde( parsec_execution_stream_t *es );

typedef struct parsec_spq_priority_list_s {
    parsec_list_item_t super;
    parsec_list_t      tasks;
    int                prio;
} parsec_spq_priority_list_t;
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_spq_priority_list_t);

static inline void parsec_spq_priority_list_construct( parsec_spq_priority_list_t*plist );
static inline void parsec_spq_priority_list_destruct( parsec_spq_priority_list_t *plist );

OBJ_CLASS_INSTANCE(parsec_spq_priority_list_t, parsec_list_item_t,
                   parsec_spq_priority_list_construct, parsec_spq_priority_list_destruct);

static inline void parsec_spq_priority_list_construct( parsec_spq_priority_list_t*plist )
{
    OBJ_CONSTRUCT(&plist->tasks, parsec_list_t);
    plist->prio = -1;
}

static inline void parsec_spq_priority_list_destruct( parsec_spq_priority_list_t*plist )
{
    OBJ_DESTRUCT(&plist->tasks);
}

/* Since we're locking the list for all operations anyway,
 * we use the lock to protect the long long int size for updates; 
 * PAPI will read size without locking, which is fine as it is
 * only an approximation of the number of tasks */
typedef struct {
    parsec_list_t super;
    long long int size;
} parsec_list_with_size_t;

PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_list_with_size_t);

static inline void parsec_list_with_size_construct( parsec_list_with_size_t*plist );

OBJ_CLASS_INSTANCE(parsec_list_with_size_t, parsec_list_t,
                   parsec_list_with_size_construct, NULL);

static inline void parsec_list_with_size_construct( parsec_list_with_size_t*plist )
{
    plist->size = 0;
}

const parsec_sched_module_t parsec_sched_spq_module = {
    &parsec_sched_spq_component,
    {
        sched_spq_install,
        flow_spq_init,
        sched_spq_schedule,
        sched_spq_select,
        NULL,
        sched_spq_register_sde,
        sched_spq_remove
    }
};

/* Example Starts Here */

static int sched_spq_install( parsec_context_t *master )
{
    (void)master;
    return 0;
}

static void sched_spq_register_sde( parsec_execution_stream_t *es )
{
    char event_name[256];
    if( NULL != es && 0 == es->th_id ) {
        snprintf(event_name, 256, "PARSEC::SCHEDULER::PENDING_TASKS::QUEUE=%d::SCHED=SPQ", es->virtual_process->vp_id);
        papi_sde_register_counter(parsec_papi_sde_handle, event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,PAPI_SDE_int,
                                  &((parsec_list_with_size_t*)es->scheduler_object)->size);
        papi_sde_add_counter_to_group(parsec_papi_sde_handle, event_name,
                                      "PARSEC::SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
        papi_sde_add_counter_to_group(parsec_papi_sde_handle, event_name,
                                      "PARSEC::SCHEDULER::PENDING_TASKS::SCHED=SPQ", PAPI_SDE_SUM);
    }
    if( NULL == es || 0 == es->th_id ) {
        papi_sde_describe_counter(parsec_papi_sde_handle, "PARSEC::SCHEDULER::PENDING_TASKS::SCHED=SPQ",
                                  "the number of pending tasks for the SPQ scheduler");
        papi_sde_describe_counter(parsec_papi_sde_handle, "PARSEC::SCHEDULER::PENDING_TASKS::QUEUE=<VPID>::SCHED=SPQ",
                                  "the number of pending tasks for the SPQ scheduler on virtual process <VPID>");
    }

}

static int flow_spq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier)
{
    parsec_vp_t *vp = es->virtual_process;

    if (es == vp->execution_streams[0])
        vp->execution_streams[0]->scheduler_object = OBJ_NEW(parsec_list_with_size_t);

    parsec_barrier_wait(barrier);

    es->scheduler_object = (void*)vp->execution_streams[0]->scheduler_object;

    sched_spq_register_sde( es );

    return 0;
}

static parsec_task_t* sched_spq_select(parsec_execution_stream_t *es,
                                       int32_t* distance)
{
    parsec_task_t* context;
    parsec_list_item_t *li;
    parsec_spq_priority_list_t *plist;
    parsec_list_with_size_t *task_list = (parsec_list_with_size_t*)es->scheduler_object;

    parsec_list_lock(&task_list->super);
    for( li = PARSEC_LIST_ITERATOR_FIRST(&task_list->super);
         li != PARSEC_LIST_ITERATOR_END(&task_list->super);
         li = PARSEC_LIST_ITERATOR_NEXT(li) ) {
        plist = (parsec_spq_priority_list_t*)li;
        if( (context = (parsec_task_t*)parsec_list_pop_front(&plist->tasks)) != NULL ) {
            *distance = plist->prio;
            task_list->size--;
            break;
        }
    }
    parsec_list_unlock(&task_list->super);
    return context;
}

static int sched_spq_schedule(parsec_execution_stream_t* es,
                             parsec_task_t* new_context,
                             int32_t distance)
{
    parsec_list_item_t *li;
    int len;
    int new_prio;
    parsec_spq_priority_list_t *plist;
    parsec_list_with_size_t *task_list = (parsec_list_with_size_t*)es->scheduler_object;

    len = 0;
    _LIST_ITEM_ITERATOR(new_context, &new_context->super, item, {len++; });

    new_prio = 1;
    parsec_list_lock(&task_list->super);
    li = PARSEC_LIST_ITERATOR_FIRST(&task_list->super);
    while( li != PARSEC_LIST_ITERATOR_END(&task_list->super) ) {
        plist = (parsec_spq_priority_list_t*)li;
        if( plist->prio == distance ) {
            new_prio = 0;
            break;
        }
        if( plist->prio > distance ) {
            break;
        }
        li = PARSEC_LIST_ITERATOR_NEXT(li);
    }
    if( new_prio ) {
        plist = OBJ_NEW(parsec_spq_priority_list_t);
        plist->prio = distance;
        parsec_list_nolock_add_before(&task_list->super, li, &plist->super);
    }
    task_list->size += len;
    parsec_list_chain_sorted(&plist->tasks,
                             (parsec_list_item_t*)new_context,
                             parsec_execution_context_priority_comparator);
    parsec_list_unlock(&task_list->super);
    return 0;
}

static void sched_spq_remove( parsec_context_t *master )
{
    int p, t;
    parsec_vp_t *vp;
    parsec_execution_stream_t *eu;
    parsec_list_item_t *li;
    parsec_list_with_size_t *plist;
    char event_name[256];

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_streams[t];
            if( eu->th_id == 0 ) {
                plist = (parsec_list_with_size_t *)eu->scheduler_object;
                while( (li = parsec_list_pop_front(&plist->super)) != NULL )
                    OBJ_RELEASE(li);
                OBJ_RELEASE( plist );
            }
            eu->scheduler_object = NULL;
        }
        snprintf(event_name, 256, "PARSEC::SCHEDULER::PENDING_TASKS::QUEUE=%d::SCHED=SPQ", p);
        papi_sde_unregister_counter(parsec_papi_sde_handle, event_name);
    }
    papi_sde_unregister_counter(parsec_papi_sde_handle, "PARSEC::SCHEDULER::PENDING_TASKS::SCHED=SPQ");
}
