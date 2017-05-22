/**
 * Copyright (c) 2017      The University of Tennessee and The University
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
#include "parsec/debug.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/spq/sched_spq.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"

/**
 * Module functions
 */
static int sched_spq_install(parsec_context_t* master);
static int sched_spq_schedule(parsec_execution_unit_t* eu_context,
                             parsec_task_t* new_context,
                             int32_t distance);
static parsec_task_t*
sched_spq_select(parsec_execution_unit_t *eu_context,
                int32_t* distance);
static int flow_spq_init(parsec_execution_unit_t* eu_context, struct parsec_barrier_t* barrier);
static void sched_spq_remove(parsec_context_t* master);

typedef struct parsec_spq_priority_list_s {
    parsec_list_item_t super;
    parsec_list_t      tasks;
    int                prio;
} parsec_spq_priority_list_t;
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_spq_priority_list_t);
OBJ_CLASS_INSTANCE(parsec_spq_priority_list_t, parsec_list_item_t,
                   NULL, NULL);

const parsec_sched_module_t parsec_sched_spq_module = {
    &parsec_sched_spq_component,
    {
        sched_spq_install,
        flow_spq_init,
        sched_spq_schedule,
        sched_spq_select,
        NULL,
        sched_spq_remove
    }
};

/* Example Starts Here */

static int sched_spq_install( parsec_context_t *master )
{
    (void)master;
    return 0;
}

static int flow_spq_init(parsec_execution_unit_t* eu_context, struct parsec_barrier_t* barrier)
{
    parsec_vp_t *vp = eu_context->virtual_process;

    if (eu_context == vp->execution_units[0])
        vp->execution_units[0]->scheduler_object = OBJ_NEW(parsec_list_t);

    parsec_barrier_wait(barrier);

    eu_context->scheduler_object = (void*)vp->execution_units[0]->scheduler_object;

    return 0;
}

static parsec_task_t* sched_spq_select(parsec_execution_unit_t *eu_context,
                                                    int32_t* distance)
{
    parsec_task_t* context;
    parsec_list_item_t *li;
    parsec_spq_priority_list_t *plist;
    parsec_list_t *task_list = (parsec_list_t*)eu_context->scheduler_object;

    parsec_list_lock(task_list);
    for( li = PARSEC_LIST_ITERATOR_FIRST(task_list);
         li != PARSEC_LIST_ITERATOR_END(task_list);
         li = PARSEC_LIST_ITERATOR_NEXT(li) ) {
        plist = (parsec_spq_priority_list_t*)li;
        if( (context = (parsec_task_t*)parsec_list_pop_front(&plist->tasks)) != NULL ) {
            *distance = plist->prio;
            break;
        }
    }
    parsec_list_unlock(task_list);
    return context;
}

static int sched_spq_schedule(parsec_execution_unit_t* eu_context,
                             parsec_task_t* new_context,
                             int32_t distance)
{
    parsec_list_item_t *li;
    int new_prio;
    parsec_spq_priority_list_t *plist;
    parsec_list_t *task_list = (parsec_list_t*)eu_context->scheduler_object;

    new_prio = 1;
    parsec_list_lock(task_list);
    li = PARSEC_LIST_ITERATOR_FIRST(task_list);
    while( li != PARSEC_LIST_ITERATOR_END(task_list) ) {
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
        parsec_list_nolock_add_before(task_list, li, &plist->super);
    }                                          
    parsec_list_chain_sorted(&plist->tasks,
                             (parsec_list_item_t*)new_context,
                             parsec_execution_context_priority_comparator);
    parsec_list_unlock(task_list);
    return 0;
}

static void sched_spq_remove( parsec_context_t *master )
{
    int p, t;
    parsec_vp_t *vp;
    parsec_execution_unit_t *eu;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            if( eu->th_id == 0 ) {
                OBJ_DESTRUCT( eu->scheduler_object );
                free(eu->scheduler_object);
            }
            eu->scheduler_object = NULL;
        }
    }
}
