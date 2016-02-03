/**
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "dague_config.h"
#include "dague/dague_internal.h"
#include "dague/debug.h"
#include "dague/mca/sched/sched.h"
#include "dague/mca/sched/ip/sched_ip.h"
#include "dague/class/dequeue.h"
#include "dague/mca/pins/pins.h"
static int SYSTEM_NEIGHBOR = 0;

/**
 * Module functions
 */
static int sched_ip_install(dague_context_t* master);
static int sched_ip_schedule(dague_execution_unit_t* eu_context, dague_execution_context_t* new_context);
static dague_execution_context_t *sched_ip_select( dague_execution_unit_t *eu_context );
static int flow_ip_init(dague_execution_unit_t* eu_context, struct dague_barrier_t* barrier);
static void sched_ip_remove(dague_context_t* master);

const dague_sched_module_t dague_sched_ip_module = {
    &dague_sched_ip_component,
    {
        sched_ip_install,
        flow_ip_init,
        sched_ip_schedule,
        sched_ip_select,
        NULL,
        sched_ip_remove
    }
};

static int sched_ip_install( dague_context_t *master )
{
    SYSTEM_NEIGHBOR = master->nb_vp * master->virtual_processes[0]->nb_cores;
    return 0;
}

static int flow_ip_init(dague_execution_unit_t* eu_context, struct dague_barrier_t* barrier)
{
    dague_vp_t *vp = eu_context->virtual_process;

    if (eu_context == vp->execution_units[0])
        vp->execution_units[0]->scheduler_object = OBJ_NEW(dague_list_t);

    dague_barrier_wait(barrier);

    eu_context->scheduler_object = (void*)vp->execution_units[0]->scheduler_object;

    return 0;
}

static dague_execution_context_t *sched_ip_select( dague_execution_unit_t *eu_context )
{
    dague_execution_context_t * context =
        (dague_execution_context_t*)dague_list_pop_back((dague_list_t*)eu_context->scheduler_object);

#if defined(PINS_ENABLE)
    if (NULL != context)
        context->victim_core = SYSTEM_NEIGHBOR;
#endif
    return context;
}

static int sched_ip_schedule( dague_execution_unit_t* eu_context,
                              dague_execution_context_t* new_context )
{
#if defined(DAGUE_DEBUG_MOTORMOUTH)
    dague_list_item_t *it = (dague_list_item_t*)new_context;
    char tmp[MAX_TASK_STRLEN];
    do {
        DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "IP:\t Pushing task %s",
                dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, (dague_execution_context_t*)it));
        it = (dague_list_item_t*)((dague_list_item_t*)it)->list_next;
    } while( it != (dague_list_item_t*)new_context );
#endif
    dague_list_chain_sorted((dague_list_t*)eu_context->scheduler_object,
                            (dague_list_item_t*)new_context,
                            dague_execution_context_priority_comparator);
    return 0;
}

static void sched_ip_remove( dague_context_t *master )
{
    int p, t;
    dague_vp_t *vp;
    dague_execution_unit_t *eu;

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
