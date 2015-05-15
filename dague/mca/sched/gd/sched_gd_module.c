/**
 * Copyright (c) 2013-2014 The University of Tennessee and The University
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
#include "dague_internal.h"
#include "dague/debug.h"
#include "dague/mca/sched/sched.h"
#include "dague/mca/sched/gd/sched_gd.h"
#include "dague/class/dequeue.h"
#include "dague/mca/pins/pins.h"

/**
 * Module functions
 */
static int sched_gd_install(dague_context_t* master);
static int sched_gd_schedule(dague_execution_unit_t* eu_context, dague_execution_context_t* new_context);
static dague_execution_context_t *sched_gd_select( dague_execution_unit_t *eu_context );
static int flow_gd_init(dague_execution_unit_t* eu_context, struct dague_barrier_t* barrier);
static void sched_gd_remove(dague_context_t* master);

const dague_sched_module_t dague_sched_gd_module = {
    &dague_sched_gd_component,
    {
        sched_gd_install,
        flow_gd_init,
        sched_gd_schedule,
        sched_gd_select,
        NULL,
        sched_gd_remove
    }
};

static int sched_gd_install( dague_context_t *master )
{
    (void)master;
    return 0;
}

static int flow_gd_init(dague_execution_unit_t* eu_context, struct dague_barrier_t* barrier)
{
    dague_vp_t *vp = eu_context->virtual_process;

    if (eu_context == vp->execution_units[0])
        vp->execution_units[0]->scheduler_object = OBJ_NEW(dague_dequeue_t);

    dague_barrier_wait(barrier);

    eu_context->scheduler_object = (void*)vp->execution_units[0]->scheduler_object;

    return 0;
}

static dague_execution_context_t *sched_gd_select( dague_execution_unit_t *eu_context )
{
    dague_execution_context_t * context =
        (dague_execution_context_t*)dague_dequeue_try_pop_front( (dague_dequeue_t*)eu_context->scheduler_object );
    return context;
}

static int sched_gd_schedule( dague_execution_unit_t* eu_context,
                              dague_execution_context_t* new_context )
{
#if defined(PINS_ENABLE)
    new_context->creator_core = 1;
#endif
    if( new_context->function->flags & DAGUE_HIGH_PRIORITY_TASK ) {
        dague_dequeue_chain_front( (dague_dequeue_t*)eu_context->scheduler_object, (dague_list_item_t*)new_context);
    } else {
        dague_dequeue_chain_back( (dague_dequeue_t*)eu_context->scheduler_object, (dague_list_item_t*)new_context);
    }
    return 0;
}

static void sched_gd_remove( dague_context_t *master )
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
