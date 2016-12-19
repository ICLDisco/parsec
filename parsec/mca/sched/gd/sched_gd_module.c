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

#include "parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/debug.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/gd/sched_gd.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"

/**
 * Module functions
 */
static int sched_gd_install(parsec_context_t* master);
static int sched_gd_schedule(parsec_execution_unit_t* eu_context, parsec_execution_context_t* new_context);
static parsec_execution_context_t *sched_gd_select( parsec_execution_unit_t *eu_context );
static int flow_gd_init(parsec_execution_unit_t* eu_context, struct parsec_barrier_t* barrier);
static void sched_gd_remove(parsec_context_t* master);

const parsec_sched_module_t parsec_sched_gd_module = {
    &parsec_sched_gd_component,
    {
        sched_gd_install,
        flow_gd_init,
        sched_gd_schedule,
        sched_gd_select,
        NULL,
        sched_gd_remove
    }
};

static int sched_gd_install( parsec_context_t *master )
{
    (void)master;
    return 0;
}

static int flow_gd_init(parsec_execution_unit_t* eu_context, struct parsec_barrier_t* barrier)
{
    parsec_vp_t *vp = eu_context->virtual_process;

    if (eu_context == vp->execution_units[0])
        vp->execution_units[0]->scheduler_object = OBJ_NEW(parsec_dequeue_t);

    parsec_barrier_wait(barrier);

    eu_context->scheduler_object = (void*)vp->execution_units[0]->scheduler_object;

    return 0;
}

static parsec_execution_context_t *sched_gd_select( parsec_execution_unit_t *eu_context )
{
    parsec_execution_context_t * context =
        (parsec_execution_context_t*)parsec_dequeue_try_pop_front( (parsec_dequeue_t*)eu_context->scheduler_object );
    return context;
}

static int sched_gd_schedule( parsec_execution_unit_t* eu_context,
                              parsec_execution_context_t* new_context )
{
#if defined(PINS_ENABLE)
    new_context->creator_core = 1;
#endif
    if( new_context->function->flags & PARSEC_HIGH_PRIORITY_TASK ) {
        parsec_dequeue_chain_front( (parsec_dequeue_t*)eu_context->scheduler_object, (parsec_list_item_t*)new_context);
    } else {
        parsec_dequeue_chain_back( (parsec_dequeue_t*)eu_context->scheduler_object, (parsec_list_item_t*)new_context);
    }
    return 0;
}

static void sched_gd_remove( parsec_context_t *master )
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
