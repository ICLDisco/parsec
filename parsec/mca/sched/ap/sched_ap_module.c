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

#include "parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/debug.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/ap/sched_ap.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"
static int SYSTEM_NEIGHBOR = 0;

/**
 * Module functions
 */
static int sched_ap_install(parsec_context_t* master);
static int sched_ap_schedule(parsec_execution_unit_t* eu_context,
                             parsec_execution_context_t* new_context,
                             int32_t distance);
static parsec_execution_context_t*
sched_ap_select(parsec_execution_unit_t *eu_context,
                int32_t* distance);
static int flow_ap_init(parsec_execution_unit_t* eu_context, struct parsec_barrier_t* barrier);
static void sched_ap_remove(parsec_context_t* master);

const parsec_sched_module_t parsec_sched_ap_module = {
    &parsec_sched_ap_component,
    {
        sched_ap_install,
        flow_ap_init,
        sched_ap_schedule,
        sched_ap_select,
        NULL,
        sched_ap_remove
    }
};

static int sched_ap_install( parsec_context_t *master )
{
    SYSTEM_NEIGHBOR = master->nb_vp * master->virtual_processes[0]->nb_cores;
    return 0;
}

static int flow_ap_init(parsec_execution_unit_t* eu_context, struct parsec_barrier_t* barrier)
{
    parsec_vp_t *vp = eu_context->virtual_process;

    if (eu_context == vp->execution_units[0])
        vp->execution_units[0]->scheduler_object = OBJ_NEW(parsec_list_t);

    parsec_barrier_wait(barrier);

    eu_context->scheduler_object = (void*)vp->execution_units[0]->scheduler_object;

    return 0;
}

static parsec_execution_context_t*
sched_ap_select(parsec_execution_unit_t *eu_context,
                int32_t* distance)
{
    parsec_execution_context_t * context =
        (parsec_execution_context_t*)parsec_list_pop_front((parsec_list_t*)eu_context->scheduler_object);
#if defined(PINS_ENABLE)
    if (NULL != context)
        context->victim_core = SYSTEM_NEIGHBOR;
#endif  /* defined(PINS_ENABLE) */
    *distance = 0;
    return context;
}

static int sched_ap_schedule(parsec_execution_unit_t* eu_context,
                             parsec_execution_context_t* new_context,
                             int32_t distance)
{
#if defined(PARSEC_DEBUG_NOISIER)
    parsec_list_item_t *it = (parsec_list_item_t*)new_context;
    char tmp[MAX_TASK_STRLEN];
    do {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "AP:\t Pushing task %s",
                parsec_snprintf_execution_context(tmp, MAX_TASK_STRLEN, (parsec_execution_context_t*)it));
        it = (parsec_list_item_t*)((parsec_list_item_t*)it)->list_next;
    } while( it != (parsec_list_item_t*)new_context );
#endif
    parsec_list_chain_sorted((parsec_list_t*)eu_context->scheduler_object,
                            (parsec_list_item_t*)new_context,
                            parsec_execution_context_priority_comparator);
    (void)distance;
    return 0;
}

static void sched_ap_remove( parsec_context_t *master )
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
