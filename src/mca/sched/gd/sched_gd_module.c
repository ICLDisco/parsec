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
 * These symbols are in a file by themselves to provide nice linker
 * semantics.  Since linkers generally pull in symbols by object
 * files, keeping these symbols as the only symbols in this file
 * prevents utility programs such as "ompi_info" from having to import
 * entire components just to query their version and parameters.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "debug.h"
#include "dague/mca/sched/sched.h"
#include "dague/mca/sched/gd/sched_gd.h"
#include "dequeue.h"
#include "dague/mca/pins/pins.h"
static int SYSTEM_NEIGHBOR = 0;

/*
 * Module functions
 */
static int sched_gd_install(dague_context_t* master);
static int sched_gd_schedule(dague_execution_unit_t* eu_context, dague_execution_context_t* new_context);
static dague_execution_context_t *sched_gd_select( dague_execution_unit_t *eu_context );
static void sched_gd_remove(dague_context_t* master);

const dague_sched_module_t dague_sched_gd_module = {
    &dague_sched_gd_component,
    {
        sched_gd_install,
        NULL,
        sched_gd_schedule,
        sched_gd_select,
        NULL,
        sched_gd_remove
    }
};

static int sched_gd_install( dague_context_t *master )
{
    int p, t;
    dague_vp_t *vp;
    dague_dequeue_t *q;

    SYSTEM_NEIGHBOR = master->nb_vp * master->virtual_processes[0]->nb_cores; // defined for instrumentation

    for(p = 0; p < master->nb_vp; p++) {
        q = malloc(sizeof(dague_dequeue_t));
        OBJ_CONSTRUCT( q, dague_dequeue_t );

        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            vp->execution_units[t]->scheduler_object = (void*)q;
        }

    }

    return 0;
}

static dague_execution_context_t *sched_gd_select( dague_execution_unit_t *eu_context )
{
    dague_execution_context_t * context =
        (dague_execution_context_t*)dague_dequeue_try_pop_front( (dague_dequeue_t*)eu_context->scheduler_object );
    if (NULL != context)
        context->victim_core = SYSTEM_NEIGHBOR;
    return context;
}

static int sched_gd_schedule( dague_execution_unit_t* eu_context,
                              dague_execution_context_t* new_context )
{
    new_context->creator_core = 1;
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
