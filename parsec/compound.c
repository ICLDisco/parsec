/*
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/scheduling.h"
#include "parsec/utils/debug.h"

/**
 * A compound is a list of taskpool that need to be executed sequentially
 * in the order in which they were added to the compound.
 */

typedef struct parsec_compound_taskpool_s {
    parsec_taskpool_t super;
    parsec_context_t* ctx;
    int32_t nb_taskpools;
    uint32_t completed_taskpools;
    parsec_taskpool_t** taskpool_array;
} parsec_compound_taskpool_t;

static void parsec_compound_taskpool_destructor( parsec_taskpool_t* tp )
{
    parsec_compound_taskpool_t* compound = (parsec_compound_taskpool_t*)tp;
    assert(PARSEC_TASKPOOL_TYPE_COMPOUND == compound->super.taskpool_type);
    PARSEC_DEBUG_VERBOSE(30, parsec_debug_output,
                         "Compound taskpool destructor %p", compound);
    free(compound->taskpool_array);
}

static int parsec_composed_taskpool_cb( parsec_taskpool_t* o, void* cbdata )
{
    parsec_compound_taskpool_t* compound = (parsec_compound_taskpool_t*)cbdata;
    int completed_taskpools = compound->completed_taskpools++;
    
    assert( o == compound->taskpool_array[completed_taskpools] ); (void)o;
    if( --compound->super.nb_pending_actions ) {
        assert( NULL != compound->taskpool_array[completed_taskpools+1] );
        PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool %p enable taskpool %p",
                             compound, compound->taskpool_array[completed_taskpools+1]);
        parsec_context_add_taskpool(compound->ctx,
                                    compound->taskpool_array[completed_taskpools+1]);
    } else {
        PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool completed %p",
                             compound);
        parsec_check_complete_cb((parsec_taskpool_t*)compound,
                                 compound->ctx, 0 /* no tp left on this compound */);
    }
    return 0;
}

static void
parsec_compound_taskpool_startup( parsec_context_t *context,
                                  parsec_taskpool_t *tp,
                                  parsec_task_t** startup_list )
{
    parsec_compound_taskpool_t* compound = (parsec_compound_taskpool_t*)tp;

    compound->ctx = context;
    compound->super.nb_pending_actions = compound->nb_taskpools;
    PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool %p starting with %d taskpools",
                         compound, compound->super.nb_pending_actions);
    for( int i = 0; i < compound->nb_taskpools; i++ ) {
        parsec_taskpool_t* o = compound->taskpool_array[i];
        assert( NULL != o );
        o->on_complete      = parsec_composed_taskpool_cb;
        o->on_complete_data = compound;
    }
    parsec_context_add_taskpool(compound->ctx, compound->taskpool_array[0]);
    (void)startup_list;
}

parsec_taskpool_t*
parsec_compose( parsec_taskpool_t* start,
                parsec_taskpool_t* next )
{
    parsec_compound_taskpool_t* compound = NULL;

    if( NULL == next )
        return start;
    if( NULL == start )
        return next;

    if( PARSEC_TASKPOOL_TYPE_COMPOUND == start->taskpool_type ) {  /* start is already a compound taskpool */
        compound = (parsec_compound_taskpool_t*)start;
        /* A compound is not a threadsafe object, once started we should not add more taskpools */
        assert(0 == compound->completed_taskpools);
        compound->taskpool_array[compound->nb_taskpools++] = next;
        /* make room for NULL terminating, if necessary */
        if( 0 == (compound->nb_taskpools % 16) ) {
            compound->taskpool_array = realloc(compound->taskpool_array,
                                               ((compound->nb_taskpools + 16) * sizeof(parsec_taskpool_t*)));
        }
        /* must always be NULL terminated */
        compound->taskpool_array[compound->nb_taskpools] = NULL;
        PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool %p add %d taskpool %p",
                             compound, compound->nb_taskpools, next );
    } else {
        compound = calloc(1, sizeof(parsec_compound_taskpool_t));
        PARSEC_OBJ_CONSTRUCT(compound, parsec_taskpool_t);
        compound->super.taskpool_type      = PARSEC_TASKPOOL_TYPE_COMPOUND;
        compound->taskpool_array = malloc(16 * sizeof(parsec_taskpool_t*));
        assert(NULL == compound->super.taskpool_name);
        asprintf(&compound->super.taskpool_name, "Compound Taskpool %d", next->taskpool_id);

        compound->taskpool_array[0] = start;
        compound->taskpool_array[1] = next;
        compound->taskpool_array[2] = NULL;
        compound->completed_taskpools = 0;
        compound->nb_taskpools = 2;
        compound->super.startup_hook = parsec_compound_taskpool_startup;
        compound->super.destructor = parsec_compound_taskpool_destructor;
        PARSEC_DEBUG_VERBOSE(30, parsec_debug_output, "Compound taskpool %p started with %p and %p taskpools",
                             compound, start, next );
    }
    return (parsec_taskpool_t*)compound;
}
