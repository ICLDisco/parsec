/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "scheduling.h"
#include <string.h>

static dplasma_execution_context_t* ready_list = NULL;
static int dplasma_execute( const dplasma_execution_context_t* exec_context );

#define DEPTH_FIRST_SCHEDULE 1

/**
 * Schedule the instance of the service based on the values of the
 * local variables stored in the execution context, by calling the
 * attached hook if any. At the end of the execution the dependencies
 * are released.
 */
int dplasma_schedule( const dplasma_execution_context_t* exec_context )
{
#if DEPTH_FIRST_SCHEDULE
    dplasma_execution_context_t* new_context;

    new_context = (dplasma_execution_context_t*)malloc(sizeof(dplasma_execution_context_t));
    memcpy( new_context, exec_context, sizeof(dplasma_execution_context_t) );
    if( NULL == ready_list ) {
        new_context->prev = new_context;
        new_context->next = new_context;
        ready_list = new_context;
    } else {
        new_context->next = ready_list;
        new_context->prev = ready_list->prev;
        ready_list->prev = new_context;
        new_context->prev->next = new_context;
    }
    return 0;
#else
    return dplasma_execute(exec_context);
#endif  /* DEPTH_FIRST_SCHEDULE */
}

int dplasma_progress(void)
{
    dplasma_execution_context_t* exec_context;

    while( NULL != ready_list ) {
        /* extract the first exeuction context from the ready list */
        exec_context = ready_list;
        if( exec_context == exec_context->next ) {
            ready_list = NULL;
        } else {
            ready_list = exec_context->next;
            ready_list->prev = exec_context->prev;
            exec_context->prev->next = ready_list;
        }

        /* We're good to go ... */
        dplasma_execute( exec_context );

        /* Release the execution context */
        free( exec_context );
    }
    return 0;
}

static int dplasma_execute( const dplasma_execution_context_t* exec_context )
{
    dplasma_t* function = exec_context->function;
    dplasma_execution_context_t new_context;
    int i, j, k, rc, value;
#ifdef _DEBUG
    char tmp[128];
#endif
    param_t* param;
    dep_t* dep;

    if( NULL != function->hook ) {
        function->hook( exec_context );
    } else {
        DEBUG(( "Execute %s\n", dplasma_service_to_string(exec_context, tmp, 128)));
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->params[i]); i++ ) {
        param = function->params[i];

        if( !(SYM_OUT & param->sym_type) ) {
            continue;  /* this is only an INPUT dependency */
        }
        for( j = 0; (j < MAX_DEP_OUT_COUNT) && (NULL != param->dep_out[j]); j++ ) {
            int dont_generate = 0;

            dep = param->dep_out[j];
            if( NULL != dep->cond ) {
                /* Check if the condition apply on the current setting */
                rc = expr_eval( dep->cond, exec_context->locals, MAX_LOCAL_COUNT, &value );
                if( 0 == value ) {
                    continue;
                }
            }
            new_context.function = dep->dplasma;
            DEBUG(( " -> %s of %s( ", dep->sym_name, dep->dplasma->name ));
            /* Check to see if any of the params are conditionals or ranges and if they are
             * if they match. If yes, then set the correct values.
             */
            for( k = 0; (k < MAX_CALL_PARAM_COUNT) && (NULL != dep->call_params[k]); k++ ) {
                new_context.locals[k].sym = dep->dplasma->locals[k];
                if( EXPR_OP_BINARY_RANGE != dep->call_params[k]->op ) {
                    rc = expr_eval( dep->call_params[k], exec_context->locals, MAX_LOCAL_COUNT, &value );
                    new_context.locals[k].min = new_context.locals[k].max = value;
                    DEBUG(( "%d ", value ));
                } else {
                    int min, max;
                    rc = expr_range_to_min_max( dep->call_params[k], exec_context->locals, MAX_LOCAL_COUNT, &min, &max );
                    if( min > max ) {
                        dont_generate = 1;
                        DEBUG(( " -- skipped" ));
                        break;  /* No reason to continue here */
                    }
                    new_context.locals[k].min = min;
                    new_context.locals[k].max = max;
                    if( min == max ) {
                        DEBUG(( "%d ", min ));
                    } else {
                        DEBUG(( "[%d..%d] ", min, max ));
                    }
                }
                new_context.locals[k].value = new_context.locals[k].min;
            }
            DEBUG(( ")\n" ));
            if( dont_generate ) {
                continue;
            }

            /* Mark the end of the list */
            if( k < MAX_CALL_PARAM_COUNT ) {
                new_context.locals[k].sym = NULL;
            }
            dplasma_release_OUT_dependencies( exec_context, param,
                                              &new_context, dep->param );
        }
    }

    return 0;
}
