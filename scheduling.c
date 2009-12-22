/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "scheduling.h"
#include <string.h>
#include "lifo.h"

static int dplasma_execute( dplasma_execution_unit_t*, const dplasma_execution_context_t* );

#define DEPTH_FIRST_SCHEDULE 0

dplasma_atomic_lifo_t ready_list;

static int32_t taskstodo;

static void set_tasks_todo(int32_t n)
{
    taskstodo = n;
}

static int all_tasks_done(void)
{
    return (taskstodo == 0);
}

static void done_task()
{
    dplasma_atomic_dec_32b(&taskstodo);
}

/**
 * Schedule the instance of the service based on the values of the
 * local variables stored in the execution context, by calling the
 * attached hook if any. At the end of the execution the dependencies
 * are released.
 */
int dplasma_schedule( dplasma_context_t* context, const dplasma_execution_context_t* exec_context )
{
#if !DEPTH_FIRST_SCHEDULE
    dplasma_execution_unit_t* eu_context;

    eu_context = &(context->execution_units[0]);

    return __dplasma_schedule( eu_context, exec_context );
#else
    return dplasma_execute(eu_context, exec_context);
#endif  /* !DEPTH_FIRST_SCHEDULE */
}

int __dplasma_schedule( dplasma_execution_unit_t* eu_context,
                        const dplasma_execution_context_t* exec_context )
{
#if !DEPTH_FIRST_SCHEDULE
    dplasma_execution_context_t* new_context;

    new_context = (dplasma_execution_context_t*)malloc(sizeof(dplasma_execution_context_t));
    memcpy( new_context, exec_context, sizeof(dplasma_execution_context_t) );
    dplasma_atomic_lifo_push( &ready_list, (dplasma_list_item_t*)new_context);
    return 0;
#else
    printf( "This internal version of the dplasma_schedule is not supposed to be called\n");
    return -1;
#endif  /* !DEPTH_FIRST_SCHEDULE */
}

void dplasma_register_nb_tasks(int n)
{
    set_tasks_todo((int32_t)n);
}

int dplasma_progress(dplasma_context_t* context)
{
    dplasma_execution_context_t* exec_context;
    dplasma_execution_unit_t* eu_context;
    int nbiterations = 0, my_id;

    /* Get my Execution Unit context */
    my_id = dplasma_atomic_inc_32b(&(context->eu_waiting)) - 1;
    eu_context = &(context->execution_units[my_id]);
    eu_context->eu_id = my_id;

    while( !all_tasks_done() ) {

        /* extract the first exeuction context from the ready list */
        exec_context = (dplasma_execution_context_t*)dplasma_atomic_lifo_pop(&(ready_list));

        if( exec_context != NULL ) {
            /* We're good to go ... */
            dplasma_execute( eu_context, exec_context );
            done_task();
            nbiterations++;
            /* Release the execution context */
            free( exec_context );
        }
    }
    return nbiterations;
}

static int dplasma_execute( dplasma_execution_unit_t* eu_context,
                            const dplasma_execution_context_t* exec_context )
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
        function->hook( eu_context, exec_context );
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
            DEBUG(( " -> %s( ", dep->dplasma->name ));
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
            dplasma_release_OUT_dependencies( eu_context,
                                              exec_context, param,
                                              &new_context, dep->param );
        }
    }

    return 0;
}
