/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern char *strdup(const char *);

#include "dplasma.h"

static const dplasma_t** dplasma_array = NULL;
static int dplasma_array_size = 0, dplasma_array_count = 0;

static char *dplasma_dump_c(FILE *out, const dplasma_t *d, const char *prefix)
{
    char *pref2 = (char*)malloc(strlen(prefix) + 3);
    static char dp_txt[4096];
    int i;
    int p = 0;

    sprintf(pref2, "%s  ", prefix);
    
    p += snprintf(dp_txt+p, 4096-p, "%s{\n", prefix);
    p += snprintf(dp_txt+p, 4096-p, "%s.name   = \"%s\",\n", pref2, d->name);
    p += snprintf(dp_txt+p, 4096-p, "%s.flags  = 0x%02x,\n", pref2, d->flags);
    p += snprintf(dp_txt+p, 4096-p, "%s.dependencies_mask = 0x%02x,\n", pref2, d->dependencies_mask);
    p += snprintf(dp_txt+p, 4096-p, "%s.nb_locals = %d,\n", pref2, d->nb_locals);
    
    p += snprintf(dp_txt+p, 4096-p, "%s.locals = {", pref2);
    for(i = 0; i < d->nb_locals; i++) {
        p += snprintf(dp_txt+p, 4096-p, "dplasma_symbol_array[%d]%s", 
                      symbol_c_index_lookup(d->locals[i]),
                      i < MAX_LOCAL_COUNT-1 ? ", " : "},\n");
    }
    for(; i < MAX_LOCAL_COUNT; i++) {
        p += snprintf(dp_txt+p, 4096-p, "NULL%s",
                      i < MAX_LOCAL_COUNT-1 ? ", " : "},\n");
    }

    p += snprintf(dp_txt+p, 4096-p, "%s.preds = {", pref2);
    for(i = 0; i < MAX_PRED_COUNT; i++) {
        p += snprintf(dp_txt+p, 4096-p, "%s%s",
                      dump_c_expression(out, d->preds[i], prefix),
                      i < MAX_PRED_COUNT-1 ? ", " : "},\n");
    }

    p += snprintf(dp_txt+p, 4096-p, "%s.params = {", pref2);
    for(i = 0; i < MAX_PARAM_COUNT; i++) {
        p += snprintf(dp_txt+p, 4096-p, "%s%s",
                      dump_c_param(out, d->params[i], prefix),
                      i < MAX_PARAM_COUNT-1 ? ", " : "},\n");
    }

    p += snprintf(dp_txt+p, 4096-p, "%s.deps = NULL,\n", pref2);
    p += snprintf(dp_txt+p, 4096-p, "%s.hook = NULL,\n", pref2);
    p += snprintf(dp_txt+p, 4096-p, "%s.body = \"%s\"\n", pref2, d->body);
    p += snprintf(dp_txt+p, 4096-p, "%s}", prefix);
    
    free(pref2);

    return dp_txt;
}

void dplasma_dump(const dplasma_t *d, const char *prefix)
{
    char *pref2 = malloc(strlen(prefix)+3);
    int i;

    sprintf(pref2, "%s  ", prefix);
    printf("%sDplasma Function: %s\n", prefix, d->name);

    printf("%s Parameter Variables:\n", prefix);
    for(i = 0; i < MAX_LOCAL_COUNT && NULL != d->locals[i]; i++) {
        symbol_dump(d->locals[i], pref2);
    }

    printf("%s Predicates:\n", prefix);
    for(i = 0; i < MAX_PRED_COUNT && NULL != d->preds[i]; i++) {
        printf("%s", pref2);
        expr_dump(d->preds[i]);
        printf("\n");
    }

    printf("%s Parameters and Dependencies:\n", prefix);
    for(i = 0; i < MAX_PARAM_COUNT && NULL != d->params[i]; i++) {
        param_dump(d->params[i], pref2);
    }

    printf("%s Required dependencies mask: 0x%x (%s/%s/%s)\n", prefix,
           (int)d->dependencies_mask, (d->flags & DPLASMA_HAS_IN_IN_DEPENDENCIES ? "I" : "N"),
           (d->flags & DPLASMA_HAS_OUT_OUT_DEPENDENCIES ? "O" : "N"),
           (d->flags & DPLASMA_HAS_IN_STRONG_DEPENDENCIES ? "S" : "N"));
    printf("%s Body:\n", prefix);
    printf("%s  %s\n", prefix, d->body);

    if( NULL != d->deps ) {
        
        printf( "Current dependencies\n" );
    }

    free(pref2);
}

int dplasma_dplasma_index( const dplasma_t *d )
{
    int i;
    for(i = 0; i < dplasma_array_count; i++) {
        if( dplasma_array[i] == d ) {
            return i;
        }
    }
    return -1;
}

void dplasma_dump_all( void )
{
    int i;

    for( i = 0; i < dplasma_array_count; i++ ) {
        printf("/**\n * dplasma_t object named %s index %d\n */\n", dplasma_array[i]->name, i );
        dplasma_dump( dplasma_array[i], "" );
    }
}

void dplasma_dump_all_c(FILE *out)
{
    int i;
    char whole[8192];
    int p = 0;
    
    p += snprintf(whole+p, 8192-p, "dplasma_t dplasma_array[%d] = {\n", dplasma_array_count);
    for(i = 0; i < dplasma_array_count; i++) {
        p += snprintf(whole+p, 8192-p, "%s", dplasma_dump_c(out, dplasma_array[i], "    " ));
        if( i < dplasma_array_count-1) {
            p += snprintf(whole+p, 8192-p, ",\n");
        }
    }
    p += snprintf(whole+p, 8192-p, "};\n");
    fprintf(out, "%s", whole);
}


int dplasma_push( const dplasma_t* d )
{
    if( dplasma_array_count >= dplasma_array_size ) {
        if( 0 == dplasma_array_size ) {
            dplasma_array_size = 4;
        } else {
            dplasma_array_size *= 2;
        }
        dplasma_array = (const dplasma_t**)realloc( dplasma_array, dplasma_array_size * sizeof(dplasma_t*) );
        if( NULL == dplasma_array ) {
            return -1;  /* No more available memory */
        }
    }
    dplasma_array[dplasma_array_count] = d;
    dplasma_array_count++;
    return 0;
}

const dplasma_t* dplasma_find( const char* name )
{
    int i;
    const dplasma_t* object;

    for( i = 0; i < dplasma_array_count; i++ ) {
        object = dplasma_array[i];
        if( 0 == strcmp( object->name, name ) ) {
            return object;
        }
    }
    return NULL;
}

dplasma_t* dplasma_find_or_create( const char* name )
{
    dplasma_t* object;

    object = (dplasma_t*)dplasma_find(name);
    if( NULL != object ) {
        return object;
    }
    object = (dplasma_t*)calloc(1, sizeof(dplasma_t));
    object->name = strdup(name);
    if( 0 == dplasma_push(object) ) {
        return object;
    }
    free(object);
    return NULL;
}

const dplasma_t* dplasma_element_at( int i )
{
    if( i < dplasma_array_count ){
        return dplasma_array[i];
    }
    return NULL;
}

#ifdef _DEBUG
#define DEBUG(ARG)  printf ARG
#else
#define DEBUG(ARG)
#endif

/**
 * Compute the correct initial values for an execution context. These values
 * are in the range and validate all possible predicates. If such values do
 * not exist this function returns -1.
 */
int dplasma_set_initial_execution_context( dplasma_execution_context_t* exec_context )
{
    int i, rc;
    const dplasma_t* object = exec_context->function;
    const expr_t** predicates = (const expr_t**)object->preds;

    /* Compute the number of local values */
    if( 0 == object->nb_locals ) {
        /* special case for the IN/OUT objects */
        return 0;
    }

    for( i = 0; i < object->nb_locals; i++ ) {
        int min;
        exec_context->locals[i].sym = object->locals[i];
        rc = dplasma_symbol_get_first_value(object->locals[i], predicates,
                                            exec_context->locals, &min);
        if( rc != EXPR_SUCCESS ) {
        initial_values_one_loop_up:
            i--;
            if( i < 0 ) {
                printf( "Impossible to find initial values. Giving up\n" );
                return -1;
            }
            rc = dplasma_symbol_get_next_value(object->locals[i], predicates,
                                               exec_context->locals, &min );
            if( rc != EXPR_SUCCESS ) {
                goto initial_values_one_loop_up;
            }
        }
    }
    if( i < MAX_LOCAL_COUNT ) {
        exec_context->locals[i].sym = NULL;
    }
    return 0;
}

/**
 * Convert the execution context to a string.
 */
char* dplasma_service_to_string( const dplasma_execution_context_t* exec_context,
                                 char* tmp,
                                 size_t length )
{
    const dplasma_t* function = exec_context->function;
    int i, index = 0;

    index += snprintf( tmp + index, length - index, "%s( ", function->name );
    if( index >= length ) return tmp;
    for( i = 0; (NULL != function->locals[i]) && (i < MAX_LOCAL_COUNT); i++ ) {
        index += snprintf( tmp + index, length - index, "%d ",
                           exec_context->locals[i].value );
        if( index >= length ) return tmp;
    }
    index += snprintf( tmp + index, length - index, ")" );
    if( index >= length ) return tmp;

    return tmp;
}

/**
 * Convert a dependency to a string under the format X(...) -> Y(...).
 */
char* dplasma_dependency_to_string( const dplasma_execution_context_t* from,
                                    const dplasma_execution_context_t* to,
                                    char* tmp,
                                    size_t length )
{
    int index = 0;

    dplasma_service_to_string( from, tmp, length );
    index = strlen(tmp);
    index += snprintf( tmp + index, length - index, "->" );
    dplasma_service_to_string( to, tmp + index, length - index );
    return tmp;
}

int plasma_show_ranges( const dplasma_t* object )
{
    dplasma_execution_context_t* exec_context = (dplasma_execution_context_t*)malloc(sizeof(dplasma_execution_context_t));
    const expr_t** predicates = (const expr_t**)object->preds;
    int i, nb_locals;

    exec_context->function = (dplasma_t*)object;

    /* Compute the number of local values */
    for( i = nb_locals = 0; (NULL != object->locals[i]) && (i < MAX_LOCAL_COUNT); i++, nb_locals++ );
    if( 0 == nb_locals ) {
        /* special case for the IN/OUT obejcts */
        return 0;
    }
    printf( "Function %s (loops %d)\n", object->name, nb_locals );

    /**
     * This section of code walk through the tree and printout the local and global
     * minimum and maximum values for all local variables.
     */
    for( i = 0; i < nb_locals; i++ ) {
        int abs_min, min, abs_max, max;
        exec_context->locals[i].sym = object->locals[i];
        dplasma_symbol_get_first_value(object->locals[i], predicates,
                                       exec_context->locals, &min);
        exec_context->locals[i].value = min;
        dplasma_symbol_get_last_value(object->locals[i], predicates,
                                      exec_context->locals, &max);

        dplasma_symbol_get_absolute_minimum_value(object->locals[i], &abs_min);
        dplasma_symbol_get_absolute_maximum_value(object->locals[i], &abs_max);

        printf( "Range for local symbol %s is [%d..%d] (global range [%d..%d]) %s\n",
                object->locals[i]->name, min, max, abs_min, abs_max,
                (0 == dplasma_symbol_is_standalone(object->locals[i]) ? "[standalone]" : "[dependent]") );
    }
    return 0;
}

/**
 * This function generate all possible execution context for a given function with
 * respect to the predicates.
 */
int dplasma_show_tasks( const dplasma_t* object )
{
    dplasma_execution_context_t* exec_context = (dplasma_execution_context_t*)malloc(sizeof(dplasma_execution_context_t));
    const expr_t** predicates = (const expr_t**)object->preds;
    int rc, actual_loop;

    exec_context->function = (dplasma_t*)object;

    printf( "Function %s (loops %d)\n", object->name, object->nb_locals );
    if( 0 == object->nb_locals ) {
        /* special case for the IN/OUT obejcts */
        return 0;
    }

    if( 0 != dplasma_set_initial_execution_context(exec_context) ) {
        /* if we can't initialize the execution context then there is no reason to
         * continue.
         */
        return -1;
    }

    actual_loop = object->nb_locals - 1;
    while(1) {
        int value;

        /* Do whatever we have to do for this context */
        {
            char tmp[128];
            printf( "Execute %s\n", dplasma_service_to_string(exec_context, tmp, 128) );
        }

        /* Go to the next valid value for this loop context */
        rc = dplasma_symbol_get_next_value( object->locals[actual_loop], predicates,
                                            exec_context->locals, &value );

        /* If no more valid values, go to the previous loop,
         * compute the next valid value and redo and reinitialize all other loops.
         */
        if( rc != EXPR_SUCCESS ) {
            int current_loop = actual_loop;
        one_loop_up:
            DEBUG(("Loop index %d based on %s failed to get next value. Going up ...\n",
                   actual_loop, object->locals[actual_loop]->name));
            if( 0 == actual_loop ) {  /* we're done */
                goto end_of_all_loops;
            }
            actual_loop--;  /* one level up */
            rc = dplasma_symbol_get_next_value( object->locals[actual_loop], predicates,
                                                exec_context->locals, &value );
            if( rc != EXPR_SUCCESS ) {
                goto one_loop_up;
            }
            DEBUG(("Keep going on the loop level %d (symbol %s value %d)\n", actual_loop,
                   object->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
            for( actual_loop++; actual_loop <= current_loop; actual_loop++ ) {
                rc = dplasma_symbol_get_first_value(object->locals[actual_loop], predicates,
                                                    exec_context->locals, &value );
                if( rc != EXPR_SUCCESS ) {  /* no values for this symbol in this context */
                    goto one_loop_up;
                }
                DEBUG(("Loop index %d based on %s get first value %d\n", actual_loop,
                       object->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
            }
            actual_loop = current_loop;  /* go back to the original loop */
        } else {
            DEBUG(("Loop index %d based on %s get next value %d\n", actual_loop,
                   object->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
        }
    }
 end_of_all_loops:

    return 0;
}

/**
 * Resolve all IN() dependencies for this particular instance of execution.
 */
int dplasma_check_IN_dependencies( const dplasma_execution_context_t* exec_context )
{
    const dplasma_t* function = exec_context->function;
    const dplasma_t* in_function = dplasma_find("IN");
    int i, j, rc, value, mask = 0;
    param_t* param;
    dep_t* dep;

    if( !(function->flags & DPLASMA_HAS_IN_IN_DEPENDENCIES) ) {
        return 0;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->params[i]); i++ ) {
        param = function->params[i];

        if( !(SYM_IN & param->sym_type) ) {
            continue;  /* this is only an OUTPUT dependency */
        }
        for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != param->dep_in[j]); j++ ) {
            dep = param->dep_in[j];
            if( NULL != dep->cond ) {
                /* Check if the condition apply on the current setting */
                rc = expr_eval( dep->cond, exec_context->locals, MAX_LOCAL_COUNT, &value );
                if( 0 == value ) {
                    continue;
                }
            }
            if( dep->dplasma == in_function ) {
                mask = (mask << 1) | 0x1;
            }
        }
    }
    return mask;
}

#define CURRENT_DEPS_INDEX(K)  (exec_context->locals[(K)].value - deps->min)

/**
 * Release all OUT dependencies for this particular instance of the service.
 */
int dplasma_release_OUT_dependencies( dplasma_execution_context_t* exec_context )
{
    dplasma_t* function = exec_context->function;
    dplasma_dependencies_t *deps, **deps_location, *last_deps;
#ifdef _DEBUG
    char tmp[128];
#endif
    int i, actual_loop, mask;

    if( 0 == function->nb_locals ) {
        /* special case for the IN/OUT objects */
        return 0;
    }

    DEBUG(("Activate dependencies for %s\n", dplasma_service_to_string(exec_context, tmp, 128)));
    deps_location = &(function->deps);
    deps = *deps_location;
    last_deps = NULL;

    for( i = 0; i < function->nb_locals; i++ ) {
        if( NULL == (*deps_location) ) {
            int min, max, number;
            dplasma_symbol_get_absolute_minimum_value( function->locals[i], &min );
            dplasma_symbol_get_absolute_maximum_value( function->locals[i], &max );
            number = max - min;
            DEBUG(("Allocate %d spaces for loop %s (min %d max %d)\n",
                   number, function->locals[i]->name, min, max));
            deps = (dplasma_dependencies_t*)calloc(1, sizeof(dplasma_dependencies_t) +
                                                   number * sizeof(dplasma_dependencies_union_t));
            deps->flags = DPLASMA_DEPENDENCIES_FLAG_ALLOCATED | DPLASMA_DEPENDENCIES_FLAG_FINAL;
            deps->symbol = function->locals[i];
            deps->min = min;
            deps->max = max;
            deps->prev = last_deps; /* chain them backawrd */
            *deps_location = deps;  /* store the deps in the right location */
            if( NULL != last_deps ) {
                last_deps->flags = DPLASMA_DEPENDENCIES_FLAG_NEXT | DPLASMA_DEPENDENCIES_FLAG_ALLOCATED;
            }
        }
        deps = *deps_location;

        DEBUG(("Prepare storage for next loop variable (value %d) at %d\n",
               exec_context->locals[i].value, CURRENT_DEPS_INDEX(i)));
        deps_location = &(deps->u.next[CURRENT_DEPS_INDEX(i)]);
        last_deps = deps;
    }

    actual_loop = function->nb_locals - 1;
    while(1) {

        mask = 0x1;

        /* Mark the dependencies and check if this particular instance can be executed */
        /* TODO: This is a pretty ugly hack as it doesn't allow us to know which dependency
         * has been already satisfied, only to track the number if satisfied dependencies.
         */
        if( !(DPLASMA_DEPENDENCIES_HACK_IN & deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)]) ) {
            mask = dplasma_check_IN_dependencies( exec_context );
            if( mask > 0 ) {
                DEBUG(("Activate IN dependencies with mask 0x%02x\n", mask));
                while( mask > 0 ) {
                    deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)] = 
                        (deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)] << 1) | 0x1;
                    mask >>= 1;
                }
            }
            mask = DPLASMA_DEPENDENCIES_HACK_IN | 0x1;
        }

        deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)] = 
            (deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)] << 1) | mask;
        if( (deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)] & (~DPLASMA_DEPENDENCIES_HACK_IN))
            == function->dependencies_mask ) {
            /* This service is ready to be executed as all dependencies are solved. Let the
             * scheduler knows about this and keep going.
             */
            dplasma_execute(exec_context);
        } else {
            DEBUG(("  => Service %s not yet ready (required mask 0x%02x actual 0x%02x)\n",
                   dplasma_service_to_string( exec_context, tmp, 128 ), (int)function->dependencies_mask,
                   (int)(deps->u.dependencies[CURRENT_DEPS_INDEX(actual_loop)] & (~DPLASMA_DEPENDENCIES_HACK_IN))));
        }

        /* Go to the next valid value for this loop context */
        exec_context->locals[actual_loop].value++;
        if( exec_context->locals[actual_loop].max < exec_context->locals[actual_loop].value ) {
            /* We're out of the range for this variable */
            int current_loop = actual_loop;
        one_loop_up:
            DEBUG(("Loop index %d based on %s failed to get next value. Going up ...\n",
                   actual_loop, function->locals[actual_loop]->name));
            if( 0 == actual_loop ) {  /* we're done */
                goto end_of_all_loops;
            }
            actual_loop--;  /* one level up */
            deps = deps->prev;

            exec_context->locals[actual_loop].value++;
            if( exec_context->locals[actual_loop].max < exec_context->locals[actual_loop].value ) {
                goto one_loop_up;
            }
            DEBUG(("Keep going on the loop level %d (symbol %s value %d)\n", actual_loop,
                   function->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
            deps_location = &(deps->u.next[CURRENT_DEPS_INDEX(actual_loop)]);
            DEBUG(("Prepare storage for next loop variable (value %d) at %d\n",
                   exec_context->locals[actual_loop].value, CURRENT_DEPS_INDEX(actual_loop)));
            for( actual_loop++; actual_loop <= current_loop; actual_loop++ ) {
                exec_context->locals[actual_loop].value = exec_context->locals[actual_loop].min;
                last_deps = deps;  /* save the deps */
                if( NULL == *deps_location ) {
                    int min, max, number;
                    dplasma_symbol_get_absolute_minimum_value( function->locals[actual_loop], &min );
                    dplasma_symbol_get_absolute_maximum_value( function->locals[actual_loop], &max );
                    number = max - min;
                    DEBUG(("Allocate %d spaces for loop %s index %d value %d (min %d max %d)\n",
                           number, function->locals[actual_loop]->name, CURRENT_DEPS_INDEX(actual_loop-1),
                           exec_context->locals[actual_loop].value, min, max));
                    deps = (dplasma_dependencies_t*)calloc(1, sizeof(dplasma_dependencies_t) +
                                                           number * sizeof(dplasma_dependencies_union_t));
                    deps->flags = DPLASMA_DEPENDENCIES_FLAG_ALLOCATED | DPLASMA_DEPENDENCIES_FLAG_FINAL;
                    deps->symbol = function->locals[actual_loop];
                    deps->min = min;
                    deps->max = max;
                    deps->prev = last_deps; /* chain them backward */
                    *deps_location = deps;
                }
                deps = *deps_location;
                deps_location = &(deps->u.next[CURRENT_DEPS_INDEX(actual_loop)]);
                DEBUG(("Prepare storage for next loop variable (value %d) at %d\n",
                       exec_context->locals[actual_loop].value, CURRENT_DEPS_INDEX(actual_loop)));
                last_deps = deps;

                DEBUG(("Loop index %d based on %s get first value %d\n", actual_loop,
                       function->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
            }
            actual_loop = current_loop;  /* go back to the original loop */
        } else {
            DEBUG(("Loop index %d based on %s get next value %d\n", actual_loop,
                   function->locals[actual_loop]->name, exec_context->locals[actual_loop].value));
        }
    }
 end_of_all_loops:

    return 0;
}

/**
 * Execute the instance of the service based on the values of the
 * local variables stored in the execution context, by calling the
 * attached hook if any. At the end of the execution the dependencies
 * are released.
 */
int dplasma_execute( const dplasma_execution_context_t* exec_context )
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
            DEBUG(( "-> %s of %s( ", dep->sym_name, dep->dplasma->name ));
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
            if( dont_generate ) {
                continue;
            }

            /* Mark the end of the list */
            if( k < MAX_CALL_PARAM_COUNT ) {
                new_context.locals[k].sym = NULL;
            }
            DEBUG(( ")\n" ));
            dplasma_release_OUT_dependencies( &new_context );
        }
    }

    return 0;
}
