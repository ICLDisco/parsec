/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "dplasma.h"

static const dplasma_t** dplasma_array = NULL;
static int dplasma_array_size = 0, dplasma_array_count = 0;

void dplasma_dump(const dplasma_t *d, const char *prefix)
{
    int i;
    char *pref2 = malloc(strlen(prefix)+3);

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

    printf("%s Body:\n", prefix);
    printf("%s  %s\n", prefix, d->body);

    free(pref2);
}

void dplasma_dump_all( void )
{
    int i;

    for( i = 0; i < dplasma_array_count; i++ ) {
        printf("/**\n * dplasma_t object named %s index %d\n */\n", dplasma_array[i]->name, i );
        dplasma_dump( dplasma_array[i], "" );
    }
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

const dplasma_t* dplasma_element_at( int i )
{
    if( i < dplasma_array_count ){
        return dplasma_array[i];
    }
    return NULL;
}

/* There is another loop after this one. */
#define DPLASMA_LOOP_NEXT       0x01
/* This is the final loop */
#define DPLASMA_LOOP_FINAL      0x02
/* This loops array is allocated */
#define DPLASMA_LOOP_ALLOCATED  0x04
typedef struct dplasma_final_loop_values_t {
    char array[1];
} dplasma_final_loop_values_t;

typedef struct dplasma_loop_values_t dplasma_loop_values_t;
typedef union dplasma_loop_union_t {
    dplasma_loop_values_t* loops[1];
    dplasma_final_loop_values_t* array[1];
} dplasma_loop_union_t;

struct dplasma_loop_values_t {
    int                   type;
    symbol_t*             symbol;
    dplasma_loop_union_t* generic_next;
    int                   min;
    int                   max;
    dplasma_loop_union_t  u;
};

dplasma_loop_values_t*
dplasma_dependency_create_loop( int type,
                                symbol_t* symbol,
                                int min,
                                int max )
{
    dplasma_loop_values_t* loops = (dplasma_loop_values_t*)malloc(sizeof(dplasma_loop_values_t));

    loops->type = type;
    loops->symbol = symbol;
    loops->generic_next = NULL;
    loops->min = min;
    loops->max = max;

    return loops;
}

typedef struct dplasma_execution_context_t {
    const dplasma_t* function;
    assignment_t locals[MAX_LOCAL_COUNT];
    dplasma_loop_values_t* loops;
} dplasma_execution_context_t;

int dplasma_dependency_activate( dplasma_execution_context_t* context, int arg_index, ... )
{
    return 0;
}

int dplasma_unroll( const dplasma_t* object )
{
    dplasma_execution_context_t* exec_context = (dplasma_execution_context_t*)malloc(sizeof(dplasma_execution_context_t));
    int i, nb_locals;
    dplasma_loop_values_t* last_loop = NULL;

    exec_context->function = object;
    printf( "Function %s\n", object->name );

    /* Compute the number of local values */
    for( i = nb_locals = 0; (NULL != object->locals[i]) && (i < MAX_LOCAL_COUNT); i++, nb_locals++ );

    for( i = 0; (NULL != object->locals[i]) && (i < MAX_LOCAL_COUNT); i++ ) {
        dplasma_loop_values_t* loop;
        int abs_min, min, abs_max, max;
        exec_context->locals[i].sym = object->locals[i];
        dplasma_symbol_get_first_value(object->locals[i], (const expr_t**)object->preds,
                                       exec_context->locals, &min);
        exec_context->locals[i].value = min;
        dplasma_symbol_get_last_value(object->locals[i], (const expr_t**)object->preds,
                                      exec_context->locals, &max);

        dplasma_symbol_get_absolute_minimum_value(object->locals[i], &abs_min);
        dplasma_symbol_get_absolute_maximum_value(object->locals[i], &abs_max);

        printf( "Range for local symbol %s is [%d..%d] (global range [%d..%d]) %s\n",
                object->locals[i]->name, min, max, abs_min, abs_max,
                (0 == dplasma_symbol_is_standalone(object->locals[i]) ? "[standalone]" : "[dependent]") );

        loop = dplasma_dependency_create_loop( DPLASMA_LOOP_FINAL, exec_context->locals[i].sym, min, max );
        if( NULL != last_loop ) {
            last_loop->type = DPLASMA_LOOP_NEXT;
            last_loop->generic_next = loop;
        } else {
            exec_context->loops = loop;
        }
        last_loop = loop;
    }
    return 0;
}
