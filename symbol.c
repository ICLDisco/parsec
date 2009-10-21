/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "symbol.h"
#include "dplasma.h"

extern int dplasma_lineno;

static const symbol_t** dplasma_symbol_array = NULL;
static int dplasma_symbol_array_count = 0,
    dplasma_symbol_array_size = 0;

void symbol_dump(const symbol_t *s, const char *prefix)
{
    if( NULL == s->name ) {
        return;
    }

    if( s->min == s->max ) {
        printf("%s%s = ", prefix, s->name);
        expr_dump(s->min);
        printf("\n" );
    } else {
        printf("%s%s = [", prefix, s->name);
        expr_dump(s->min);
        printf(" .. ");
        expr_dump(s->max);
        printf("]\n");
    }
}

void symbol_dump_all( const char* prefix )
{
    const symbol_t* symbol;
    int i;

    for( i = 0; i < dplasma_symbol_array_count; i++ ) {
        symbol = dplasma_symbol_array[i];
        symbol_dump( symbol, prefix );
    }
}

int dplasma_add_global_symbol( const char* name, const expr_t* expr )
{
    symbol_t* symbol;

    if( NULL != dplasma_search_global_symbol(name) ) {
        printf( "Overwrite an already defined symbol %s at line %d\n",
                name, dplasma_lineno );
        return -1;
    }

    if( dplasma_symbol_array_count >= dplasma_symbol_array_size ) {
        if( 0 == dplasma_symbol_array_size ) {
            dplasma_symbol_array_size = 4;
        } else {
            dplasma_symbol_array_size *= 2;
        }
        dplasma_symbol_array = (const symbol_t**)realloc( dplasma_symbol_array,
                                                          dplasma_symbol_array_size * sizeof(symbol_t*) );
        if( NULL == dplasma_symbol_array ) {
            return -1;  /* No more available memory */
        }
    }

    symbol = (symbol_t*)calloc(1, sizeof(symbol_t));
    symbol->name = strdup(name);
    symbol->min = expr;
    symbol->max = expr;

    dplasma_symbol_array[dplasma_symbol_array_count] = symbol;
    dplasma_symbol_array_count++;
    return EXPR_SUCCESS;
}

const symbol_t* dplasma_search_global_symbol( const char* name )
{
    int i;
    const symbol_t* symbol;

    for( i = 0; i < dplasma_symbol_array_count; i++ ) {
        symbol = dplasma_symbol_array[i];
        if( 0 == strcmp(symbol->name, name) ) {
            return symbol;
        }
    }
    return NULL;
}

int dplasma_symbol_get_first_value( const symbol_t* symbol,
                                    const expr_t** predicates,
                                    assignment_t* local_context,
                                    int* pvalue )
{
    assignment_t* assignment;
    int rc, min, max, val, old_value, pred_index, pred_val, valid_value;

    rc = expr_eval( symbol->min, local_context, MAX_LOCAL_COUNT, &min );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the min expression for symbol %s\n", symbol->name);
        return rc;
    }
    rc = expr_eval( symbol->max, local_context, MAX_LOCAL_COUNT, &max );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the max expression for symbol %s\n", symbol->name);
        return rc;
    }

    rc = dplasma_add_assignment( symbol, local_context, MAX_LOCAL_COUNT, &assignment );
    if( DPLASMA_ASSIGN_ERROR == rc ) {
        /* the symbol cannot be added to the local context. Bail out */
        return rc;
    }

    /* If there are no predicates we're good to go */
    if( NULL == predicates ) {
        assignment->value = min;
        return EXPR_SUCCESS;
    }

    old_value = assignment->value;

    for( val = min; val <= max; val++ ) {
        /* Update the variable */
        assignment->value = val;
        valid_value = 1;  /* suppose this is the right value */
        /* Any valid value have to match all predicats. */
        for( pred_index = 0;
             (pred_index < MAX_PRED_COUNT) && (NULL != predicates[pred_index]);
             pred_index++ ) {
            /* If we fail to evaluate the expression, let's suppose we don't have
             * all the required symbols in the assignment array.
             */
            if( EXPR_SUCCESS == expr_eval(predicates[pred_index],
                                          local_context, MAX_LOCAL_COUNT,
                                          &pred_val) ) {
                if( 0 == pred_val ) {
                    /* This particular value doesn't fit. Go to the next one */
                    valid_value = 0;
                    break;
                }
            }
        }
        if( 1 == valid_value ) {
            /* If we're here, then we have the correct value. */
            *pvalue = val;
            return EXPR_SUCCESS;
        }
    }

    assignment->value = old_value; /* restore the old value */
    return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
}

int dplasma_symbol_get_last_value( const symbol_t* symbol,
                                   const expr_t** predicates,
                                   assignment_t* local_context,
                                   int* pvalue )
{
    assignment_t* assignment;
    int rc, min, max, val, old_value, pred_index, pred_val, valid_value;

    rc = expr_eval( symbol->min, local_context, MAX_LOCAL_COUNT, &min );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the min expression for symbol %s\n", symbol->name);
        return rc;
    }
    rc = expr_eval( symbol->max, local_context, MAX_LOCAL_COUNT, &max );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the max expression for symbol %s\n", symbol->name);
        return rc;
    }

    rc = dplasma_add_assignment( symbol, local_context, MAX_LOCAL_COUNT, &assignment );
    if( DPLASMA_ASSIGN_ERROR == rc ) {
        /* the symbol cannot be added to the local context. Bail out */
        return rc;
    }

    /* If there are no predicates we're good to go */
    if( NULL == predicates ) {
        assignment->value = max;
        return EXPR_SUCCESS;
    }

    old_value = assignment->value;

    for( val = max; val >= min; val-- ) {
        assignment->value = val;
        valid_value = 1;  /* suppose this is the right value */
        for( pred_index = 0;
             (pred_index < MAX_PRED_COUNT) && (NULL != predicates[pred_index]);
             pred_index++ ) {
            /* If we fail to evaluate the expression, let's suppose we don't have
             * all the required symbols in the assignment array.
             */
            if( EXPR_SUCCESS == expr_eval(predicates[pred_index],
                                          local_context, MAX_LOCAL_COUNT,
                                          &pred_val) ) {
                if( 0 == pred_val ) {
                    /* This particular value doesn't fit. Go to the next one */
                    valid_value = 0;
                    break;
                }
            }
        }
        if( 1 == valid_value ) {
            /* If we're here, then we have the correct value. */
            *pvalue = val;
            return EXPR_SUCCESS;
        }
    }

    assignment->value = old_value; /* restore the old value */
    return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
}

int dplasma_symbol_get_next_value( const symbol_t* symbol,
                                   const expr_t** predicates,
                                   assignment_t* local_context,
                                   int* pvalue )
{
    assignment_t* assignment;
    int rc, max, val, old_value, pred_index, pred_val, valid_value;

    rc = expr_eval( symbol->max, local_context, MAX_LOCAL_COUNT, &max );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the max expression for symbol %s\n", symbol->name);
        return rc;
    }

    rc = dplasma_find_assignment( symbol->name, local_context, MAX_LOCAL_COUNT, &assignment );
    if( DPLASMA_ASSIGN_ERROR == rc ) {
        /* the symbol is not yet on the assignment list, so there is ABSOLUTELY
         * no reason to ask for the next value.
         */
        return rc;
    }
    /* If there are no predicates we're good to go */
    if( NULL == predicates ) {
        assignment->value = (*pvalue) + 1;
        return EXPR_SUCCESS;
    }
    old_value = assignment->value;

    for( val = (*pvalue) + 1; val <= max; val++ ) {
        assignment->value = val;
        valid_value = 1;
        for( pred_index = 0;
             (pred_index < MAX_PRED_COUNT) && (NULL != predicates[pred_index]);
             pred_index++ ) {
            /* If we fail to evaluate the expression, let's suppose we don't have
             * all the required symbols in the assignment array.
             */
            if( EXPR_SUCCESS == expr_eval(predicates[pred_index],
                                          local_context, MAX_LOCAL_COUNT,
                                          &pred_val) ) {
                if( 0 == pred_val ) {
                    /* This particular value doesn't fit. Go to the next one */
                    valid_value = 0;
                    break;
                }
            }
        }
        if( 1 == valid_value ) {
            /* If we're here, then we have the correct value. */
            *pvalue = val;
            return EXPR_SUCCESS;
        }
    }

    assignment->value = old_value; /* restore the old value */
    return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
}

