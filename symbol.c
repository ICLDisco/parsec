/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "symbol.h"

static symbol_t** dplasma_symbol_array = NULL;
static int dplasma_symbol_array_count = 0,
    dplasma_symbol_array_size = 0;

int dplasma_symbol_get_count(void) 
{
    return dplasma_symbol_array_count;
}

const symbol_t *dplasma_symbol_get_element_at( int i )
{
    if( i >= dplasma_symbol_array_count ) {
        return NULL;
    } else {
        return dplasma_symbol_array[i];
    }
}

void dplasma_load_symbols( symbol_t **array, int size )
{
    int i, j, found;

    if( size + dplasma_symbol_array_count > dplasma_symbol_array_size ) {
        dplasma_symbol_array_size = size + dplasma_symbol_array_count;
        dplasma_symbol_array = (symbol_t ** )realloc(dplasma_symbol_array, dplasma_symbol_array_size * sizeof(symbol_t*));
    }

    for(i = 0; i < size; i++) {
        found = 0;
        for(j = 0; j < dplasma_symbol_array_count; j++) {
            if( !strcmp( array[i]->name, dplasma_symbol_array[j]->name) ) {
                found = 1;
                break;
            }
        }
        if( 0 == found ) {
            dplasma_symbol_array[dplasma_symbol_array_count] = array[i];
            dplasma_symbol_array_count++;
        }
    }
}

void symbol_dump(const symbol_t *s, const char *prefix)
{
    if( NULL == s->name ) {
        return;
    }

    if( s->min == s->max ) {
        if( EXPR_FLAG_CONSTANT & s->min->flags ) {
            printf("%s%s:%s%s = {%d = ", prefix, s->name,
                   (DPLASMA_SYMBOL_IS_GLOBAL & s->flags ? "G" : "L"),
                   (DPLASMA_SYMBOL_IS_STANDALONE & s->flags ? "S" : "D"), s->min->value);
            expr_dump(stdout, s->min);
            printf("}\n" );
        } else {
            printf("%s%s:%s%s = ", prefix, s->name,
                   (DPLASMA_SYMBOL_IS_GLOBAL & s->flags ? "G" : "L"),
                   (DPLASMA_SYMBOL_IS_STANDALONE & s->flags ? "S" : "D"));
            expr_dump(stdout, s->min);
            printf("\n" );
        }
    } else {
        printf("%s%s:%s%s = [", prefix, s->name,
               (DPLASMA_SYMBOL_IS_GLOBAL & s->flags ? "G" : "L"),
               (DPLASMA_SYMBOL_IS_STANDALONE & s->flags ? "S" : "D"));
        expr_dump(stdout, s->min);
        printf(" .. ");
        expr_dump(stdout, s->max);
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

int symbol_c_index_lookup( const symbol_t *symbol )
{
    int i;
    for(i = 0; i < dplasma_symbol_array_count; i++) {
        if ( symbol == dplasma_symbol_array[i] ) {
            return i;
        }
    }
    return -1;
}

int dplasma_assign_global_symbol( const char *name, const expr_t *expr )
{
    symbol_t* symbol;

    if( NULL == (symbol = dplasma_search_global_symbol(name)) ) {
        DEBUG(("Cannot assign symbol %s: not defined\n", name));
        return -1;
    }

    symbol->min = expr;
    symbol->max = expr;

    return EXPR_SUCCESS;
}

int dplasma_add_global_symbol( const char *name )
{
    symbol_t* symbol;

    if( NULL != dplasma_search_global_symbol(name) ) {
        DEBUG(("Add global symbol cst: symbol %s is already defined\n", name));
        return -1;
    }

    if( dplasma_symbol_array_count >= dplasma_symbol_array_size ) {
        if( 0 == dplasma_symbol_array_size ) {
            dplasma_symbol_array_size = 4;
        } else {
            dplasma_symbol_array_size *= 2;
        }
        dplasma_symbol_array = (symbol_t**)realloc( dplasma_symbol_array,
                                                    dplasma_symbol_array_size * sizeof(symbol_t*) );
        if( NULL == dplasma_symbol_array ) {
            return -1;  /* No more available memory */
        }
    }

    symbol = (symbol_t*)calloc(1, sizeof(symbol_t));
    symbol->flags = DPLASMA_SYMBOL_IS_GLOBAL;
    symbol->name = strdup(name);

    dplasma_symbol_array[dplasma_symbol_array_count] = symbol;
    dplasma_symbol_array_count++;
    return EXPR_SUCCESS;
}

int dplasma_add_global_symbol_cst( const char* name, const expr_t* expr )
{
    int ret;
    ret = dplasma_add_global_symbol( name );
    if( ret != EXPR_SUCCESS ) 
        return ret;
    return dplasma_assign_global_symbol( name, expr );
}

symbol_t* dplasma_search_global_symbol( const char* name )
{
    symbol_t* symbol;
    int i;

    for( i = 0; i < dplasma_symbol_array_count; i++ ) {
        symbol = dplasma_symbol_array[i];
        if( 0 == strcmp(symbol->name, name) ) {
            return symbol;
        }
    }
    return NULL;
}

static int dplasma_expr_parse_callback( const symbol_t* symbol, void* data )
{
    int* pvalue = (int*)data;

    if( !(DPLASMA_SYMBOL_IS_GLOBAL & symbol->flags) ) {
        /* Allow us to count the number of local symbols in the expression */
        (*pvalue)++;
    }
    return EXPR_SUCCESS;
}

int dplasma_symbol_is_standalone( const symbol_t* symbol )
{
    int rc, symbols_count = 0;

    rc = expr_parse_symbols( symbol->min, &dplasma_expr_parse_callback, (void*)&symbols_count );
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }
    if( 0 == symbols_count ) {
        rc = expr_parse_symbols( symbol->max, &dplasma_expr_parse_callback, (void*)&symbols_count );
        if( EXPR_SUCCESS != rc ) {
            return rc;
        }
        return (0 == symbols_count ? EXPR_SUCCESS : EXPR_FAILURE_UNKNOWN);
    }
    return EXPR_FAILURE_UNKNOWN;
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

    /* TODO: as we only accept increasing ranges, we cannot tolerate minimum values
     * bigger than the maximum ones */
    if( min > max ) {
        return EXPR_FAILURE_UNKNOWN;
    }

    /* If there are no predicates we're good to go */
    if( NULL == predicates ) {
        assignment->value = min;
        *pvalue = min;
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
        *pvalue = max;
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
        val = assignment->value + 1;
        if( val <= max ) {
            *pvalue = val;
            assignment->value = val;
            return EXPR_SUCCESS;
        }
        return EXPR_FAILURE_UNKNOWN;
    }
    old_value = assignment->value;

    for( val = assignment->value + 1; val <= max; val++ ) {
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

int dplasma_symbol_validate_value( const symbol_t* symbol,
                                   const expr_t** predicates,
                                   assignment_t* local_context )
{
    assignment_t* assignment;
    int rc, min, max, pred_index, pred_val, valid_value = 1;

    rc = dplasma_find_assignment( symbol->name, local_context, MAX_LOCAL_COUNT, &assignment );
    if( DPLASMA_ASSIGN_ERROR == rc ) {
        /* the symbol is not yet on the assignment list, so there is ABSOLUTELY
         * no reason to ask for the next value.
         */
        return rc;
    }

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

    /* Make sure the current value is in the legal range */
    if( (assignment->value < min) || (assignment->value > max) || (min > max) ) {
        return EXPR_FAILURE_UNKNOWN;
    }

    /* If there are no predicates we're good to go */
    if( NULL == predicates ) {
        return EXPR_SUCCESS;
    }

    for( pred_index = 0;
         (pred_index < MAX_PRED_COUNT) && (NULL != predicates[pred_index]);
         pred_index++ ) {
        if( EXPR_SUCCESS == expr_eval(predicates[pred_index],
                                      local_context, MAX_LOCAL_COUNT,
                                      &pred_val) ) {
            if( 0 == pred_val ) {
                /* This particular value doesn't fit. Return */
                valid_value = 0;
                break;
            }
        }
    }
    
    return ( 1 == valid_value ? EXPR_SUCCESS : EXPR_FAILURE_CANNOT_EVALUATE_RANGE);
}

int dplasma_symbol_get_absolute_minimum_value( const symbol_t* symbol,
                                               int* pvalue )
{
    int rc, min_min, min_max;

    rc = expr_absolute_range( symbol->min, &min_min, &min_max );
    *pvalue = min_min;
    return rc;
}

int dplasma_symbol_get_absolute_maximum_value( const symbol_t* symbol,
                                               int* pvalue )
{
    int rc, max_min, max_max;

    rc = expr_absolute_range( symbol->max, &max_min, &max_max );
    *pvalue = max_max;
    return rc;
}
