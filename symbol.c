/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "symbol.h"

static symbol_t** dague_symbol_array = NULL;
static int dague_symbol_array_count = 0,
    dague_symbol_array_size = 0;

int dague_symbol_get_count(void) 
{
    return dague_symbol_array_count;
}

const symbol_t *dague_symbol_get_element_at( int i )
{
    if( i >= dague_symbol_array_count ) {
        return NULL;
    } else {
        return dague_symbol_array[i];
    }
}

void dague_load_symbols( symbol_t **array, int size )
{
    int i, j, found;

    if( size + dague_symbol_array_count > dague_symbol_array_size ) {
        dague_symbol_array_size = size + dague_symbol_array_count;
        dague_symbol_array = (symbol_t ** )realloc(dague_symbol_array, dague_symbol_array_size * sizeof(symbol_t*));
    }

    for(i = 0; i < size; i++) {
        found = 0;
        for(j = 0; j < dague_symbol_array_count; j++) {
            if( !strcmp( array[i]->name, dague_symbol_array[j]->name) ) {
                found = 1;
                break;
            }
        }
        if( 0 == found ) {
            dague_symbol_array[dague_symbol_array_count] = array[i];
            dague_symbol_array_count++;
        }
    }
}

void symbol_dump(const symbol_t *s, const struct dague_object *dague_object, const char *prefix)
{
    if( NULL == s->name ) {
        return;
    }

    if( s->min == s->max ) {
        if( EXPR_FLAG_CONSTANT & s->min->flags ) {
            printf("%s%s:%s%s = {%d = ", prefix, s->name,
                   (DAGUE_SYMBOL_IS_GLOBAL & s->flags ? "G" : "L"),
                   (DAGUE_SYMBOL_IS_STANDALONE & s->flags ? "S" : "D"), s->min->value);
            expr_dump(stdout, dague_object, s->min);
            printf("}\n" );
        } else {
            printf("%s%s:%s%s = ", prefix, s->name,
                   (DAGUE_SYMBOL_IS_GLOBAL & s->flags ? "G" : "L"),
                   (DAGUE_SYMBOL_IS_STANDALONE & s->flags ? "S" : "D"));
            expr_dump(stdout, dague_object, s->min);
            printf("\n" );
        }
    } else {
        printf("%s%s:%s%s = [", prefix, s->name,
               (DAGUE_SYMBOL_IS_GLOBAL & s->flags ? "G" : "L"),
               (DAGUE_SYMBOL_IS_STANDALONE & s->flags ? "S" : "D"));
        expr_dump(stdout, dague_object, s->min);
        printf(" .. ");
        expr_dump(stdout, dague_object, s->max);
        printf("]\n");
    }
}

void symbol_dump_all( const char* prefix, const struct dague_object *dague_object )
{
    const symbol_t* symbol;
    int i;

    for( i = 0; i < dague_symbol_array_count; i++ ) {
        symbol = dague_symbol_array[i];
        symbol_dump( symbol, dague_object, prefix );
    }
}

int symbol_c_index_lookup( const symbol_t *symbol )
{
    int i;
    for(i = 0; i < dague_symbol_array_count; i++) {
        if ( symbol == dague_symbol_array[i] ) {
            return i;
        }
    }
    return -1;
}

int dague_assign_global_symbol( const char *name, const expr_t *expr )
{
    symbol_t* symbol;

    if( NULL == (symbol = dague_search_global_symbol(name)) ) {
        DEBUG(("Cannot assign symbol %s: not defined\n", name));
        return -1;
    }

    symbol->min = expr;
    symbol->max = expr;

    return EXPR_SUCCESS;
}

int dague_add_global_symbol( const char *name )
{
    symbol_t* symbol;

    if( NULL != dague_search_global_symbol(name) ) {
        DEBUG(("Add global symbol cst: symbol %s is already defined\n", name));
        return -1;
    }

    if( dague_symbol_array_count >= dague_symbol_array_size ) {
        if( 0 == dague_symbol_array_size ) {
            dague_symbol_array_size = 4;
        } else {
            dague_symbol_array_size *= 2;
        }
        dague_symbol_array = (symbol_t**)realloc( dague_symbol_array,
                                                    dague_symbol_array_size * sizeof(symbol_t*) );
        if( NULL == dague_symbol_array ) {
            return -1;  /* No more available memory */
        }
    }

    symbol = (symbol_t*)calloc(1, sizeof(symbol_t));
    symbol->flags = DAGUE_SYMBOL_IS_GLOBAL;
    symbol->name = strdup(name);

    dague_symbol_array[dague_symbol_array_count] = symbol;
    dague_symbol_array_count++;
    return EXPR_SUCCESS;
}

int dague_add_global_symbol_cst( const char* name, const expr_t* expr )
{
    int ret;
    ret = dague_add_global_symbol( name );
    if( ret != EXPR_SUCCESS ) 
        return ret;
    return dague_assign_global_symbol( name, expr );
}

symbol_t* dague_search_global_symbol( const char* name )
{
    symbol_t* symbol;
    int i;

    for( i = 0; i < dague_symbol_array_count; i++ ) {
        symbol = dague_symbol_array[i];
        if( 0 == strcmp(symbol->name, name) ) {
            return symbol;
        }
    }
    return NULL;
}

static int dague_expr_parse_callback( const symbol_t* symbol, void* data )
{
    int* pvalue = (int*)data;

    if( !(DAGUE_SYMBOL_IS_GLOBAL & symbol->flags) ) {
        /* Allow us to count the number of local symbols in the expression */
        (*pvalue)++;
    }
    return EXPR_SUCCESS;
}

int dague_symbol_is_standalone( const symbol_t* symbol )
{
    int rc, symbols_count = 0;

    rc = expr_parse_symbols( symbol->min, &dague_expr_parse_callback, (void*)&symbols_count );
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }
    if( 0 == symbols_count ) {
        rc = expr_parse_symbols( symbol->max, &dague_expr_parse_callback, (void*)&symbols_count );
        if( EXPR_SUCCESS != rc ) {
            return rc;
        }
        return (0 == symbols_count ? EXPR_SUCCESS : EXPR_FAILURE_UNKNOWN);
    }
    return EXPR_FAILURE_UNKNOWN;
}

int dague_symbol_get_first_value( const dague_object_t *dague_object,
                                  const symbol_t* symbol,
                                  const expr_t* predicate,
                                  assignment_t* local_context,
                                  int* pvalue )
{
    assignment_t* assignment;
    int rc, min, max, val, pred_val;

    rc = expr_eval( dague_object, symbol->min, local_context, MAX_LOCAL_COUNT, &min );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the min expression for symbol %s\n", symbol->name);
        return rc;
    }
    rc = expr_eval( dague_object, symbol->max, local_context, MAX_LOCAL_COUNT, &max );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the max expression for symbol %s\n", symbol->name);
        return rc;
    }

    rc = dague_add_assignment( symbol, local_context, MAX_LOCAL_COUNT, &assignment );
    if( DAGUE_ASSIGN_ERROR == rc ) {
        /* the symbol cannot be added to the local context. Bail out */
        return rc;
    }

    /* TODO: as we only accept increasing ranges, we cannot tolerate minimum values
     * bigger than the maximum ones */
    if( min > max ) {
        return EXPR_FAILURE_UNKNOWN;
    }

    /* If there are no predicate we're good to go */
    if( NULL == predicate ) {
        assignment->value = min;
        *pvalue = min;
        return EXPR_SUCCESS;
    }

    for( val = min; val <= max; val++ ) {
        /* If this value matches the predicate, we are done */
        if( EXPR_SUCCESS == expr_eval(dague_object, predicate,
                                      local_context, MAX_LOCAL_COUNT,
                                      &pred_val) ) {
            if( 0 != pred_val ) {
                assignment->value = val;
                *pvalue = val;
                return EXPR_SUCCESS;
            }
        }
    }
    
    return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
}

int dague_symbol_get_last_value( const dague_object_t *dague_object,
                                 const symbol_t* symbol,
                                 const expr_t* predicate,
                                 assignment_t* local_context,
                                 int* pvalue )
{
    assignment_t* assignment;
    int rc, min, max, val, pred_val;

    rc = expr_eval( dague_object, symbol->min, local_context, MAX_LOCAL_COUNT, &min );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the min expression for symbol %s\n", symbol->name);
        return rc;
    }
    rc = expr_eval( dague_object, symbol->max, local_context, MAX_LOCAL_COUNT, &max );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the max expression for symbol %s\n", symbol->name);
        return rc;
    }

    rc = dague_add_assignment( symbol, local_context, MAX_LOCAL_COUNT, &assignment );
    if( DAGUE_ASSIGN_ERROR == rc ) {
        /* the symbol cannot be added to the local context. Bail out */
        return rc;
    }

    /* If there are no predicates we're good to go */
    if( NULL == predicate ) {
        assignment->value = max;
        *pvalue = max;
        return EXPR_SUCCESS;
    }

    for( val = max; val >= min; val-- ) {
        if( EXPR_SUCCESS == expr_eval(dague_object, predicate,
                                      local_context, MAX_LOCAL_COUNT,
                                      &pred_val) ) {
            if( 0 != pred_val ) {
                assignment->value = val;
                *pvalue = val;
                return EXPR_SUCCESS;
            }
        }
    }

    return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
}

int dague_symbol_get_next_value( const dague_object_t *dague_object,
                                 const symbol_t* symbol,
                                 const expr_t* predicate,
                                 assignment_t* local_context,
                                 int* pvalue )
{
    assignment_t* assignment;
    int rc, max, val, pred_val;

    rc = expr_eval( dague_object, symbol->max, local_context, MAX_LOCAL_COUNT, &max );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the max expression for symbol %s\n", symbol->name);
        return rc;
    }

    rc = dague_find_assignment( symbol->name, local_context, MAX_LOCAL_COUNT, &assignment );
    if( DAGUE_ASSIGN_ERROR == rc ) {
        /* the symbol is not yet on the assignment list, so there is ABSOLUTELY
         * no reason to ask for the next value.
         */
        return rc;
    }
    /* If there are no predicates we're good to go */
    if( NULL == predicate ) {
        val = assignment->value + 1;
        if( val <= max ) {
            *pvalue = val;
            assignment->value = val;
            return EXPR_SUCCESS;
        }
        return EXPR_FAILURE_UNKNOWN;
    }

    for( val = assignment->value + 1; val <= max; val++ ) {
        if( EXPR_SUCCESS == expr_eval(dague_object,
                                      predicate,
                                      local_context, MAX_LOCAL_COUNT,
                                      &pred_val) ) {
            if( 0 != pred_val ) {
                assignment->value = val;
                *pvalue = val;
                return EXPR_SUCCESS;
            }
        }
    }

    return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
}

int dague_symbol_validate_value( const dague_object_t *dague_object,
                                 const symbol_t* symbol,
                                 const expr_t* predicate,
                                 assignment_t* local_context )
{
    assignment_t* assignment;
    int rc, min, max, pred_val;

    rc = dague_find_assignment( symbol->name, local_context, MAX_LOCAL_COUNT, &assignment );
    if( DAGUE_ASSIGN_ERROR == rc ) {
        /* the symbol is not yet on the assignment list, so there is ABSOLUTELY
         * no reason to ask for the next value.
         */
        return rc;
    }

    rc = expr_eval( dague_object, symbol->min, local_context, MAX_LOCAL_COUNT, &min );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the min expression for symbol %s\n", symbol->name);
        return rc;
    }

    rc = expr_eval( dague_object, symbol->max, local_context, MAX_LOCAL_COUNT, &max );
    if( EXPR_SUCCESS != rc ) {
        printf(" Cannot evaluate the max expression for symbol %s\n", symbol->name);
        return rc;
    }

    /* Make sure the current value is in the legal range */
    if( (assignment->value < min) || (assignment->value > max) || (min > max) ) {
        return EXPR_FAILURE_UNKNOWN;
    }

    /* If there are no predicates we're good to go */
    if( NULL == predicate ) {
        return EXPR_SUCCESS;
    }

    if( EXPR_SUCCESS == expr_eval(dague_object, predicate,
                                  local_context, MAX_LOCAL_COUNT,
                                  &pred_val) ) {
        if( 0 == pred_val ) {
            return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
        }
        return EXPR_SUCCESS;
    }
    
    return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
}

int dague_symbol_get_absolute_minimum_value( const dague_object_t *dague_object, const symbol_t* symbol,
                                               int* pvalue )
{
    int rc, min_min, min_max;

    rc = expr_absolute_range( dague_object, symbol->min, &min_min, &min_max );
    *pvalue = min_min;
    return rc;
}

int dague_symbol_get_absolute_maximum_value( const dague_object_t *dague_object, const symbol_t* symbol,
                                               int* pvalue )
{
    int rc, max_min, max_max;

    rc = expr_absolute_range( dague_object, symbol->max, &max_min, &max_max );
    *pvalue = max_max;
    return rc;
}
