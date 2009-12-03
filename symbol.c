/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern char *strdup(const char *);

#include "symbol.h"
#include "dplasma.h"

extern int dplasma_lineno;

static const symbol_t** dplasma_symbol_array = NULL;
static int dplasma_symbol_array_count = 0,
    dplasma_symbol_array_size = 0;

typedef struct symb_list {
    symbol_t *s;
    char *c_name;
    struct symb_list *next;
} symb_list_t;

char *dump_c_symbol(FILE *out, const symbol_t *s, char *init_func_body, int init_func_body_size)
{
    static symb_list_t *already_dumped = NULL;
    int i;
    symb_list_t *e;
    char mn[64];
    char mm[64];
    
    /* Did we already dump this symbol (pointer-wise)? */
    for(i = 0, e=already_dumped; e != NULL; i++, e = e->next ) {
        if(e->s == s) {
            return e->c_name;
        }
    }
    
    e = (symb_list_t*)calloc(1, sizeof(symb_list_t));
    e->s = (symbol_t*)s;
    e->c_name = (char*)malloc(64);
    sprintf(e->c_name, "&symb%d", i);
    e->next = already_dumped;
    already_dumped = e;

    sprintf(mn, "%s", dump_c_expression(out, s->min, init_func_body, init_func_body_size));
    sprintf(mm, "%s", dump_c_expression(out, s->max, init_func_body, init_func_body_size));
    
    fprintf(out, "static symbol_t symb%d = { .flags = 0x%08x, .name = \"%s\", .min = %s, .max = %s };\n",
            i,
            s->flags, s->name, mn, mm);

    return e->c_name;
}

void dump_all_global_symbols_c(FILE *out, char *init_func_body, int init_func_body_size)
{
    int i;
    char whole[4096];
    int l = 0;
    l += snprintf(whole+l, 4096-l, "static symbol_t *dplasma_symbols[] = {\n");
    for(i = 0; i < dplasma_symbol_array_count; i++) {
        l += snprintf(whole+l, 4096-l, "   %s%s", dump_c_symbol(out, dplasma_symbol_array[i], init_func_body, init_func_body_size),
                      i < dplasma_symbol_array_count-1 ? ",\n" : "};\n");
    }
    fprintf(out, "%s", whole);

    fprintf(out, "\n");

    for(i = 0; i < dplasma_symbol_array_count; i++) {
        if( (dplasma_symbol_array[i]->min->flags & EXPR_FLAG_CONSTANT) &&
            (dplasma_symbol_array[i]->max->flags & EXPR_FLAG_CONSTANT) &&
            (dplasma_symbol_array[i]->min->value == dplasma_symbol_array[i]->max->value) ) {
            /* strangely enough, this should be always the case... TODO: talk with the others -- Thomas */
            fprintf(out, "int %s = %d;\n", dplasma_symbol_array[i]->name, dplasma_symbol_array[i]->min->value);
        } else {
            fprintf(out, "int %s;\n", dplasma_symbol_array[i]->name);
        }
    }
    fprintf(out, "\n");
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
            expr_dump(s->min);
            printf("}\n" );
        } else {
            printf("%s%s:%s%s = ", prefix, s->name,
                   (DPLASMA_SYMBOL_IS_GLOBAL & s->flags ? "G" : "L"),
                   (DPLASMA_SYMBOL_IS_STANDALONE & s->flags ? "S" : "D"));
            expr_dump(s->min);
            printf("\n" );
        }
    } else {
        printf("%s%s:%s%s = [", prefix, s->name,
               (DPLASMA_SYMBOL_IS_GLOBAL & s->flags ? "G" : "L"),
               (DPLASMA_SYMBOL_IS_STANDALONE & s->flags ? "S" : "D"));
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

int dplasma_add_global_symbol( const char* name, const expr_t* expr )
{
    symbol_t* symbol;

    if( NULL != dplasma_search_global_symbol(name) ) {
        DEBUG(("Symbol %d at line %d is already defined\n", name, dplasma_lineno));
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

    /* TODO: check that this symbol doesn't depend on anything except others global symbols. */
    symbol = (symbol_t*)calloc(1, sizeof(symbol_t));
    symbol->flags = DPLASMA_SYMBOL_IS_GLOBAL | DPLASMA_SYMBOL_IS_STANDALONE;
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
        /* If we fail to evaluate the expression, let's suppose we don't have
         * all the required symbols in the assignment array.
         */
        if( EXPR_SUCCESS == expr_depend_on_symbol(predicates[pred_index], symbol) ) {
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
