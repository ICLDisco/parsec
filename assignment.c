/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "assignment.h"
#include <stdlib.h>

int dplasma_find_assignment( const char* name,
                             const assignment_t* context,
                             unsigned int context_size,
                             assignment_t** where)
{
    const assignment_t* assignment;
    int i;

    for( i = 0; (i < context_size) && (NULL != context[i].sym); i++ ) {
        assignment = &context[i];
        if( 0 == strcmp(assignment->sym->name, name) ) {
            *where = (assignment_t*)assignment;
            return EXPR_SUCCESS;
        }
    }
    return EXPR_FAILURE_SYMBOL_NOT_FOUND;
}

int dplasma_add_assignment( const symbol_t* symbol,
                            assignment_t* context,
                            unsigned int context_size,
                            assignment_t** where )
{
    int i;

    for( i = 0; (i < context_size) && (NULL != context[i].sym); i++ ) {
        if( 0 == strcmp(context[i].sym->name, symbol->name) ) {
            *where = &context[i];
            return EXPR_SUCCESS;
        }
    }
    if( i < context_size ) {
        context[i].sym = (symbol_t*)symbol;
        *where = &context[i];
        return EXPR_SUCCESS;
    }
    return EXPR_FAILURE_UNKNOWN;
}
