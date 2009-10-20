/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _assigment_h
#define _assigment_h

typedef struct assignment assignment_t;

#include "symbol.h"

struct assignment {
    symbol_t  *sym;
    int        value;
};

int dplasma_find_assignment( const char* name,
                             const assignment_t* context,
                             unsigned int context_size,
                             assignment_t** where);
int dplasma_add_assignment( const symbol_t* symbol,
                            assignment_t* context,
                            unsigned int context_size,
                            assignment_t** where );

#endif
