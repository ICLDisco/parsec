/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _dplasma_h
#define _dplasma_h

typedef struct dplasma dplasma_t;

#define MAX_LOCAL_COUNT  3
#define MAX_PRED_COUNT   3
#define MAX_PARAM_COUNT  3

#include "symbol.h"
#include "expr.h"
#include "params.h"
#include "dep.h"

struct dplasma {
    char      *name;
    symbol_t  *locals[MAX_LOCAL_COUNT];
    expr_t    *preds[MAX_PRED_COUNT];
    param_t   *params[MAX_PARAM_COUNT];
    char      *body;
};

/**
 * Dump the content of a dplams_t object.
 */
void dplasma_dump(const dplasma_t *d, const char *prefix);

/**
 * Dump all defined dplasma_t obejcts.
 */
void dplasma_dump_all( void );

/**
 * Add the dplasma_t object to a global list.
 */
int dplasma_push( const dplasma_t* d );

/**
 * Find a dplasma_t object by name. If no object with such a name
 * exist return NULL.
 */
const dplasma_t* dplasma_find( const char* name );

/**
 * Return the i'th dplasma_t object. If no such element exists
 * return NULL.
 */
const dplasma_t* dplasma_element_at( int i );

/**
 * Unroll a dplasma_t object.
 */
int dplasma_unroll( const dplasma_t* object );

#endif
