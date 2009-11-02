/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _dplasma_h
#define _dplasma_h

typedef struct dplasma dplasma_t;

#define MAX_LOCAL_COUNT  5
#define MAX_PRED_COUNT   5
#define MAX_PARAM_COUNT  5

#include <stdint.h>
#include "symbol.h"
#include "expr.h"
#include "params.h"
#include "dep.h"

/* There is another loop after this one. */
#define DPLASMA_DEPENDENCIES_FLAG_NEXT       0x01
/* This is the final loop */
#define DPLASMA_DEPENDENCIES_FLAG_FINAL      0x02
/* This loops array is allocated */
#define DPLASMA_DEPENDENCIES_FLAG_ALLOCATED  0x04

/* TODO: Another ugly hack. The first time the IN dependencies are
 *       checked leave a trace in order to avoid doing it again.
 */
#define DPLASMA_DEPENDENCIES_HACK_IN         0x80

typedef struct dplasma_dependencies_t dplasma_dependencies_t;
typedef union {
    unsigned char           dependencies[1];
    dplasma_dependencies_t* next[1];
} dplasma_dependencies_union_t;

struct dplasma_dependencies_t {
    int                     flags;
    symbol_t*               symbol;
    int                     min;
    int                     max;
    dplasma_dependencies_t* prev;
    /* keep this as the last field in the structure */
    dplasma_dependencies_union_t u; 
};

typedef struct dplasma_execution_context_t dplasma_execution_context_t;
typedef int (dplasma_hook_t)(const dplasma_execution_context_t*);

#define DPLASMA_HAS_IN_IN_DEPENDENCIES     0x0001
#define DPLASMA_HAS_OUT_OUT_DEPENDENCIES   0x0002
#define DPLASMA_HAS_IN_STRONG_DEPENDENCIES 0x0004
struct dplasma {
    const char*             name;
    int                     flags;
    unsigned char           dependencies_mask;
    symbol_t*               locals[MAX_LOCAL_COUNT];
    expr_t*                 preds[MAX_PRED_COUNT];
    param_t*                params[MAX_PARAM_COUNT];
    dplasma_dependencies_t* deps;
    dplasma_hook_t*         hook;
    char*                   body;
};

struct dplasma_execution_context_t {
    dplasma_t* function;
    assignment_t locals[MAX_LOCAL_COUNT];
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
 * Find a dplasma_t object by name. If no object with such a name
 * exist one will be created and automatically added to the global
 * list.
 */
dplasma_t* dplasma_find_or_create( const char* name );

/**
 * Return the i'th dplasma_t object. If no such element exists
 * return NULL.
 */
const dplasma_t* dplasma_element_at( int i );

/**
 * Unroll a dplasma_t object.
 */
int dplasma_unroll( const dplasma_t* object );

/**
 * Some others declarations.
 */
int dplasma_complete_execution( const dplasma_execution_context_t* exec_context );
int dplasma_set_initial_execution_context( dplasma_execution_context_t* exec_context );

#endif
