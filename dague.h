/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_H_HAS_BEEN_INCLUDED
#define DAGUE_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "debug.h"
#ifdef HAVE_HWLOC
#include "dague_hwloc.h"
#endif

typedef struct dague_t dague_t;
typedef struct dague_remote_deps_t dague_remote_deps_t;
typedef struct dague_execution_context_t dague_execution_context_t;
typedef struct dague_dependencies_t dague_dependencies_t;

#define MAX_LOCAL_COUNT  5
#define MAX_PRED_COUNT   5
#define MAX_PARAM_COUNT  5

#ifdef HAVE_PAPI
#define MAX_EVENTS 3
#endif

#include "symbol.h"
#include "expr.h"
#include "params.h"
#include "dep.h"
#include "execution_unit.h"
#include "lifo.h"
#include "datarepo.h"

/* There is another loop after this one. */
#define DAGUE_DEPENDENCIES_FLAG_NEXT       0x01
/* This is the final loop */
#define DAGUE_DEPENDENCIES_FLAG_FINAL      0x02
/* This loops array is allocated */
#define DAGUE_DEPENDENCIES_FLAG_ALLOCATED  0x04

/* TODO: Another ugly hack. The first time the IN dependencies are
 *       checked leave a trace in order to avoid doing it again.
 */
#define DAGUE_DEPENDENCIES_HACK_IN         0x80

typedef union {
    unsigned int            dependencies[1];
    dague_dependencies_t* next[1];
} dague_dependencies_union_t;

struct dague_dependencies_t {
    int                     flags;
    symbol_t*               symbol;
    int                     min;
    int                     max;
    dague_dependencies_t* prev;
    /* keep this as the last field in the structure */
    dague_dependencies_union_t u; 
};

typedef int (dague_hook_t)(struct dague_execution_unit_t*, dague_execution_context_t*);
typedef int (dague_release_deps_t)(struct dague_execution_unit_t*, const dague_execution_context_t*, int, const struct dague_remote_deps_t *, gc_data_t **data);

typedef enum  {
    DAGUE_ITERATE_STOP,
    DAGUE_ITERATE_CONTINUE
} dague_ontask_iterate_t;

typedef dague_ontask_iterate_t (dague_ontask_function_t)(struct dague_execution_unit_t *, const dague_execution_context_t *, void *);
typedef void (dague_traverse_function_t)(struct dague_execution_unit_t *, const dague_execution_context_t *, dague_ontask_function_t *, void *);

#if defined(DAGUE_CACHE_AWARE)
typedef unsigned int (dague_cache_rank_function_t)(dague_execution_context_t *exec_context, const cache_t *cache, unsigned int reward);
#endif

#define DAGUE_HAS_IN_IN_DEPENDENCIES     0x0001
#define DAGUE_HAS_OUT_OUT_DEPENDENCIES   0x0002
#define DAGUE_HAS_IN_STRONG_DEPENDENCIES 0x0004
#define DAGUE_HIGH_PRIORITY_TASK         0x0008

struct dague_t {
    const char*             name;
    uint16_t                flags;
    uint16_t                dependencies_mask;
    uint16_t                nb_locals;
    uint16_t                nb_params;
    const symbol_t*         params[MAX_LOCAL_COUNT];
    const symbol_t*         locals[MAX_LOCAL_COUNT];
    const expr_t*           pred;
    const param_t*          in[MAX_PARAM_COUNT];
    const param_t*          out[MAX_PARAM_COUNT];
    const expr_t*           priority;
    int                     deps;                  /**< This is the index of the dependency array in the __DAGUE_object_t */
#if defined(DAGUE_CACHE_AWARE)
    dague_cache_rank_function_t *cache_rank_function;
#endif
    dague_hook_t*         hook;
    dague_traverse_function_t *iterate_successors;
    dague_release_deps_t* release_deps;
    char*                 body;
};

struct dague_object;

struct dague_execution_context_t {
    dague_list_item_t list_item;
    struct dague_object *dague_object;
    const  dague_t      *function;
    int32_t      priority;
    void        *pointers[MAX_PARAM_COUNT*2];
    assignment_t locals[MAX_LOCAL_COUNT];
};

extern int DAGUE_TILE_SIZE;

#if defined(DAGUE_PROFILING)
extern int schedule_poll_begin, schedule_poll_end;
extern int schedule_push_begin, schedule_push_end;
extern int schedule_sleep_begin, schedule_sleep_end;
#endif

typedef struct dague_object {
  /** All dague_object_t structures hold these two arrays **/
  int                    nb_functions;
  const dague_t        **functions_array;
  dague_dependencies_t **dependencies_array;
} dague_object_t;

struct dague_ddesc;

typedef int (*rank_of_fct_t)(struct dague_ddesc *mat, ...);
typedef void *(*data_of_fct_t)(struct dague_ddesc *mat, ...);

typedef struct dague_ddesc {
   rank_of_fct_t rank_of;
   data_of_fct_t data_of;
   int           myrank;
} dague_ddesc_t;

#endif  /* DAGUE_H_HAS_BEEN_INCLUDED */
