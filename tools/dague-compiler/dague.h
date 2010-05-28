/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGuE_H_HAS_BEEN_INCLUDED
#define DAGuE_H_HAS_BEEN_INCLUDED

#include "DAGuE_config.h"
#include "debug.h"
#ifdef HAVE_HWLOC
#include "DAGuE_hwloc.h"
#endif

typedef struct DAGuE_t DAGuE_t;
typedef struct DAGuE_remote_deps_t DAGuE_remote_deps_t;
typedef struct DAGuE_execution_context_t DAGuE_execution_context_t;
typedef struct DAGuE_dependencies_t DAGuE_dependencies_t;

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
#define DAGuE_DEPENDENCIES_FLAG_NEXT       0x01
/* This is the final loop */
#define DAGuE_DEPENDENCIES_FLAG_FINAL      0x02
/* This loops array is allocated */
#define DAGuE_DEPENDENCIES_FLAG_ALLOCATED  0x04

/* TODO: Another ugly hack. The first time the IN dependencies are
 *       checked leave a trace in order to avoid doing it again.
 */
#define DAGuE_DEPENDENCIES_HACK_IN         0x80

typedef union {
    unsigned int            dependencies[1];
    DAGuE_dependencies_t* next[1];
} DAGuE_dependencies_union_t;

struct DAGuE_dependencies_t {
    int                     flags;
    symbol_t*               symbol;
    int                     min;
    int                     max;
    DAGuE_dependencies_t* prev;
    /* keep this as the last field in the structure */
    DAGuE_dependencies_union_t u; 
};

typedef int (DAGuE_hook_t)(struct DAGuE_execution_unit_t*, DAGuE_execution_context_t*);
typedef int (DAGuE_release_deps_t)(struct DAGuE_execution_unit_t*, const DAGuE_execution_context_t*, int, const struct DAGuE_remote_deps_t *, gc_data_t **data);

typedef enum  {
    DAGuE_TRAVERSE_STOP,
    DAGuE_TRAVERSE_CONTINUE
} DAGuE_ontask_iterate_t;

typedef DAGuE_ontask_iterate_t (DAGuE_ontask_function_t)(struct DAGuE_execution_unit_t *, const DAGuE_execution_context_t *, int, void *);
typedef void (DAGuE_traverse_function_t)(struct DAGuE_execution_unit_t *, const DAGuE_execution_context_t *, int, DAGuE_ontask_function_t *, void *);

#if defined(DAGuE_CACHE_AWARE)
typedef unsigned int (DAGuE_cache_rank_function_t)(DAGuE_execution_context_t *exec_context, const cache_t *cache, unsigned int reward);
#endif

#define DAGuE_HAS_IN_IN_DEPENDENCIES     0x0001
#define DAGuE_HAS_OUT_OUT_DEPENDENCIES   0x0002
#define DAGuE_HAS_IN_STRONG_DEPENDENCIES 0x0004
#define DAGuE_HIGH_PRIORITY_TASK         0x0008

struct DAGuE_t {
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
#if defined(DAGuE_CACHE_AWARE)
    DAGuE_cache_rank_function_t *cache_rank_function;
#endif
    DAGuE_hook_t*         hook;
    DAGuE_traverse_function_t *preorder;
    DAGuE_release_deps_t* release_deps;
    char*                 body;
};

struct DAGuE_object;

struct DAGuE_execution_context_t {
    DAGuE_list_item_t list_item;
    struct DAGuE_object *DAGuE_object;
    DAGuE_t*   function;
    int32_t      priority;
    void        *pointers[MAX_PARAM_COUNT*2];
    assignment_t locals[MAX_LOCAL_COUNT];
};

extern int DAGuE_TILE_SIZE;

#if defined(DAGuE_PROFILING)
extern int schedule_poll_begin, schedule_poll_end;
extern int schedule_push_begin, schedule_push_end;
extern int schedule_sleep_begin, schedule_sleep_end;
#endif

typedef struct DAGuE_object {
  /** All DAGuE_object_t structures hold these two arrays **/
  int                    nb_functions;
  const DAGuE_t        **functions_array;
  DAGuE_dependencies_t **dependencies_array;
} DAGuE_object_t;

struct DAGuE_ddesc;

typedef int (*rank_of_fct_t)(struct DAGuE_ddesc *mat, ...);
typedef void *(*data_of_fct_t)(struct DAGuE_ddesc *mat, ...);

typedef struct DAGuE_ddesc {
   rank_of_fct_t rank_of;
   data_of_fct_t data_of;
   int           myrank;
} DAGuE_ddesc_t;

#endif  /* DAGuE_H_HAS_BEEN_INCLUDED */
