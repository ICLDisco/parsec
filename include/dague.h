/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_H_HAS_BEEN_INCLUDED
#define DAGUE_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

#include <stdint.h>
#include <stddef.h>

#include "debug.h"
#ifdef HAVE_HWLOC
#include "dague_hwloc.h"
#endif
#if defined(DAGUE_USE_COUNTER_FOR_DEPENDENCIES)
typedef uint32_t dague_dependency_t;
#else
typedef uint32_t dague_dependency_t;
#endif

typedef struct dague_t                   dague_t;
typedef struct dague_object              dague_object_t;
typedef struct dague_remote_deps_t       dague_remote_deps_t;
typedef struct dague_execution_context_t dague_execution_context_t;
typedef struct dague_dependencies_t      dague_dependencies_t;
typedef struct dague_data_pair_t         dague_data_pair_t;

typedef void* (*dague_allocate_data_t)(size_t matrix_size);
typedef void (*dague_free_data_t)(void *data);
extern dague_allocate_data_t dague_data_allocate;
extern dague_free_data_t     dague_data_free;

#ifdef HAVE_PAPI
#define MAX_EVENTS 3
#endif

#include "symbol.h"
#include "expr.h"
#include "params.h"
#include "dep.h"
#include "execution_unit.h"
#include "mempool.h"
#include "arena.h"
#include "datarepo.h"

/* There is another loop after this one. */
#define DAGUE_DEPENDENCIES_FLAG_NEXT       0x01
/* This is the final loop */
#define DAGUE_DEPENDENCIES_FLAG_FINAL      0x02
/* This loops array is allocated */
#define DAGUE_DEPENDENCIES_FLAG_ALLOCATED  0x04

/* The first time the IN dependencies are
 *       checked leave a trace in order to avoid doing it again.
 */
#define DAGUE_DEPENDENCIES_TASK_DONE      ((dague_dependency_t)(1<<31))
#define DAGUE_DEPENDENCIES_IN_DONE        ((dague_dependency_t)(1<<30))
#define DAGUE_DEPENDENCIES_BITMASK        (~(DAGUE_DEPENDENCIES_TASK_DONE|DAGUE_DEPENDENCIES_IN_DONE))

typedef union {
    dague_dependency_t    dependencies[1];
    dague_dependencies_t* next[1];
} dague_dependencies_union_t;

struct dague_dependencies_t {
    int                     flags;
    const symbol_t*         symbol;
    int                     min;
    int                     max;
    dague_dependencies_t* prev;
    /* keep this as the last field in the structure */
    dague_dependencies_union_t u; 
};

typedef int (dague_hook_t)(struct dague_execution_unit*, dague_execution_context_t*);
typedef int (dague_release_deps_t)(struct dague_execution_unit*, dague_execution_context_t*, int, struct dague_remote_deps_t *, dague_arena_chunk_t **data);

typedef enum  {
    DAGUE_ITERATE_STOP,
    DAGUE_ITERATE_CONTINUE
} dague_ontask_iterate_t;

typedef dague_ontask_iterate_t (dague_ontask_function_t)(struct dague_execution_unit *eu, 
                                                         dague_execution_context_t *newcontext, 
                                                         dague_execution_context_t *oldcontext, 
                                                         int param_index, int outdep_index, 
                                                         int rank_src, int rank_dst,
                                                         dague_arena_t* arena,
                                                         void *param);
typedef void (dague_traverse_function_t)(struct dague_execution_unit *, dague_execution_context_t *, dague_ontask_function_t *, void *);

#if defined(DAGUE_SCHED_CACHE_AWARE)
typedef unsigned int (dague_cache_rank_function_t)(dague_execution_context_t *exec_context, const cache_t *cache, unsigned int reward);
#endif

#define DAGUE_HAS_IN_IN_DEPENDENCIES     0x0001
#define DAGUE_HAS_OUT_OUT_DEPENDENCIES   0x0002
#define DAGUE_HAS_IN_STRONG_DEPENDENCIES 0x0004
#define DAGUE_HIGH_PRIORITY_TASK         0x0008

struct dague_t {
    const char*             name;
    uint16_t                flags;
    uint16_t                function_id;
    dague_dependency_t      dependencies_goal;
    uint16_t                nb_locals;
    uint16_t                nb_params;
    const symbol_t*         params[MAX_LOCAL_COUNT];
    const symbol_t*         locals[MAX_LOCAL_COUNT];
    const expr_t*           pred;
    const param_t*          in[MAX_PARAM_COUNT];
    const param_t*          out[MAX_PARAM_COUNT];
    const expr_t*           priority;
    int                     deps;                  /**< This is the index of the dependency array in the __DAGUE_object_t */
#if defined(DAGUE_SCHED_CACHE_AWARE)
    dague_cache_rank_function_t *cache_rank_function;
#endif
    dague_hook_t*             hook;
    dague_hook_t*             complete_execution;
    dague_traverse_function_t *iterate_successors;
    dague_release_deps_t      *release_deps;
    char*                     body;
};

struct dague_data_pair_t {
    data_repo_entry_t   *data_repo;
    dague_arena_chunk_t *data;
#if defined(HAVE_CUDA)
    struct gpu_elem_t   *gpu_data;
#endif  /* defined(HAVE_CUDA) */
};

struct dague_execution_context_t {
    dague_list_item_t       list_item;
    dague_thread_mempool_t *mempool_owner;  /* Why do we need this? */
    dague_object_t         *dague_object;
    const  dague_t         *function;
    int32_t                 priority;    
    dague_data_pair_t       data[MAX_PARAM_COUNT];
    assignment_t            locals[MAX_LOCAL_COUNT];
};

#if defined(DAGUE_PROF_TRACE)
extern int schedule_poll_begin, schedule_poll_end;
extern int schedule_push_begin, schedule_push_end;
extern int schedule_sleep_begin, schedule_sleep_end;
#endif

typedef void (*dague_startup_fn_t)(dague_execution_unit_t *eu_context, 
                                   dague_object_t *dague_object,
                                   dague_execution_context_t** startup_list);

struct dague_object {
    /** All dague_object_t structures hold these two arrays **/
    uint32_t                   object_id;
    uint32_t                   nb_local_tasks;
    uint32_t                   nb_functions;
    dague_startup_fn_t         startup_hook;
    const dague_t**            functions_array;
#if defined(DAGUE_PROF_TRACE)
    const int*                 profiling_array;
#endif  /* defined(DAGUE_PROF_TRACE) */
    dague_dependencies_t**     dependencies_array;
    dague_arena_t**            arenas_array;
};

void dague_destruct_dependencies(dague_dependencies_t* d);

int dague_release_local_OUT_dependencies( dague_object_t *dague_object,
                                          dague_execution_unit_t* eu_context,
                                          const dague_execution_context_t* restrict origin,
                                          const param_t* restrict origin_param,
                                          dague_execution_context_t* restrict exec_context,
                                          const param_t* restrict dest_param,
                                          dague_execution_context_t** pready_list );
int dague_release_OUT_dependencies( const dague_object_t *dague_object,
                                    dague_execution_unit_t* eu_context,
                                    const dague_execution_context_t* restrict origin,
                                    const param_t* restrict origin_param,
                                    dague_execution_context_t* restrict exec_context,
                                    const param_t* restrict dest_param,
                                    int forward_remote );

const dague_t* dague_find(const dague_object_t *dague_object, const char *fname);
dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[]);
int dague_fini( dague_context_t** pcontext );
char* dague_service_to_string( const dague_execution_context_t* exec_context,
                               char* tmp,
                               size_t length );

/* This must be included here for the DISTRIBUTED macro, and after many constants have been defined */
#include "remote_dep.h"

typedef struct {
    int nb_released;
    uint32_t output_usage;
    data_repo_entry_t *output_entry;
    int action_mask;
    dague_remote_deps_t *deps;
    dague_arena_chunk_t **data;
    dague_execution_context_t* ready_list;
#if defined(DISTRIBUTED)
    int remote_deps_count;
    dague_remote_deps_t *remote_deps;
#endif
} dague_release_dep_fct_arg_t;

dague_ontask_iterate_t dague_release_dep_fct(struct dague_execution_unit *eu, 
                                             dague_execution_context_t *newcontext, 
                                             dague_execution_context_t *oldcontext, 
                                             int param_index, int outdep_index, 
                                             int rank_src, int rank_dst,
                                             dague_arena_t* arena,
                                             void *param);

/**< Retrieve the local object attached to a unique object id */
dague_object_t* dague_object_lookup( uint32_t object_id );
/**< Register the object with the engine. Create the unique identifier for the object */
int dague_object_register( dague_object_t* object );
/**< Start the dague execution and launch the ready tasks */
int dague_object_start( dague_object_t* object);

static inline dague_execution_context_t*
dague_list_add_single_elem_by_priority( dague_execution_context_t** list, dague_execution_context_t* elem )
{
    if( NULL == *list ) {
        DAGUE_LIST_ITEM_SINGLETON(elem);
        *list = elem;
    } else {
        dague_execution_context_t* position = *list;
        
        while( position->priority > elem->priority ) {
            position = (dague_execution_context_t*)position->list_item.list_next;
            if( position == (*list) ) break;
        }
        elem->list_item.list_next = (dague_list_item_t*)position;
        elem->list_item.list_prev = position->list_item.list_prev;
        elem->list_item.list_next->list_prev = (dague_list_item_t*)elem;
        elem->list_item.list_prev->list_next = (dague_list_item_t*)elem;
        if( (position == *list) && (position->priority < elem->priority) ) {
            *list = elem;
        }
    }
    return *list;
}

/* gdb helpers */
void dague_dump_object( dague_object_t* object );
void dague_dump_execution_context( dague_execution_context_t* exec_context );

#endif  /* DAGUE_H_HAS_BEEN_INCLUDED */
