/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_H_HAS_BEEN_INCLUDED
#define DAGUE_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

#include <stddef.h>

#include "debug.h"

typedef struct dague_function            dague_function_t;
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

#include "dague_description_structures.h"
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
typedef int (dague_release_deps_t)(struct dague_execution_unit*,
                                   dague_execution_context_t*,
                                   uint32_t,
                                   struct dague_remote_deps_t *);

typedef enum  {
    DAGUE_ITERATE_STOP,
    DAGUE_ITERATE_CONTINUE
} dague_ontask_iterate_t;

typedef dague_ontask_iterate_t (dague_ontask_function_t)(struct dague_execution_unit *eu, 
                                                         dague_execution_context_t *newcontext, 
                                                         dague_execution_context_t *oldcontext, 
                                                         int flow_index, int outdep_index, 
                                                         int rank_src, int rank_dst,
                                                         dague_arena_t* arena,
                                                         void *param);
typedef void (dague_traverse_function_t)(struct dague_execution_unit *,
                                         dague_execution_context_t *,
                                         uint32_t,
                                         dague_ontask_function_t *,
                                         void *);

#if defined(DAGUE_SCHED_CACHE_AWARE)
typedef unsigned int (dague_cache_rank_function_t)(dague_execution_context_t *exec_context, const struct cache_t *cache, unsigned int reward);
#endif

#define DAGUE_HAS_IN_IN_DEPENDENCIES     0x0001
#define DAGUE_HAS_OUT_OUT_DEPENDENCIES   0x0002
#define DAGUE_HAS_IN_STRONG_DEPENDENCIES 0x0004
#define DAGUE_HIGH_PRIORITY_TASK         0x0008

#if defined(DAGUE_SIM)
typedef int (dague_sim_cost_fct_t)(const dague_execution_context_t *exec_context);
#endif
typedef uint64_t (dague_functionkey_fn_t)(const dague_object_t *dague_object, const assignment_t *assignments);

struct dague_function {
    const char                  *name;
    uint16_t                     flags;
    uint16_t                     function_id;
    uint8_t                      nb_parameters;
    uint8_t                      nb_definitions;
    dague_dependency_t           dependencies_goal;
    const symbol_t              *params[MAX_LOCAL_COUNT];
    const symbol_t              *locals[MAX_LOCAL_COUNT];
    const expr_t                *pred;
    const dague_flow_t          *in[MAX_PARAM_COUNT];
    const dague_flow_t          *out[MAX_PARAM_COUNT];
    const expr_t                *priority;
    int                          deps;  /**< This is the index of the dependency array in the __DAGUE_object_t */
#if defined(DAGUE_SIM)
    dague_sim_cost_fct_t        *sim_cost_fct;
#endif
#if defined(DAGUE_SCHED_CACHE_AWARE)
    dague_cache_rank_function_t *cache_rank_function;
#endif
    dague_hook_t                *hook;
    dague_hook_t                *complete_execution;
    dague_traverse_function_t   *iterate_successors;
    dague_release_deps_t        *release_deps;
    dague_functionkey_fn_t      *key;
    char                        *body;
};

struct dague_data_pair_t {
    data_repo_entry_t   *data_repo;
    dague_arena_chunk_t *data;
#if defined(HAVE_CUDA)
    struct gpu_elem_t   *gpu_data;
#endif  /* defined(HAVE_CUDA) */
};

/**
 * The minimal execution context contains only the smallest amount of information
 * required to be able to flow through the execution graph, by following data-flow
 * from one task to another. As an example, it contains the local variables but
 * not the data pairs. We need this in order to be able to only copy the minimal
 * amount of information when a new task is constructed.
 */
#define DAGUE_MINIMAL_EXECUTION_CONTEXT                  \
    dague_list_item_t        list_item;                  \
    dague_thread_mempool_t  *mempool_owner;              \
    dague_object_t          *dague_object;               \
    const  dague_function_t *function;                   \
    int32_t                  priority;                   \
    assignment_t             locals[MAX_LOCAL_COUNT];

struct dague_minimal_execution_context_t {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
} dague_minimal_execution_context_t;

struct dague_execution_context_t {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_SIM)
    int                     sim_exec_date;
#endif
    dague_data_pair_t       data[MAX_PARAM_COUNT];
};

#if defined(DAGUE_PROF_TRACE)
extern int schedule_poll_begin, schedule_poll_end;
extern int schedule_push_begin, schedule_push_end;
extern int schedule_sleep_begin, schedule_sleep_end;
#endif

typedef void (*dague_startup_fn_t)(dague_context_t *context, 
                                   dague_object_t *dague_object,
                                   dague_execution_context_t** startup_list);
typedef int (*dague_completion_cb_t)(dague_object_t* dague_object, void*);

struct dague_object {
    /** All dague_object_t structures hold these two arrays **/
    uint32_t                   object_id;
    volatile uint32_t          nb_local_tasks;
    uint32_t                   nb_functions;
    dague_startup_fn_t         startup_hook;
    const dague_function_t**   functions_array;
#if defined(DAGUE_PROF_TRACE)
    const int*                 profiling_array;
#endif  /* defined(DAGUE_PROF_TRACE) */
    /* Completion callback. Triggered when the all tasks associated with
     * a particular dague object have been completed.
     */
    dague_completion_cb_t      complete_cb;
    void*                      complete_cb_data;
    dague_dependencies_t**     dependencies_array;
};

#if defined(DAGUE_PROF_TRACE)
#define DAGUE_PROF_FUNC_KEY_START(dague_object, function_index) \
    (dague_object)->profiling_array[2 * (function_index)]
#define DAGUE_PROF_FUNC_KEY_END(dague_object, function_index) \
    (dague_object)->profiling_array[1 + 2 * (function_index)]
#endif

void dague_destruct_dependencies(dague_dependencies_t* d);

int dague_release_local_OUT_dependencies( dague_object_t *dague_object,
                                          dague_execution_unit_t* eu_context,
                                          const dague_execution_context_t* restrict origin,
                                          const dague_flow_t* restrict origin_flow,
                                          dague_execution_context_t* restrict exec_context,
                                          const dague_flow_t* restrict dest_flow,
                                          data_repo_entry_t* dest_repo_entry,
                                          dague_execution_context_t** pready_list );

const dague_function_t* dague_find(const dague_object_t *dague_object, const char *fname);
dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[]);
int dague_fini( dague_context_t** pcontext );
int dague_enqueue( dague_context_t* context, dague_object_t* object);
int dague_progress(dague_context_t* context);
char* dague_service_to_string( const dague_execution_context_t* exec_context,
                               char* tmp,
                               size_t length );
/* Accessors to set and get the completion callback */
int dague_set_complete_callback( dague_object_t* dague_object,
                                 dague_completion_cb_t complete_cb, void* complete_data );
int dague_get_complete_callback( const dague_object_t* dague_object,
                                 dague_completion_cb_t* complete_cb, void** complete_data );
/* This must be included here for the DISTRIBUTED macro, and after many constants have been defined */
#include "remote_dep.h"

typedef struct {
    int nb_released;
    uint32_t output_usage;
    data_repo_entry_t *output_entry;
    int action_mask;
    dague_remote_deps_t *deps;
    dague_execution_context_t* ready_list;
#if defined(DISTRIBUTED)
    int remote_deps_count;
    dague_remote_deps_t *remote_deps;
#endif
} dague_release_dep_fct_arg_t;

dague_ontask_iterate_t dague_release_dep_fct(struct dague_execution_unit *eu, 
                                             dague_execution_context_t *newcontext, 
                                             dague_execution_context_t *oldcontext, 
                                             int flow_index, int outdep_index, 
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
