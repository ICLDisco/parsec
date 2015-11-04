/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_INTERNAL_H_HAS_BEEN_INCLUDED
#define DAGUE_INTERNAL_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "dague.h"
#include "dague/types.h"
#include "dague/class/list_item.h"
#include "dague/class/hash_table.h"
#include "dague/dague_description_structures.h"
#include "dague/profiling.h"

/**
 * A classical way to find the container that contains a particular structure.
 * Read more at http://en.wikipedia.org/wiki/Offsetof.
 */
#define container_of(ptr, type, member) \
    ((type *)((char *)ptr - offsetof(type,member)))

BEGIN_C_DECLS

typedef struct dague_function_s        dague_function_t;
typedef struct dague_remote_deps_s     dague_remote_deps_t;
typedef struct dague_arena_chunk_s     dague_arena_chunk_t;
typedef struct dague_data_pair_s       dague_data_pair_t;
typedef struct dague_dependencies_s    dague_dependencies_t;
typedef struct data_repo_s             data_repo_t;

/**< The most basic execution flow. Each virtual process includes
 *   multiple execution units (posix threads + local data) */
//typedef struct dague_execution_unit_s  dague_execution_unit_t;
/**< Each MPI process includes multiple virtual processes (and a
 *   single comm. thread) */
typedef struct dague_vp_s              dague_vp_t;
/* The description of the content of each data mouvement/copy */
typedef struct dague_dep_data_description_s  dague_dep_data_description_t;

typedef void (*dague_startup_fn_t)(dague_context_t *context,
                                   dague_handle_t *dague_handle,
                                   dague_execution_context_t** startup_list);
typedef void (*dague_destruct_fn_t)(dague_handle_t* dague_handle);

struct dague_handle_s {
    dague_list_item_t             super;
    /** All dague_handle_t structures hold these two arrays **/
    uint32_t                   handle_id;
    volatile uint32_t          nb_local_tasks;
    uint32_t                   nb_functions;
    int32_t                    priority;
    uint32_t                   devices_mask;
    dague_context_t           *context;
    dague_startup_fn_t         startup_hook;
    const dague_function_t**   functions_array;
#if defined(DAGUE_PROF_TRACE)
    const int*                 profiling_array;
#endif  /* defined(DAGUE_PROF_TRACE) */
    /* A set of callbacks at critical moments in the lifetime of a handle:
     * enqueue and completion. The enqueue is called when the handle is
     * enqueue into a context, while the completion is triggered when all
     * the tasks associated with a particular dague handle have been completed.
     */
    dague_event_cb_t           on_enqueue;
    void*                      on_enqueue_data;
    dague_event_cb_t           on_complete;
    void*                      on_complete_data;
    dague_destruct_fn_t        destructor;
    dague_dependencies_t**     dependencies_array;
    data_repo_t**              repo_array;
};

DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_handle_t);

#define DAGUE_DEVICES_ALL				   UINT32_MAX

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
#define DAGUE_DEPENDENCIES_STARTUP_TASK   ((dague_dependency_t)(1<<29))
#define DAGUE_DEPENDENCIES_BITMASK        (~(DAGUE_DEPENDENCIES_TASK_DONE|DAGUE_DEPENDENCIES_IN_DONE|DAGUE_DEPENDENCIES_STARTUP_TASK))

typedef union {
    dague_dependency_t    dependencies[1];
    dague_dependencies_t* next[1];
} dague_dependencies_union_t;

struct dague_dependencies_s {
    int                     flags;
    const symbol_t*         symbol;
    int                     min;
    int                     max;
    dague_dependencies_t* prev;
    /* keep this as the last field in the structure */
    dague_dependencies_union_t u;
};

void dague_destruct_dependencies(dague_dependencies_t* d);

/**
 * Functions for DAG manipulation.
 */
typedef enum  {
    DAGUE_ITERATE_STOP,
    DAGUE_ITERATE_CONTINUE
} dague_ontask_iterate_t;

typedef int (dague_release_deps_t)(struct dague_execution_unit_s*,
                                   dague_execution_context_t*,
                                   uint32_t,
                                   dague_remote_deps_t*);
#if defined(DAGUE_SIM)
typedef int (dague_sim_cost_fct_t)(const dague_execution_context_t *exec_context);
#endif

/**
 *
 */
typedef dague_ontask_iterate_t (dague_ontask_function_t)(struct dague_execution_unit_s *eu,
                                                         const dague_execution_context_t *newcontext,
                                                         const dague_execution_context_t *oldcontext,
                                                         const dep_t* dep,
                                                         dague_dep_data_description_t *data,
                                                         int rank_src, int rank_dst, int vpid_dst,
                                                         void *param);
/**
 *
 */
typedef void (dague_traverse_function_t)(struct dague_execution_unit_s *,
                                         const dague_execution_context_t *,
                                         uint32_t,
                                         dague_ontask_function_t *,
                                         void *);

/**
 *
 */
typedef uint64_t (dague_functionkey_fn_t)(const dague_handle_t *dague_handle,
                                          const assignment_t *assignments);
/**
 *
 */
typedef float (dague_evaluate_function_t)(const dague_execution_context_t* task);

/**
 *
 */
typedef enum dague_hook_return_e {
    DAGUE_HOOK_RETURN_DONE    =  0,  /* This execution succeeded */
    DAGUE_HOOK_RETURN_AGAIN   = -1,  /* Reschedule later */
    DAGUE_HOOK_RETURN_NEXT    = -2,  /* Try next variant [if any] */
    DAGUE_HOOK_RETURN_DISABLE = -3,  /* Disable the device, something went wrong */
    DAGUE_HOOK_RETURN_ASYNC   = -4,  /* The task is outside our reach, the completion will
                                      * be triggered asynchronously. */
    DAGUE_HOOK_RETURN_ERROR   = -5,  /* Some other major error happened */
} dague_hook_return_t;
typedef dague_hook_return_t (dague_hook_t)(struct dague_execution_unit_s*, dague_execution_context_t*);

/**
 *
 */
typedef struct dague_data_ref_s {
    struct dague_ddesc_s *ddesc;
    dague_data_key_t key;
} dague_data_ref_t;

typedef int (dague_data_ref_fn_t)(dague_execution_context_t *exec_context,
                                  dague_data_ref_t *ref);

/**
 *
 */
typedef int (dague_task_fct_t)(dague_execution_context_t *exec_context);

#define DAGUE_HAS_IN_IN_DEPENDENCIES     0x0001
#define DAGUE_HAS_OUT_OUT_DEPENDENCIES   0x0002
#define DAGUE_HAS_IN_STRONG_DEPENDENCIES 0x0004
#define DAGUE_HIGH_PRIORITY_TASK         0x0008
#define DAGUE_IMMEDIATE_TASK             0x0010
#define DAGUE_USE_DEPS_MASK              0x0020
#define DAGUE_HAS_CTL_GATHER             0X0040

typedef struct __dague_internal_incarnation_s {
    int32_t                    type;
    dague_evaluate_function_t *evaluate;
    dague_hook_t              *hook;
    char                      *dyld;
    dague_hook_t              *dyld_fn;
} __dague_chore_t;

struct dague_function_s {
    const char                  *name;

    uint16_t                     flags;
    uint8_t                      function_id;  /**< index in the dependency and in the function array */

    uint8_t                      nb_flows;
    uint8_t                      nb_parameters;
    uint8_t                      nb_locals;

    dague_dependency_t           dependencies_goal;
    const symbol_t              *params[MAX_LOCAL_COUNT];
    const symbol_t              *locals[MAX_LOCAL_COUNT];
    const dague_flow_t          *in[MAX_PARAM_COUNT];
    const dague_flow_t          *out[MAX_PARAM_COUNT];
    const expr_t                *priority;

    dague_data_ref_fn_t         *initial_data;   /**< Populates an array of data references, of maximal size MAX_PARAM_COUNT */
    dague_data_ref_fn_t         *final_data;     /**< Populates an array of data references, of maximal size MAX_PARAM_COUNT */
    dague_data_ref_fn_t         *data_affinity;  /**< Populates an array of data references, of size 1 */
    dague_functionkey_fn_t      *key;
#if defined(DAGUE_SIM)
    dague_sim_cost_fct_t        *sim_cost_fct;
#endif
    dague_hook_t                *prepare_input;
    const __dague_chore_t       *incarnations;
    dague_hook_t                *prepare_output;

    dague_traverse_function_t   *iterate_successors;
    dague_traverse_function_t   *iterate_predecessors;
    dague_release_deps_t        *release_deps;
    dague_hook_t                *complete_execution;
    dague_hook_t                *pushback;
    dague_hook_t                *fini;
};

struct dague_data_pair_s {
    struct data_repo_entry_s    *data_repo;
    struct dague_data_copy_s    *data_in;
    struct dague_data_copy_s    *data_out;
};

/**
 * Description of the state of the task. It indicates what will be the next
 * next stage in the life-time of a task to be executed.
 */
#define DAGUE_TASK_STATUS_NONE           0x00
#define DAGUE_TASK_STATUS_PREPARE_INPUT  0x01
#define DAGUE_TASK_STATUS_EVAL           0x02
#define DAGUE_TASK_STATUS_HOOK           0x03
#define DAGUE_TASK_STATUS_PREPARE_OUTPUT 0x04
#define DAGUE_TASK_STATUS_COMPLETE       0x05

/**
 * The minimal execution context contains only the smallest amount of information
 * required to be able to flow through the execution graph, by following data-flow
 * from one task to another. As an example, it contains the local variables but
 * not the data pairs. We need this in order to be able to only copy the minimal
 * amount of information when a new task is constructed.
 */
#define DAGUE_MINIMAL_EXECUTION_CONTEXT              \
    dague_hashtable_item_t         super;            \
    dague_handle_t                *dague_handle;     \
    const  dague_function_t       *function;         \
    int32_t                        priority;         \
    uint8_t                        status;           \
    uint8_t                        hook_id;          \
    uint8_t                        chore_id;         \
    uint8_t                        unused;

struct dague_minimal_execution_context_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    /* WARNING: The following locals field must ABSOLUTELY stay contiguous with
     * prof_info so that the locals are part of the event specific infos */
    assignment_t               locals[MAX_LOCAL_COUNT];
};

struct dague_execution_context_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    /* WARNING: The following locals field must ABSOLUTELY stay contiguous with
     * prof_info so that the locals are part of the event specific infos */
    assignment_t               locals[MAX_LOCAL_COUNT];
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
    int                        execution_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    dague_data_pair_t          data[MAX_PARAM_COUNT];
};
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_execution_context_t);

#define DAGUE_COPY_EXECUTION_CONTEXT(dest, src) \
    do {                                                                \
        /* this should not be copied over from the old execution context */ \
        dague_thread_mempool_t *_mpool = (dest)->super.mempool_owner;         \
        /* we copy everything but the dague_list_item_t at the beginning, to \
         * avoid copying uninitialized stuff from the stack             \
         */                                                             \
        memcpy( ((char*)(dest)) + sizeof(dague_list_item_t),            \
                ((char*)(src)) + sizeof(dague_list_item_t),             \
                sizeof(struct dague_minimal_execution_context_s) - sizeof(dague_list_item_t) ); \
        (dest)->super.mempool_owner = _mpool;                                 \
    } while (0)

/**
 * Profiling data.
 */
#if defined(DAGUE_PROF_TRACE)

extern int schedule_poll_begin, schedule_poll_end;
extern int schedule_push_begin, schedule_push_end;
extern int schedule_sleep_begin, schedule_sleep_end;
extern int queue_add_begin, queue_add_end;
extern int queue_remove_begin, queue_remove_end;
extern int device_delegate_begin, device_delegate_end;

#define DAGUE_PROF_FUNC_KEY_START(dague_handle, function_index) \
    (dague_handle)->profiling_array[2 * (function_index)]
#define DAGUE_PROF_FUNC_KEY_END(dague_handle, function_index) \
    (dague_handle)->profiling_array[1 + 2 * (function_index)]

#define DAGUE_TASK_PROF_TRACE(PROFILE, KEY, TASK)                       \
    DAGUE_PROFILING_TRACE((PROFILE),                                    \
                          (KEY),                                        \
                          (TASK)->function->key((TASK)->dague_handle, (TASK)->locals), \
                          (TASK)->dague_handle->handle_id, (void*)&(TASK)->prof_info)

#define DAGUE_TASK_PROF_TRACE_IF(COND, PROFILE, KEY, TASK)   \
    if(!!(COND)) {                                           \
        DAGUE_TASK_PROF_TRACE((PROFILE), (KEY), (TASK));     \
    }
#else
#define DAGUE_TASK_PROF_TRACE(PROFILE, KEY, TASK)
#define DAGUE_TASK_PROF_TRACE_IF(COND, PROFILE, KEY, TASK)
#endif  /* defined(DAGUE_PROF_TRACE) */


/**
 * Dependencies management.
 */
typedef struct {
    uint32_t action_mask;
    uint32_t output_usage;
    struct data_repo_entry_s *output_entry;
    dague_execution_context_t** ready_lists;
#if defined(DISTRIBUTED)
    struct dague_remote_deps_s *remote_deps;
#endif
} dague_release_dep_fct_arg_t;

dague_ontask_iterate_t dague_release_dep_fct(struct dague_execution_unit_s *eu,
                                             const dague_execution_context_t *newcontext,
                                             const dague_execution_context_t *oldcontext,
                                             const dep_t* dep,
                                             dague_dep_data_description_t* data,
                                             int rank_src, int rank_dst, int vpid_dst,
                                             void *param);

/** deps is an array of size MAX_PARAM_COUNT
 *  Returns the number of output deps on which there is a final output
 */
int dague_task_deps_with_final_output(const dague_execution_context_t *task,
                                      const dep_t **deps);

void dague_dependencies_mark_task_as_startup(dague_execution_context_t* exec_context);

int dague_release_local_OUT_dependencies(dague_execution_unit_t* eu_context,
                                         const dague_execution_context_t* origin,
                                         const dague_flow_t* origin_flow,
                                         const dague_execution_context_t* exec_context,
                                         const dague_flow_t* dest_flow,
                                         struct data_repo_entry_s* dest_repo_entry,
                                         dague_dep_data_description_t* data,
                                         dague_execution_context_t** pready_list);


/**
 * This is a convenience macro for the wrapper file. Do not call this destructor
 * directly from the applications, or face memory leaks as it only release the
 * most internal structures, while leaving the datatypes and the tasks management
 * buffers untouched. Instead, from the application layer call the _Destruct.
 */
#define DAGUE_INTERNAL_HANDLE_DESTRUCT(OBJ)                            \
    do {                                                               \
        void* __obj = (void*)(OBJ);                                    \
        ((dague_handle_t*)__obj)->destructor((dague_handle_t*)__obj);  \
        (OBJ) = NULL;                                                  \
    } while (0)

#define dague_execution_context_priority_comparator offsetof(dague_execution_context_t, priority)

/**
 * Search the dague_handle_t for a function named fname, and return it if such
 * a function exists. Returns NULL otherwise.
 */
const dague_function_t* dague_find(const dague_handle_t *dague_handle, const char *fname);

#if defined(DAGUE_SIM)
int dague_getsimulationdate( dague_context_t *dague_context );
#endif

END_C_DECLS

#endif  /* DAGUE_INTERNAL_H_HAS_BEEN_INCLUDED */
