/*
 * Copyright (c) 2012-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#if !defined(PARSEC_CONFIG_H_HAS_BEEN_INCLUDED)
#error parsec_internal.h header should only be used after parsec_config.h has been included.
#endif  /* !defined(PARSEC_CONFIG_H_HAS_BEEN_INCLUDED) */

#ifndef PARSEC_INTERNAL_H_HAS_BEEN_INCLUDED
#define PARSEC_INTERNAL_H_HAS_BEEN_INCLUDED

#include "parsec.h"
#include "parsec/data.h"
#include "parsec/class/list_item.h"
#include "parsec/class/hash_table.h"
#include "parsec/parsec_description_structures.h"
#include "parsec/profiling.h"
#include "parsec/mempool.h"

BEGIN_C_DECLS

/**
 * @defgroup parsec_internal_runtime Internal Runtime
 * @ingroup parsec_internal
 *   The Internal Runtime Module holds all functions and data structures
 *   that allow to build the PaRSEC runtime system.
 * @{
 */

/**
 * @brief A classical way to find the container that contains a particular structure.
 *
 * @details Read more at http://en.wikipedia.org/wiki/Offsetof.
 */
#define container_of(ptr, type, member) \
    ((type *)((char *)ptr - offsetof(type,member)))

/**
 * @brief A Remote dependency
 */
typedef struct parsec_remote_deps_s     parsec_remote_deps_t;
/**
 * @brief A temporary memory allocated using the Arena system
 */
typedef struct parsec_arena_chunk_s     parsec_arena_chunk_t;
/**
 * @brief A data and its corresponding data repository entry
 */
typedef struct parsec_data_pair_s       parsec_data_pair_t;
/**
 * @brief A dependency tracking structure
 */
typedef struct parsec_dependencies_s    parsec_dependencies_t;
/**
 * @brief A data repository entry
 */
typedef struct data_repo_s             data_repo_t;
/**
 * @brief A Virtual Process
 */
typedef struct parsec_vp_s              parsec_vp_t;
/**
 * @brief A description of the content of each data mouvement/copy
 */
typedef struct parsec_dep_data_description_s  parsec_dep_data_description_t;

/**
 * @brief The prototype of startup functions
 *
 * @details Startup functions generate a list of tasks ready to execute from
 *          a PaRSEC handle
 * @param[in] context the general PaRSEC context
 * @param[inout] tp the DAG in which to look for list of startup tasks
 * @param[out] A list of tasks ready to execute
 */
typedef void (*parsec_startup_fn_t)(parsec_context_t *context,
                                   parsec_taskpool_t *tp,
                                   parsec_task_t** startup_list);
/**
 * @brief The prototype of a handle termination / destruction function
 */
typedef void (*parsec_destruct_fn_t)(parsec_taskpool_t* tp);

/**
 * @brief a PaRSEC taskpool represents an a collection of tasks (with or without their dependencies).
 *        as provided by the Domain Specific Language.
 */
struct parsec_taskpool_s {
    parsec_list_item_t         super;     /**< A PaRSEC handle is also a list_item, so it can be chained into different lists */
    uint32_t                   taskpool_id; /**< Taskpool are uniquely globally consisntently named */
    volatile int32_t           nb_tasks;  /**< A placeholder for the upper level to count (if necessary) the tasks
                                           *   in the handle. This value is checked upon each task completion by
                                           *   the runtime, to see if the handle is completed (a nb_tasks equal
                                           *   to zero signal a completed handle). However, in order to prevent
                                           *   multiple completions of the handle due to multiple tasks completing
                                           *   simultaneously, the runtime reuse this value (once set to zero), for
                                           *   internal purposes (in which case it is atomically set to
                                           *   PARSEC_RUNTIME_RESERVED_NB_TASKS).
                                           */
    uint16_t                   nb_task_classes; /**< The number of task classes defined in this handle */
    uint16_t                   devices_mask; /**< A bitmask on what devices this handle may use */
    int32_t                    initial_number_tasks; /**< Counts the number of task classes initially ready */
    int32_t                    priority;             /**< A constant used to bump the priority of tasks related to this handle */
    int32_t                    taskpool_type;
    volatile uint32_t          nb_pending_actions;  /**< Internal counter of pending actions tracking all runtime
                                                     *   activities (such as communications, data movement, and
                                                     *   so on). Also, its value is increase by one for all the tasks
                                                     *   in the handle. This extra reference will be removed upon
                                                     *   completion of all tasks.
                                                     */
    parsec_context_t           *context;   /**< The PaRSEC context on which this handle was generated */
    parsec_startup_fn_t         startup_hook; /**< Function pointer to a function that generates initial tasks */
    const parsec_task_class_t** task_classes_array; /**< Array of task classes that build this DAG */
#if defined(PARSEC_PROF_TRACE)
    const int*                   profiling_array; /**< Array of profiling keys to start/stop each of the task classes
                                                   *   The array is indexed on the same index as task_classes_array */
#endif  /* defined(PARSEC_PROF_TRACE) */
    parsec_event_cb_t           on_enqueue;      /**< Callback called when the handle is enqueued (scheduled) */
    void*                       on_enqueue_data; /**< Data to pass to on_enqueue when called */
    parsec_event_cb_t           on_complete;     /**< Callback called when the handle is completed */
    void*                       on_complete_data;/**< Data to pass to on_complete when called */
    parsec_update_ref_t         update_nb_runtime_task;
    parsec_destruct_fn_t        destructor;      /**< handle-specific destructor function */
    void**                      dependencies_array; /**< Array of multidimensional dependencies
                                                     *   Indexed on the same index as task_classes_array */
    data_repo_t**               repo_array; /**< Array of data repositories
                                             *   Indexed on the same index as functions array */
};

PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_taskpool_t);

/**
 * @brief Bitmask representing all possible devices
 */
#define PARSEC_DEVICES_ALL                                  UINT16_MAX

/**
 * @brief Dependencies flag: there is another loop after this one.
 */
#define PARSEC_DEPENDENCIES_FLAG_NEXT       0x01
/**
 * @brief Dependencies flag: This is the final loop.
 */
#define PARSEC_DEPENDENCIES_FLAG_FINAL      0x02
/**
 * @brief Dependencies flag: This loops array is allocated.
 */
#define PARSEC_DEPENDENCIES_FLAG_ALLOCATED  0x04

/* When providing user-defined functions to count the number of tasks,
 * the user can return PARSEC_UNDETERMINED_NB_TASKS to say explicitely
 * that they will call the object termination function themselves.
 */
#define PARSEC_UNDETERMINED_NB_TASKS (0x0fffffff)
#define PARSEC_RUNTIME_RESERVED_NB_TASKS (int32_t)(0xffffffff)

/* The first time the IN dependencies are
 * checked leave a trace in order to avoid doing it again.
 */
#define PARSEC_DEPENDENCIES_TASK_DONE      ((parsec_dependency_t)(1<<31))
#define PARSEC_DEPENDENCIES_IN_DONE        ((parsec_dependency_t)(1<<30))
#define PARSEC_DEPENDENCIES_STARTUP_TASK   ((parsec_dependency_t)(1<<29))
#define PARSEC_DEPENDENCIES_BITMASK        (~(PARSEC_DEPENDENCIES_TASK_DONE|PARSEC_DEPENDENCIES_IN_DONE|PARSEC_DEPENDENCIES_STARTUP_TASK))

/**
 * This structure is used internally by the parsec_dependencies_t structures
 */
typedef union {
    parsec_dependency_t    dependencies[1];
    parsec_dependencies_t* next[1];
} parsec_dependencies_union_t;

/**
 * This structure is used when dependencies are resolved as a multi-dimensional
 * array indexed by the tasks parameters
 */
struct parsec_dependencies_s {
    int                   flags;
    int                   min;
    int                   max;
    /* keep this as the last field in the structure */
    parsec_dependencies_union_t u;
};

size_t parsec_destruct_dependencies(parsec_dependencies_t* d);

/**
 * This structure is used when dependencies are resolved using a dynamic
 * hash table
 */
struct parsec_hashable_dependency_s {
    hash_table_item_t        ht_item;
    parsec_thread_mempool_t *mempool_owner;
    parsec_dependency_t      dependency;
};
typedef struct parsec_hashable_dependency_s parsec_hashable_dependency_t;

/**
 * Functions for DAG manipulation.
 */
typedef enum  {
    PARSEC_ITERATE_STOP,
    PARSEC_ITERATE_CONTINUE
} parsec_ontask_iterate_t;

typedef int (parsec_release_deps_t)(struct parsec_execution_unit_s*,
                                   parsec_task_t*,
                                   uint32_t,
                                   parsec_remote_deps_t*);
#if defined(PARSEC_SIM)
typedef int (parsec_sim_cost_fct_t)(const parsec_task_t *exec_context);
#endif

/**
 *
 */
typedef parsec_ontask_iterate_t (parsec_ontask_function_t)(struct parsec_execution_unit_s *eu,
                                                         const parsec_task_t *newcontext,
                                                         const parsec_task_t *oldcontext,
                                                         const dep_t* dep,
                                                         parsec_dep_data_description_t *data,
                                                         int rank_src, int rank_dst, int vpid_dst,
                                                         void *param);
/**
 *
 */
typedef void (parsec_traverse_function_t)(struct parsec_execution_unit_s *,
                                         const parsec_task_t *,
                                         uint32_t,
                                         parsec_ontask_function_t *,
                                         void *);

/**
 *
 */
typedef uint64_t (parsec_functionkey_fn_t)(const parsec_taskpool_t *tp,
                                          const assignment_t *assignments);
/**
 *
 */
typedef float (parsec_evaluate_function_t)(const parsec_task_t* task);

/**
 * Retrieve the datatype for each flow (for input) or dependency (for output)
 * for a particular task. This function behave as an iterator: flow_mask
 * contains the mask of dependencies or flows that should be monitored, the left
 * most bit set to 1 to indicate input flows, and to 0 for output flows. If we
 * want to extract the input flows then the mask contains a bit set to 1 for
 * each index of a flow we are interested on. For the output flows, the mask
 * contains the bits for the dependencies we need to extract the datatype. Once
 * the data structure has been updated, the flow_mask is updated to contain only
 * the remaining flows for this task. Thus, iterating until flow_mask is 0 is
 * the way to extract all the datatypes for a particular task.
 *
 * @return PARSEC_HOOK_RETURN_NEXT if the data structure has been updated (in
 * which case the function is safe to be called again), and
 * PARSEC_HOOK_RETURN_DONE otherwise (the data structure has not been updated and
 * there is no reason to call this function again for the same task.
 */
typedef int (parsec_datatype_lookup_t)(struct parsec_execution_unit_s* eu,
                                      const parsec_task_t * this_task,
                                      uint32_t * flow_mask,
                                      parsec_dep_data_description_t * data);

/**
 * Allocate a new task that matches the function_t type. This can also be a task
 * of a generic type (such as parsec_task_t).
 */
typedef int (parsec_new_task_function_t)(const parsec_task_t** task);

/**
 *
 */
typedef parsec_hook_return_t (parsec_hook_t)(struct parsec_execution_unit_s*, parsec_task_t*);

/**
 *
 */
typedef struct parsec_data_ref_s {
    struct parsec_ddesc_s *ddesc;
    parsec_data_key_t key;
} parsec_data_ref_t;

typedef int (parsec_data_ref_fn_t)(parsec_task_t *exec_context,
                                  parsec_data_ref_t *ref);

#define PARSEC_HAS_IN_IN_DEPENDENCIES     0x0001
#define PARSEC_HAS_OUT_OUT_DEPENDENCIES   0x0002
#define PARSEC_HIGH_PRIORITY_TASK         0x0008
#define PARSEC_IMMEDIATE_TASK             0x0010
#define PARSEC_USE_DEPS_MASK              0x0020
#define PARSEC_HAS_CTL_GATHER             0X0040

/**
 * Find the dependency corresponding to a given execution context.
 */
typedef parsec_dependency_t *(parsec_find_dependency_fn_t)(const parsec_taskpool_t *tp,
                                                           parsec_execution_unit_t *eu_context,
                                                           const parsec_task_t* exec_context);
parsec_dependency_t *parsec_default_find_deps(const parsec_taskpool_t *tp,
                                              parsec_execution_unit_t *eu_context,
                                              const parsec_task_t* exec_context);
parsec_dependency_t *parsec_hash_find_deps(const parsec_taskpool_t *tp,
                                           parsec_execution_unit_t *eu_context,
                                           const parsec_task_t* exec_context);

typedef struct __parsec_internal_incarnation_s {
    int32_t                     type;
    parsec_evaluate_function_t *evaluate;
    parsec_hook_t              *hook;
    char                       *dyld;
    parsec_hook_t              *dyld_fn;
} __parsec_chore_t;

struct parsec_task_class_s {
    const char                  *name;

    uint16_t                     flags;
    uint8_t                      task_class_id;  /**< index in the dependency and in the function array */

    uint8_t                      nb_flows;
    uint8_t                      nb_parameters;
    uint8_t                      nb_locals;

    parsec_dependency_t          dependencies_goal;
    const symbol_t              *params[MAX_LOCAL_COUNT];
    const symbol_t              *locals[MAX_LOCAL_COUNT];
    const parsec_flow_t         *in[MAX_PARAM_COUNT];
    const parsec_flow_t         *out[MAX_PARAM_COUNT];
    const expr_t                *priority;

    parsec_data_ref_fn_t        *initial_data;   /**< Populates an array of data references, of maximal size MAX_PARAM_COUNT */
    parsec_data_ref_fn_t        *final_data;     /**< Populates an array of data references, of maximal size MAX_PARAM_COUNT */
    parsec_data_ref_fn_t        *data_affinity;  /**< Populates an array of data references, of size 1 */
    parsec_functionkey_fn_t     *key;
#if defined(PARSEC_SIM)
    parsec_sim_cost_fct_t       *sim_cost_fct;
#endif
    parsec_datatype_lookup_t    *get_datatype;
    parsec_hook_t               *prepare_input;
    const __parsec_chore_t      *incarnations;
    parsec_hook_t               *prepare_output;

    parsec_find_dependency_fn_t *find_deps;

    parsec_traverse_function_t  *iterate_successors;
    parsec_traverse_function_t  *iterate_predecessors;
    parsec_release_deps_t       *release_deps;
    parsec_hook_t               *complete_execution;
    parsec_new_task_function_t  *new_task;
    parsec_hook_t               *release_task;
    parsec_hook_t               *fini;
};

struct parsec_data_pair_s {
    struct data_repo_entry_s     *data_repo;
    struct parsec_data_copy_s    *data_in;
    struct parsec_data_copy_s    *data_out;
};

/**
 * Global configuration variables controling the startup mechanism
 * and directly the startup speed.
 */
PARSEC_DECLSPEC extern size_t parsec_task_startup_iter;
PARSEC_DECLSPEC extern size_t parsec_task_startup_chunk;

/**
 * Global configuration variable controlling the getrusage report.
 */
PARSEC_DECLSPEC extern int parsec_want_rusage;

/**
 * Description of the state of the task. It indicates what will be the next
 * next stage in the life-time of a task to be executed.
 */
#define PARSEC_TASK_STATUS_NONE           (uint8_t)0x00
#define PARSEC_TASK_STATUS_PREPARE_INPUT  (uint8_t)0x01
#define PARSEC_TASK_STATUS_EVAL           (uint8_t)0x02
#define PARSEC_TASK_STATUS_HOOK           (uint8_t)0x03
#define PARSEC_TASK_STATUS_PREPARE_OUTPUT (uint8_t)0x04
#define PARSEC_TASK_STATUS_COMPLETE       (uint8_t)0x05

/**
 * The minimal execution context contains only the smallest amount of information
 * required to be able to flow through the execution graph, by following data-flow
 * from one task to another. As an example, it contains the local variables but
 * not the data pairs. We need this in order to be able to only copy the minimal
 * amount of information when a new task is constructed.
 */
#define PARSEC_MINIMAL_EXECUTION_CONTEXT             \
    parsec_list_item_t             super;            \
    parsec_thread_mempool_t       *mempool_owner;    \
    parsec_taskpool_t             *taskpool;         \
    const  parsec_task_class_t    *task_class;       \
    int32_t                        priority;         \
    uint8_t                        status;           \
    uint8_t                        chore_id;         \
    uint8_t                        unused[2];

struct parsec_minimal_execution_context_s {
    PARSEC_MINIMAL_EXECUTION_CONTEXT
#if defined(PARSEC_PROF_TRACE)
    parsec_profile_ddesc_info_t prof_info;
#endif /* defined(PARSEC_PROF_TRACE) */
    /* WARNING: The following locals field must ABSOLUTELY stay contiguous with
     * prof_info so that the locals are part of the event specific infos */
    assignment_t               locals[MAX_LOCAL_COUNT];
};

struct parsec_task_s{
    PARSEC_MINIMAL_EXECUTION_CONTEXT
#if defined(PARSEC_PROF_TRACE)
    parsec_profile_ddesc_info_t prof_info;
#endif /* defined(PARSEC_PROF_TRACE) */
    /* WARNING: The following locals field must ABSOLUTELY stay contiguous with
     * prof_info so that the locals are part of the event specific infos */
    assignment_t               locals[MAX_LOCAL_COUNT];
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(PARSEC_SIM)
    int                        sim_exec_date;
#endif
    parsec_data_pair_t         data[MAX_PARAM_COUNT];
};
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_task_t);

#define PARSEC_COPY_EXECUTION_CONTEXT(dest, src) \
    do {                                                                \
        /* this should not be copied over from the old execution context */ \
        parsec_thread_mempool_t *_mpool = (dest)->mempool_owner;        \
        /* we copy everything but the parsec_list_item_t at the beginning, to \
         * avoid copying uninitialized stuff from the stack             \
         */                                                             \
        memcpy( ((char*)(dest)) + sizeof(parsec_list_item_t),           \
                ((char*)(src)) + sizeof(parsec_list_item_t),            \
                sizeof(struct parsec_minimal_execution_context_s) - sizeof(parsec_list_item_t) ); \
        (dest)->mempool_owner = _mpool;                                 \
    } while (0)

/**
 * Profiling data.
 */
#if defined(PARSEC_PROF_TRACE)

extern int schedule_poll_begin, schedule_poll_end;
extern int schedule_push_begin, schedule_push_end;
extern int schedule_sleep_begin, schedule_sleep_end;
extern int queue_add_begin, queue_add_end;
extern int queue_remove_begin, queue_remove_end;
extern int device_delegate_begin, device_delegate_end;

#define PARSEC_PROF_FUNC_KEY_START(tp, tc_index) \
    (tp)->profiling_array[2 * (tc_index)]
#define PARSEC_PROF_FUNC_KEY_END(tp, tc_index) \
    (tp)->profiling_array[1 + 2 * (tc_index)]

#define PARSEC_TASK_PROF_TRACE(PROFILE, KEY, TASK)                       \
    PARSEC_PROFILING_TRACE((PROFILE),                                    \
                          (KEY),                                        \
                          (TASK)->task_class->key((TASK)->taskpool, (assignment_t *)&(TASK)->locals), \
                          (TASK)->taskpool->taskpool_id, (void*)&(TASK)->prof_info)

#define PARSEC_TASK_PROF_TRACE_IF(COND, PROFILE, KEY, TASK)   \
    if(!!(COND)) {                                           \
        PARSEC_TASK_PROF_TRACE((PROFILE), (KEY), (TASK));     \
    }
#else
#define PARSEC_TASK_PROF_TRACE(PROFILE, KEY, TASK)
#define PARSEC_TASK_PROF_TRACE_IF(COND, PROFILE, KEY, TASK)
#endif  /* defined(PARSEC_PROF_TRACE) */


/**
 * Dependencies management.
 */
typedef struct {
    uint32_t                     action_mask;
    uint32_t                     output_usage;
    struct data_repo_entry_s    *output_entry;
    parsec_task_t **ready_lists;
#if defined(DISTRIBUTED)
    struct parsec_remote_deps_s *remote_deps;
#endif
} parsec_release_dep_fct_arg_t;


/**
 * Generic function to return a task in the corresponding mempool.
 */
parsec_hook_return_t
parsec_release_task_to_mempool(parsec_execution_unit_t *eu,
                              parsec_task_t *this_task);

parsec_ontask_iterate_t parsec_release_dep_fct(struct parsec_execution_unit_s *eu,
                                             const parsec_task_t *newcontext,
                                             const parsec_task_t *oldcontext,
                                             const dep_t* dep,
                                             parsec_dep_data_description_t* data,
                                             int rank_src, int rank_dst, int vpid_dst,
                                             void *param);

/** deps is an array of size MAX_PARAM_COUNT
 *  Returns the number of output deps on which there is a final output
 */
int parsec_task_deps_with_final_output(const parsec_task_t *task,
                                      const dep_t **deps);

int parsec_ptg_update_runtime_task( parsec_taskpool_t *tp, int32_t nb_tasks );

void parsec_dependencies_mark_task_as_startup(parsec_task_t* exec_context, parsec_execution_unit_t *eu_context);

int parsec_release_local_OUT_dependencies(parsec_execution_unit_t* eu_context,
                                         const parsec_task_t* origin,
                                         const parsec_flow_t* origin_flow,
                                         const parsec_task_t* exec_context,
                                         const parsec_flow_t* dest_flow,
                                         struct data_repo_entry_s* dest_repo_entry,
                                         parsec_dep_data_description_t* data,
                                         parsec_task_t** pready_list);


/**
 * This is a convenience macro for the wrapper file. Do not call this destructor
 * directly from the applications, or face memory leaks as it only release the
 * most internal structures, while leaving the datatypes and the tasks management
 * buffers untouched. Instead, from the application layer call the _Destruct.
 */
#define PARSEC_INTERNAL_TASKPOOL_DESTRUCT(OBJ)                            \
    do {                                                               \
        void* __obj = (void*)(OBJ);                                    \
        ((parsec_taskpool_t*)__obj)->destructor((parsec_taskpool_t*)__obj);  \
        (OBJ) = NULL;                                                  \
    } while (0)

#define parsec_execution_context_priority_comparator offsetof(parsec_task_t, priority)

#if defined(PARSEC_SIM)
int parsec_getsimulationdate( parsec_context_t *parsec_context );
#endif

/** @} */

END_C_DECLS

#endif  /* PARSEC_INTERNAL_H_HAS_BEEN_INCLUDED */
