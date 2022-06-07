/*
 * Copyright (c) 2012-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#if !defined(PARSEC_CONFIG_H_HAS_BEEN_INCLUDED)
#error parsec_internal.h header should only be used after parsec_config.h has been included.
#endif  /* !defined(PARSEC_CONFIG_H_HAS_BEEN_INCLUDED) */

#ifndef PARSEC_INTERNAL_H_HAS_BEEN_INCLUDED
#define PARSEC_INTERNAL_H_HAS_BEEN_INCLUDED

#include "parsec/runtime.h"
#include "parsec/data_internal.h"
#include "parsec/class/list_item.h"
#include "parsec/class/parsec_hash_table.h"
#include "parsec/parsec_description_structures.h"
#include "parsec/profiling.h"
#include "parsec/mempool.h"
#include "parsec/arena.h"
#include "parsec/datarepo.h"
#include "parsec/data.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"
#include "parsec/class/info.h"
#include "parsec/mca/pins/pins.h"

#if defined(PARSEC_PROF_GRAPHER)
#include "parsec/parsec_prof_grapher.h"
#endif  /* defined(PARSEC_PROF_GRAPHER) */
#include "parsec/mca/device/device.h"

#include <string.h>

BEGIN_C_DECLS

/**
 * Arena-datatype management.
 */
struct parsec_arena_datatype_s {
    parsec_arena_t           *arena;       /**< allocator for this datatype */
    parsec_datatype_t         opaque_dtt;  /**< datatype */
    parsec_hash_table_item_t  ht_item;     /**< sometimes, arena datatype are stored in hash tables */
};

int parsec_arena_datatype_construct(parsec_arena_datatype_t *adt,
                                   size_t elem_size,
                                   size_t alignment,
                                   parsec_datatype_t opaque_dtt);

/* NULL terminated local hostname of the current PaRSEC process */
PARSEC_DECLSPEC extern const char* parsec_hostname;

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
 * @brief A Virtual Process
 */
typedef struct parsec_vp_s              parsec_vp_t;

/**
 * @brief The prototype of startup functions
 *
 * @details Startup functions generate a list of tasks ready to execute from
 *          a PaRSEC taskpool
 * @param[in] context the general PaRSEC context
 * @param[inout] tp the DAG in which to look for list of startup tasks
 * @param[out] A list of tasks ready to execute
 */
typedef void (*parsec_startup_fn_t)(parsec_context_t *context,
                                   parsec_taskpool_t *tp,
                                   parsec_task_t** startup_list);
/**
 * @brief The prototype of a taskpool termination / destruction function
 */
typedef void (*parsec_destruct_fn_t)(parsec_taskpool_t* tp);

/**
 * Types of known taskpools. This should be extended as new types of taskpools are
 * to PaRSEC.
 */
#define PARSEC_TASKPOOL_TYPE_PTG       0x0001
#define PARSEC_TASKPOOL_TYPE_COMPOUND  0x0002
#define PARSEC_TASKPOOL_TYPE_DTD       0x0004

/**
 * @brief a PaRSEC taskpool represents an a collection of tasks (with or without their dependencies).
 *        as provided by the Domain Specific Language.
 */
struct parsec_taskpool_s {
    parsec_list_item_t         super;     /**< A PaRSEC taskpool is also a list_item, so it can be chained into different lists */
    uint32_t                   taskpool_id; /**< Taskpool are uniquely globally consistently named */
    char*                      taskpool_name; /**< Taskpools are not uniquely named for profiling */
    volatile int32_t           nb_tasks;  /**< A placeholder for the upper level to count (if necessary) the tasks
                                           *   in the taskpool. This value is checked upon each task completion by
                                           *   the runtime, to see if the taskpool is completed (a nb_tasks equal
                                           *   to zero signal a completed taskpool). However, in order to prevent
                                           *   multiple completions of the taskpool due to multiple tasks completing
                                           *   simultaneously, the runtime reuse this value (once set to zero), for
                                           *   internal purposes (in which case it is atomically set to
                                           *   PARSEC_RUNTIME_RESERVED_NB_TASKS).
                                           */
    int16_t                    taskpool_type;
    uint16_t                   devices_index_mask; /**< A bitmask of devices indexes this taskpool has been registered with */
    uint32_t                   nb_task_classes;      /**< Number of task classes in the taskpool */
    int32_t                    priority;             /**< A constant used to bump the priority of tasks related to this taskpool */
    volatile int32_t           nb_pending_actions;  /**< Internal counter of pending actions tracking all runtime
                                                     *   activities (such as communications, data movement, and
                                                     *   so on). Also, its value is increase by one for all the tasks
                                                     *   in the taskpool. This extra reference will be removed upon
                                                     *   completion of all tasks.
                                                     */
    parsec_context_t*           context;   /**< The PaRSEC context on which this taskpool was enqueued */
    parsec_startup_fn_t         startup_hook;  /**< Pointer to the function that generates initial tasks */
    const parsec_task_class_t** task_classes_array; /**< Array of task classes that build this DAG */
#if defined(PARSEC_PROF_TRACE)
    const int*                  profiling_array; /**< Array of profiling keys to start/stop each of the task classes
                                                  *   The array is indexed on the same index as task_classes_array */
#endif  /* defined(PARSEC_PROF_TRACE) */
    parsec_event_cb_t           on_enqueue;      /**< Callback called when the taskpool is enqueued (scheduled) */
    void*                       on_enqueue_data; /**< Data to pass to on_enqueue when called */
    parsec_event_cb_t           on_complete;     /**< Callback called when the taskpool is completed */
    void*                       on_complete_data;/**< Data to pass to on_complete when called */
    parsec_update_ref_t         update_nb_runtime_task;
    void**                      dependencies_array; /**< Array of multidimensional dependencies
                                                     *   Indexed on the same index as task_classes_array */
    data_repo_t**               repo_array; /**< Array of data repositories
                                             *   Indexed on the same index as functions array */
};

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_taskpool_t);

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
    parsec_hash_table_item_t  ht_item;
    parsec_thread_mempool_t  *mempool_owner;
    parsec_dependency_t       dependency;
};
typedef struct parsec_hashable_dependency_s parsec_hashable_dependency_t;

/**
 * Functions for DAG manipulation.
 */
typedef enum  {
    PARSEC_ITERATE_STOP,
    PARSEC_ITERATE_CONTINUE
} parsec_ontask_iterate_t;

typedef int (parsec_release_deps_t)(struct parsec_execution_stream_s*,
                                   parsec_task_t*,
                                   uint32_t,
                                   parsec_remote_deps_t*);
#if defined(PARSEC_SIM)
typedef int (parsec_sim_cost_fct_t)(const parsec_task_t *task);
#endif

/**
 *
 */
typedef parsec_ontask_iterate_t (parsec_ontask_function_t)(struct parsec_execution_stream_s* es,
                                                         const parsec_task_t *newcontext,
                                                         const parsec_task_t *oldcontext,
                                                         const parsec_dep_t* dep,
                                                         parsec_dep_data_description_t *data,
                                                         int rank_src, int rank_dst, int vpid_dst,
                                                         data_repo_t *successor_repo, parsec_key_t successor_repo_key,
                                                         void *param);
/**
 *
 */
typedef void (parsec_traverse_function_t)(struct parsec_execution_stream_s *,
                                         const parsec_task_t *,
                                         uint32_t,
                                         parsec_ontask_function_t *,
                                         void *);

/**
 * Returns the key associated with the task
 */
typedef parsec_key_t (parsec_functionkey_fn_t)(const parsec_taskpool_t *tp,
                                               const parsec_assignment_t *assignments);
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
typedef int (parsec_datatype_lookup_t)(struct parsec_execution_stream_s* es,
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
typedef parsec_hook_return_t (parsec_hook_t)(struct parsec_execution_stream_s*, parsec_task_t*);

/**
 *
 */
typedef struct parsec_data_ref_s {
    struct parsec_data_collection_s *dc;
    parsec_data_key_t key;
} parsec_data_ref_t;

typedef int (parsec_data_ref_fn_t)(parsec_task_t *task,
                                  parsec_data_ref_t *ref);

#define PARSEC_HAS_IN_IN_DEPENDENCIES     0x0001
#define PARSEC_HAS_OUT_OUT_DEPENDENCIES   0x0002
#define PARSEC_HIGH_PRIORITY_TASK         0x0008
#define PARSEC_IMMEDIATE_TASK             0x0010
#define PARSEC_USE_DEPS_MASK              0x0020
#define PARSEC_HAS_CTL_GATHER             0X0040

#define PARSEC_TASK_CLASS_TYPE_PTG        0x01
#define PARSEC_TASK_CLASS_TYPE_DTD        0x02

/**
 * Find the dependency corresponding to a given execution context.
 */
typedef parsec_dependency_t *(parsec_find_dependency_fn_t)(const parsec_taskpool_t *tp,
                                                           parsec_execution_stream_t *es,
                                                           const parsec_task_t* task);
parsec_dependency_t *parsec_default_find_deps(const parsec_taskpool_t *tp,
                                              parsec_execution_stream_t *es,
                                              const parsec_task_t* task);
parsec_dependency_t *parsec_hash_find_deps(const parsec_taskpool_t *tp,
                                           parsec_execution_stream_t *es,
                                           const parsec_task_t* task);

typedef struct __parsec_internal_incarnation_s {
    int32_t                     type;
    parsec_evaluate_function_t *evaluate;
    parsec_hook_t              *hook;
    char                       *dyld;
    parsec_hook_t              *dyld_fn;
} __parsec_chore_t;

typedef struct parsec_property_s {
    const char   *name;
    const parsec_expr_t *expr;
} parsec_property_t;

struct parsec_task_class_s {
    const char                  *name;

    uint16_t                     flags;
    uint8_t                      task_class_id;  /**< index in the dependency and in the function array */

    uint8_t                      nb_flows;
    uint8_t                      nb_parameters;
    uint8_t                      nb_locals;

    uint8_t                      task_class_type;

    parsec_dependency_t          dependencies_goal;
    const parsec_symbol_t       *params[MAX_LOCAL_COUNT];
    const parsec_symbol_t       *locals[MAX_LOCAL_COUNT];
    const parsec_flow_t         *in[MAX_PARAM_COUNT];
    const parsec_flow_t         *out[MAX_PARAM_COUNT];
    const parsec_expr_t         *priority;
    const parsec_property_t     *properties;     /**< {NULL, NULL} terminated array of properties holding all function-specific properties expressions */

    parsec_data_ref_fn_t        *initial_data;   /**< Populates an array of data references, of maximal size MAX_PARAM_COUNT */
    parsec_data_ref_fn_t        *final_data;     /**< Populates an array of data references, of maximal size MAX_PARAM_COUNT */
    parsec_data_ref_fn_t        *data_affinity;  /**< Populates an array of data references, of size 1 */
    parsec_key_fn_t             *key_functions;
    parsec_functionkey_fn_t     *make_key;
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
    struct parsec_data_copy_s    *data_in;
    struct parsec_data_copy_s    *data_out;
    struct data_repo_entry_s     *source_repo_entry; /* Data repo entry from which the input data is consumed.
                                                      * Setup on during data lookup and used during release deps. */
    data_repo_t                  *source_repo;       /* Data repo from which the input data is consumed.
                                                      * Setup on during data lookup and used during release deps. */
    int                           fulfill;     /* flag used during data lookup to indicated that the input data of the task
                                                * has already been reshaped.
                                                * Necessary as data lookup can return ASYNC when reshaping each input flow. */
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
    uint8_t                        chore_mask;       \
    uint8_t                        unused[2];        \
    struct data_repo_entry_s      *repo_entry; /* The task contains its own data repo entry;
                                                * It is created during datalookup if it hasn't
                                                * been already created by a predecessor
                                                */


struct parsec_minimal_execution_context_s {
    PARSEC_MINIMAL_EXECUTION_CONTEXT
#if defined(PARSEC_PROF_TRACE)
    parsec_task_prof_info_t         prof_info;
#endif /* defined(PARSEC_PROF_TRACE) */
    /* WARNING: The following locals field must ABSOLUTELY stay contiguous with
     * prof_info so that the locals are part of the event specific infos */
    parsec_assignment_t             locals[MAX_LOCAL_COUNT];
};

struct parsec_task_s {
    PARSEC_MINIMAL_EXECUTION_CONTEXT
#if defined(PARSEC_PROF_TRACE)
    parsec_task_prof_info_t    prof_info;
#endif /* defined(PARSEC_PROF_TRACE) */
    /* WARNING: The following locals field must ABSOLUTELY stay contiguous with
     * prof_info so that the locals are part of the event specific infos */
    parsec_assignment_t        locals[MAX_LOCAL_COUNT];
#if defined(PARSEC_SIM)
    int                        sim_exec_date;
#endif
    parsec_data_pair_t         data[MAX_PARAM_COUNT];
};
PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(parsec_task_t);

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
        (dest)->repo_entry = NULL;                                      \
    } while (0)

/**
 * Profiling data.
 */
#if defined(PARSEC_PROF_TRACE)

extern int schedule_poll_begin, schedule_poll_end;
extern int schedule_push_begin, schedule_push_end;
extern int schedule_sleep_begin, schedule_sleep_end;
extern int queue_remove_begin, queue_remove_end;
extern int device_delegate_begin, device_delegate_end;

#define PARSEC_PROF_FUNC_KEY_START(tp, tc_index) \
    (tp)->profiling_array[2 * (tc_index)]
#define PARSEC_PROF_FUNC_KEY_END(tp, tc_index) \
    (tp)->profiling_array[1 + 2 * (tc_index)]

#define PARSEC_TASK_PROF_TRACE(PROFILE, KEY, TASK)                      \
    PARSEC_PROFILING_TRACE((PROFILE),                                   \
                           (KEY),                                       \
                           (TASK)->task_class->key_functions->          \
                           key_hash((TASK)->task_class->make_key(       \
                              (TASK)->taskpool, (TASK)->locals ), NULL), \
                              (TASK)->taskpool->taskpool_id, (void*)&(TASK)->prof_info); 

#define PARSEC_TASK_PROF_TRACE_IF(COND, PROFILE, KEY, TASK)  \
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
    data_repo_entry_t           *output_entry; /* Current task repo entry */
    data_repo_t                 *output_repo;  /* Current task repo */
                                               /* Both entry and repo need to be passed to the callbacks
                                                * of iterate successors during release deps of. The entry
                                                * is set as input on the successor ready tasks during
                                                * local release, and the repo also so the successors
                                                * consume appropriately.
                                                */
    parsec_task_t              **ready_lists;
#if defined(DISTRIBUTED)
    struct parsec_remote_deps_s *remote_deps;
#endif
} parsec_release_dep_fct_arg_t;


/**
 * Generic function to return a task in the corresponding mempool.
 */
parsec_hook_return_t
parsec_release_task_to_mempool(parsec_execution_stream_t *es,
                              parsec_task_t *this_task);

parsec_ontask_iterate_t
parsec_release_dep_fct(struct parsec_execution_stream_s *es,
                       const parsec_task_t *newcontext,
                       const parsec_task_t *oldcontext,
                       const parsec_dep_t* dep,
                       parsec_dep_data_description_t* data,
                       int rank_src, int rank_dst, int vpid_dst,
                       data_repo_t *successor_repo, parsec_key_t successor_repo_key,
                       void *param);

/**
 * Function to create reshaping promises during iterate_succesors.
 */
parsec_ontask_iterate_t
parsec_set_up_reshape_promise(parsec_execution_stream_t *es,
                              const parsec_task_t *newcontext,
                              const parsec_task_t *oldcontext,
                              const parsec_dep_t* dep,
                              parsec_dep_data_description_t* data,
                              int src_rank, int dst_rank, int dst_vpid,
                              data_repo_t *successor_repo, parsec_key_t successor_repo_key,
                              void *param);

int
parsec_get_copy_reshape_from_desc(parsec_execution_stream_t *es,
                                  parsec_taskpool_t* tp,
                                  parsec_task_t *task,
                                  uint8_t dep_flow_index,
                                  data_repo_t *reshape_repo,
                                  parsec_key_t reshape_entry_key,
                                  parsec_dep_data_description_t *data,
                                  parsec_data_copy_t**reshape);

int
parsec_get_copy_reshape_from_dep(parsec_execution_stream_t *es,
                                 parsec_taskpool_t* tp,
                                 parsec_task_t *task,
                                 uint8_t dep_flow_index,
                                 data_repo_t *reshape_repo,
                                 parsec_key_t reshape_entry_key,
                                 parsec_dep_data_description_t *data,
                                 parsec_data_copy_t**reshape);


/** deps is an array of size MAX_PARAM_COUNT
 *  Returns the number of output deps on which there is a final output
 */
int parsec_task_deps_with_final_output(const parsec_task_t *task,
                                      const parsec_dep_t **deps);

int32_t parsec_add_fetch_runtime_task( parsec_taskpool_t *tp, int32_t nb_tasks );

void parsec_dependencies_mark_task_as_startup(parsec_task_t* task, parsec_execution_stream_t *es);

int
parsec_release_local_OUT_dependencies(parsec_execution_stream_t* es,
                                      const parsec_task_t* origin,
                                      const parsec_flow_t* origin_flow,
                                      const parsec_task_t* task,
                                      const parsec_flow_t* dest_flow,
                                      parsec_dep_data_description_t* data,
                                      parsec_task_t** pready_ring,
                                      data_repo_t* target_repo,
                                      parsec_data_copy_t* target_dc,
                                      data_repo_entry_t* target_repo_entry);

#define parsec_execution_context_priority_comparator offsetof(parsec_task_t, priority)

#if defined(PARSEC_SIM)
int parsec_getsimulationdate( parsec_context_t *parsec_context );
#endif

/*********************** Global Info Handles *****************************/

/**
 * @brief Device-level info
 *
 * @details infos stored under this handle exist per device:
 *      there is one info_array per device (as obtained by
 *      parsec_mca_device_get), and this array is indexed
 *      by the info IDs stored inside this global.
 *
 *      The info-specific object passed to the constructor
 *      function for info objects is the corresponding
 *      parsec_device_module_t*.
 */
extern parsec_info_t parsec_per_device_infos;

/**
 * @brief Stream-level info
 *
 * @details infos stored under this handle exist per device-stream:
 *      some devices, like the CUDA device, define multiple streams
 *      of execution; they define one info_array per such stream,
 *      and this array is indexed by the info IDs stored inside this
 *      global.
 *
 *      the info-specific object passed to the constructor function
 *      for info objects is the corresponding parsec_gpu_exec_stream_t*
 */
extern parsec_info_t parsec_per_stream_infos;

/** @} */

END_C_DECLS

#endif  /* PARSEC_INTERNAL_H_HAS_BEEN_INCLUDED */
