/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_PARSEC_REMOTE_DEP_H__
#define __USE_PARSEC_REMOTE_DEP_H__

/** @addtogroup parsec_internal_communication
 *  @{
 */


#define PARSEC_REMOTE_DEP_USE_THREADS

#include "parsec/bindthread.h"
#include "parsec/class/dequeue.h"
#include "parsec/class/lifo.h"
#include "parsec/parsec_description_structures.h"
#include "parsec/parsec_internal.h"
#include "parsec/parsec_comm_engine.h"
#include "parsec/scheduling.h"
#include "parsec/parsec_internal.h"

typedef struct dep_cmd_item_s dep_cmd_item_t;
typedef union dep_cmd_u dep_cmd_t;

typedef unsigned long remote_dep_datakey_t;

#define PARSEC_ACTION_DEPS_MASK                  0x00FFFFFF
#define PARSEC_ACTION_RELEASE_LOCAL_DEPS         0x01000000
#define PARSEC_ACTION_RELEASE_LOCAL_REFS         0x02000000
#define PARSEC_ACTION_GET_REPO_ENTRY             0x04000000
#define PARSEC_ACTION_SEND_INIT_REMOTE_DEPS      0x10000000
#define PARSEC_ACTION_SEND_REMOTE_DEPS           0x20000000
#define PARSEC_ACTION_RECV_INIT_REMOTE_DEPS      0x40000000
#define PARSEC_ACTION_RELEASE_REMOTE_DEPS        (PARSEC_ACTION_SEND_INIT_REMOTE_DEPS | PARSEC_ACTION_SEND_REMOTE_DEPS)

enum {
    REMOTE_DEP_ACTIVATE_TAG = 2,
    REMOTE_DEP_GET_DATA_TAG,
    REMOTE_DEP_PUT_END_TAG,
    REMOTE_DEP_MAX_CTRL_TAG
} parsec_remote_dep_tag_t;

typedef struct remote_dep_wire_activate_s {
    remote_dep_datakey_t deps;         /**< a pointer to the dep structure on the source */
    remote_dep_datakey_t output_mask;  /**< the mask of the output dependencies satisfied by this activation message */
    uint32_t             taskpool_id;
    uint32_t             task_class_id;
    uint32_t             length;
    parsec_assignment_t  locals[MAX_LOCAL_COUNT];
} remote_dep_wire_activate_t;

typedef struct remote_dep_wire_get_s {
    remote_dep_datakey_t       source_deps;
    remote_dep_datakey_t       remote_callback_data;
    remote_dep_datakey_t       output_mask;
    uintptr_t                  callback_fn;
    parsec_ce_mem_reg_handle_t remote_memory_handle;
} remote_dep_wire_get_t;

/**
 * This structure holds the key information for any data mouvement. It contains the arena
 * where the data is allocated from, or will be allocated from. It also contains the
 * pointer to the buffer involved in the communication (or NULL if the data will be
 * allocated before the reception). Finally, it contains the triplet allowing a correct send
 * or receive operation: the memory layout, the number fo repetitions and the displacement
 * from the data pointer where the operation will start. If the memory layout is NULL the
 * one attached to the arena must be used instead.
 */
struct parsec_dep_data_description_s {
    struct parsec_arena_s     *arena;
    struct parsec_data_copy_s *data;
    parsec_datatype_t          layout;
    uint64_t                   count;
    int64_t                    displ;
};

struct remote_dep_output_param_s {
    /** Never change this structure without understanding the
     *   "subtle" relation with remote_deps_allocation_init in
     *  remote_dep.c
     */
    parsec_list_item_t                    super;
    parsec_remote_deps_t                 *parent;
    struct parsec_dep_data_description_s  data;        /**< The data propagated by this message. */
    uint32_t                             deps_mask;   /**< A bitmask of all the output dependencies
                                                       propagated by this message. The bitmask uses
                                                       depedencies indexes not flow indexes. */
    int32_t                              priority;    /**< the priority of the message */
    uint32_t                             count_bits;  /**< The number of participants */
    uint32_t*                            rank_bits;   /**< The array of bits representing the propagation path */
};

struct parsec_remote_deps_s {
    parsec_list_item_t               super;
    parsec_lifo_t                   *origin;    /**< The memory arena where the data pointer is comming from */
    struct parsec_taskpool_s        *taskpool;  /**< parsec taskpool generating this data transfer */
    int32_t                          pending_ack;  /**< Number of releases before completion */
    int32_t                          from;    /**< From whom we received the control */
    int32_t                          root;    /**< The root of the control message */
    uint32_t                         incoming_mask;  /**< track all incoming actions (receives) */
    uint32_t                         outgoing_mask;  /**< track all outgoing actions (send) */
    remote_dep_wire_activate_t       msg;     /**< A copy of the message control */
    void                            *eager_msg; /**< A pointer to the eager buffer if this is an eager msg, otherwise NULL */
    int32_t                          max_priority;
    int32_t                          priority;
    uint32_t                        *remote_dep_fw_mask;  /**< list of peers already notified about
                                                            * the control sequence (only used for control messages) */
    struct data_repo_entry_s        *repo_entry;
    struct remote_dep_output_param_s output[1];
};
/* { item .. remote_dep_fw_mask (points to fw_mask_bitfield),
 *   output[0] .. output[max_deps < MAX_PARAM_COUNT],
 *   (max_dep_count x (np+31)/32 uint32_t) rank_bits
 *   ((np+31)/32 x uint32_t) fw_mask_bitfield } */

/* This int can take the following values:
 * - negative: no communication engine has been enabled
 * - 0: the communication engine is not turned on
 * - positive: the meaning is defined by the communication engine.
 */
extern int parsec_communication_engine_up;
extern int parsec_comm_output_stream;
extern int parsec_comm_verbose;
extern parsec_execution_stream_t parsec_comm_es;
extern int parsec_param_comm_thread_multiple;

#ifdef DISTRIBUTED

typedef struct {
    parsec_lifo_t freelist;
    uint32_t     max_dep_count;
    uint32_t     max_nodes_number;
    uint32_t     elem_size;
} parsec_remote_dep_context_t;

extern parsec_remote_dep_context_t parsec_remote_dep_context;

void remote_deps_allocation_init(int np, int max_deps);
void remote_deps_allocation_fini(void);

parsec_remote_deps_t* remote_deps_allocate( parsec_lifo_t* lifo );

#define PARSEC_ALLOCATE_REMOTE_DEPS_IF_NULL(REMOTE_DEPS, TASK, COUNT) \
    if( NULL == (REMOTE_DEPS) ) { /* only once per function */                 \
        (REMOTE_DEPS) = (parsec_remote_deps_t*)remote_deps_allocate(&parsec_remote_dep_context.freelist); \
    }

/* This returns the deps to the freelist, no use counter */
void remote_deps_free(parsec_remote_deps_t* deps);

int parsec_remote_dep_init(parsec_context_t* context);
int parsec_remote_dep_fini(parsec_context_t* context);
int parsec_remote_dep_on(parsec_context_t* context);
int parsec_remote_dep_off(parsec_context_t* context);

/* Poll for remote completion of tasks that would enable some work locally */
int parsec_remote_dep_progress(parsec_execution_stream_t* es);

/* Inform the communication engine from the creation of new taskpools */
int parsec_remote_dep_new_taskpool(parsec_taskpool_t* tp);

/* Send remote dependencies to target processes */
int parsec_remote_dep_activate(parsec_execution_stream_t* es,
                               const parsec_task_t* origin,
                               parsec_remote_deps_t* remote_deps,
                               uint32_t propagation_mask);

/* Memcopy a particular data using datatype specification */
void parsec_remote_dep_memcpy(parsec_execution_stream_t* es,
                             parsec_taskpool_t* tp,
                             parsec_data_copy_t *dst,
                             parsec_data_copy_t *src,
                             parsec_dep_data_description_t* data);

/* Perform a memcpy with datatypes by doing a local sendrecv */
int remote_dep_nothread_memcpy(parsec_execution_stream_t* es,
                                      dep_cmd_item_t *item);
/* This function adds a command in the command queue to activate
 * release_deps of dep we had to delay in DTD runs.
 */
int
remote_dep_dequeue_delayed_dep_release(parsec_remote_deps_t *deps);


#if defined(PARSEC_DIST_COLLECTIVES)
/* Propagate an activation order from the current node down the original tree */
int parsec_remote_dep_propagate(parsec_execution_stream_t* es,
                               const parsec_task_t* task,
                               parsec_remote_deps_t* deps);
#endif

#else
#define parsec_remote_dep_init(ctx)            1
#define parsec_remote_dep_fini(ctx)            0
#define parsec_remote_dep_on(ctx)              0
#define parsec_remote_dep_off(ctx)             0
#define parsec_remote_dep_progress(ctx)        0
#define parsec_remote_dep_activate(ctx, o, r) -1
#define parsec_remote_dep_new_taskpool(ctx)    0
#endif /* DISTRIBUTED */

/** @} */

#define DEP_NB_CONCURENT 3

extern int parsec_comm_gets_max;
extern int parsec_comm_gets;
extern int parsec_comm_puts_max;
extern int parsec_comm_puts;

/**
 * The order is important as it will be used to compute the index in the
 * pending array of messages.
 */
typedef enum dep_cmd_action_t {
    DEP_ACTIVATE      = -1,
    DEP_NEW_TASKPOOL  =  0,
    DEP_MEMCPY,
    DEP_RELEASE,
    DEP_DTD_DELAYED_RELEASE,
    DEP_PUT_DATA,
    DEP_GET_DATA,
    DEP_CTL,
    DEP_LAST  /* always the last element. it shoud not be used */
} dep_cmd_action_t;

union dep_cmd_u {
    struct {
        remote_dep_wire_get_t      task;
        int                        peer;
        parsec_ce_mem_reg_handle_t remote_memory_handle;
    } activate;
    struct {
        parsec_remote_deps_t  *deps;
    } release;
    struct {
        int enable;
    } ctl;
    struct {
        parsec_taskpool_t    *tp;
    } new_taskpool;
    struct {
        parsec_taskpool_t             *taskpool;
        parsec_data_copy_t            *source;
        parsec_data_copy_t            *destination;
        parsec_datatype_t              layout;
        int64_t                        displ_src;
        int64_t                        displ_dst;
        uint64_t                       count;
    } memcpy;
};

struct dep_cmd_item_s {
    parsec_list_item_t super;
    parsec_list_item_t pos_list;
    dep_cmd_action_t  action;
    int               priority;
    dep_cmd_t         cmd;
};

#define dep_cmd_prio (offsetof(dep_cmd_item_t, priority))
#define dep_mpi_pos_list (offsetof(dep_cmd_item_t, priority) - offsetof(dep_cmd_item_t, pos_list))
#define rdep_prio (offsetof(parsec_remote_deps_t, max_priority))

/**
 * These functions will be inherited from the current remote_dep_mpi.c
 * and for the time being will remain in there.
 */
void* remote_dep_dequeue_main(parsec_context_t* context);
int remote_dep_dequeue_new_taskpool(parsec_taskpool_t* tp);

int remote_dep_dequeue_on(parsec_context_t* context);
int remote_dep_dequeue_off(parsec_context_t* context);
#   define remote_dep_init(ctx) remote_dep_dequeue_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_dequeue_on(ctx)
#   define remote_dep_off(ctx)  remote_dep_dequeue_off(ctx)
#   define remote_dep_new_taskpool(tp) remote_dep_dequeue_new_taskpool(tp)
#   define remote_dep_send(es, rank, deps) remote_dep_dequeue_send(es, rank, deps)
#   define remote_dep_progress(es, cycles) remote_dep_dequeue_nothread_progress(es, cycles)

int remote_dep_dequeue_nothread_progress(parsec_execution_stream_t* es, int cycles);

int remote_dep_bind_thread(parsec_context_t* context);
int remote_dep_complete_and_cleanup(parsec_remote_deps_t** deps,
                                int ncompleted);

/* comm_yield mode: see valid values in the corresponding mca_register */
extern int comm_yield;
/* comm_yield_duration (ns) */
extern int comm_yield_ns;

/* make sure we don't leave before serving all data deps */
static inline void
remote_dep_inc_flying_messages(parsec_taskpool_t* handle)
{
    assert( handle->nb_pending_actions > 0 );
    (void)parsec_atomic_fetch_inc_int32( &(handle->nb_pending_actions) );
}

/* allow for termination when all deps have been served */
static inline void
remote_dep_dec_flying_messages(parsec_taskpool_t *handle)
{
    (void)parsec_taskpool_update_runtime_nbtask(handle, -1);
}

int remote_dep_set_ctx(parsec_context_t* context, intptr_t opaque_comm_ctx );

parsec_remote_deps_t* remote_deps_allocate( parsec_lifo_t* lifo );

void remote_deps_allocation_init(int np, int max_output_deps);

void remote_dep_mpi_get_end(parsec_execution_stream_t* es,
                            int idx,
                            parsec_remote_deps_t* deps);

typedef struct {
    int rank_src;  // 0
    int rank_dst;  // 4
    uint64_t tid;  // 8
    uint32_t tpid;  // 16
    uint32_t tcid;  // 20
} parsec_profile_remote_dep_mpi_info_t; // 24 bytes

#ifdef PARSEC_PROF_TRACE
extern parsec_thread_profiling_t* MPIctl_prof;
extern parsec_thread_profiling_t* MPIsnd_prof;
extern parsec_thread_profiling_t* MPIrcv_prof;
extern int MPI_Activate_sk, MPI_Activate_ek;
extern int MPI_Data_ctl_sk, MPI_Data_ctl_ek;
extern int MPI_Data_plds_sk, MPI_Data_plds_ek;
extern int MPI_Data_pldr_sk, MPI_Data_pldr_ek;
extern int activate_cb_trace_sk, activate_cb_trace_ek;
extern int put_cb_trace_sk, put_cb_trace_ek;

#define TAKE_TIME_WITH_INFO(PROF, KEY, I, src, dst, rdw)                \
    if( parsec_profile_enabled ) {                                      \
        parsec_profile_remote_dep_mpi_info_t __info;                    \
        parsec_taskpool_t *__tp = parsec_taskpool_lookup( (rdw).taskpool_id ); \
        const parsec_task_class_t *__tc = __tp->task_classes_array[(rdw).task_class_id ]; \
        __info.rank_src = (src);                                        \
        __info.rank_dst = (dst);                                        \
        __info.tpid = __tp->taskpool_id;                                \
        __info.tcid = (rdw).task_class_id;                              \
        __info.tid  = __tc->key_functions->key_hash(                    \
                             __tc->make_key(__tp, (rdw).locals), NULL); \
        PARSEC_PROFILING_TRACE((PROF), (KEY), (I),                      \
                               PROFILE_OBJECT_ID_NULL, &__info);        \
    }

#define TAKE_TIME(PROF, KEY, I) PARSEC_PROFILING_TRACE((PROF), (KEY), (I), PROFILE_OBJECT_ID_NULL, NULL)

#else
#define TAKE_TIME_WITH_INFO(PROF, KEY, I, src, dst, rdw) do {} while(0)
#define TAKE_TIME(PROF, KEY, I) do {} while(0)
#endif  /* PARSEC_PROF_TRACE */

char*
remote_dep_cmd_to_string(remote_dep_wire_activate_t* origin,
                         char* str,
                         size_t len);

extern int parsec_comm_gets_max;
extern int parsec_comm_gets;
extern int parsec_comm_puts_max;
extern int parsec_comm_puts;

#define MIN_MPI_TAG (REMOTE_DEP_MAX_CTRL_TAG+1)
#if defined(PARSEC_HAVE_MPI_OVERTAKE)
extern int parsec_param_enable_mpi_overtake;
#endif

#endif /* __USE_PARSEC_REMOTE_DEP_H__ */
