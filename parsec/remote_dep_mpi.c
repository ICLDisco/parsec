/*
 * Copyright (c) 2009-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* /!\  THIS FILE IS NOT INTENDED TO BE COMPILED ON ITS OWN
 *      It should be included from remote_dep.c if PARSEC_HAVE_MPI is defined
 */
#include "parsec/parsec_config.h"

#include <mpi.h>
#include "profiling.h"
#include "parsec/class/list.h"
#include "parsec/utils/output.h"
#include "parsec/utils/debug.h"
#include "parsec/debug_marks.h"
#include "parsec/data.h"
#include "parsec/papi_sde.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"
#include "parsec/parsec_binary_profile.h"

#define PARSEC_REMOTE_DEP_USE_THREADS

#define PARSEC_DTD_SKIP_SAVING -1

static char **dep_activate_buff;

typedef struct dep_cmd_item_s dep_cmd_item_t;
typedef union dep_cmd_u dep_cmd_t;

static int remote_dep_mpi_setup(parsec_context_t* context);
static int remote_dep_mpi_cleanup(parsec_context_t* context);
static int remote_dep_mpi_fini(parsec_context_t* context);
static int remote_dep_mpi_on(parsec_context_t* context);
static int remote_dep_mpi_progress(parsec_execution_stream_t* es);
static int remote_dep_get_datatypes(parsec_execution_stream_t* es,
                                    parsec_remote_deps_t* origin,
                                    int storage_id, int *position);
static parsec_remote_deps_t*
remote_dep_release_incoming(parsec_execution_stream_t* es,
                            parsec_remote_deps_t* origin,
                            remote_dep_datakey_t complete_mask);

static int remote_dep_nothread_send(parsec_execution_stream_t* es,
                                    dep_cmd_item_t **head_item);
static int remote_dep_nothread_memcpy(parsec_execution_stream_t* es,
                                      dep_cmd_item_t *item);

static int remote_dep_dequeue_send(parsec_execution_stream_t* es, int rank, parsec_remote_deps_t* deps);
static int remote_dep_dequeue_new_taskpool(parsec_taskpool_t* tp);
#ifdef PARSEC_REMOTE_DEP_USE_THREADS
static int remote_dep_dequeue_init(parsec_context_t* context);
static int remote_dep_dequeue_fini(parsec_context_t* context);
static int remote_dep_dequeue_on(parsec_context_t* context);
static int remote_dep_dequeue_off(parsec_context_t* context);
/*static int remote_dep_dequeue_progress(parsec_execution_stream_t* es);*/
#   define remote_dep_init(ctx) remote_dep_dequeue_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_dequeue_on(ctx)
#   define remote_dep_off(ctx)  remote_dep_dequeue_off(ctx)
#   define remote_dep_new_taskpool(tp) remote_dep_dequeue_new_taskpool(tp)
#   define remote_dep_send(es, rank, deps) remote_dep_dequeue_send(es, rank, deps)
#   define remote_dep_progress(es, cycles) remote_dep_dequeue_nothread_progress(es, cycles)

#else
static int remote_dep_dequeue_nothread_init(parsec_context_t* context);
static int remote_dep_dequeue_nothread_fini(parsec_context_t* context);
#   define remote_dep_init(ctx) remote_dep_dequeue_nothread_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_nothread_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_mpi_on(ctx)
#   define remote_dep_off(ctx)  0
#   define remote_dep_new_taskpool(tp) remote_dep_dequeue_new_taskpool(tp)
#   define remote_dep_send(es, rank, deps) remote_dep_dequeue_send(es, rank, deps)
#   define remote_dep_progress(es, cycles) remote_dep_dequeue_nothread_progress(es, cycles)
#endif
static int remote_dep_dequeue_nothread_progress(parsec_execution_stream_t* es, int cycles);

#include "parsec/class/dequeue.h"

#include "parsec/utils/mca_param.h"
/**
 * Number of data movements to be extracted at each step. Bigger the number
 * larger the amount spent in ordering the tasks, but greater the potential
 * benefits of doing things in the right order.
 */
static void remote_dep_mpi_params(parsec_context_t* context);
static int parsec_param_nb_tasks_extracted = 20;
/* For the meaning of aggregate, short and eager, refer to the
 * param register help text for comm_aggregate,
 * comm_short_limit and comm_eager_limit respectively.
 */
static size_t parsec_param_short_limit = RDEP_MSG_SHORT_LIMIT;
#if RDEP_MSG_EAGER_LIMIT != 0
/* Disable this by default as it is currently broken
static size_t parsec_param_eager_limit = RDEP_MSG_EAGER_LIMIT;
*/
static size_t parsec_param_eager_limit = 0;
#endif  /* RDEP_MSG_EAGER_LIMIT != 0 */
static int parsec_param_enable_aggregate = 1;
#if defined(PARSEC_HAVE_MPI_OVERTAKE)
static int parsec_param_enable_mpi_overtake = 1;
#endif

#define DEP_NB_CONCURENT 3
static int DEP_NB_REQ;

static int parsec_comm_activations_max = 2*DEP_NB_CONCURENT;
static int parsec_comm_data_get_max    = 2*DEP_NB_CONCURENT;
static int parsec_comm_gets_max        = DEP_NB_CONCURENT * MAX_PARAM_COUNT;
static int parsec_comm_gets            = 0;
static int parsec_comm_puts_max        = DEP_NB_CONCURENT * MAX_PARAM_COUNT;
static int parsec_comm_puts            = 0;
static int parsec_comm_last_active_req = 0;

/* The internal communicator used by the communication engine to host its requests and
 * other operations. It is a copy of the context->comm_ctx (which is a duplicate of
 * whatever the user provides).
 */
static MPI_Comm dep_comm = MPI_COMM_NULL;
/* The internal communicator for all intra-node communications */
static MPI_Comm dep_self = MPI_COMM_NULL;

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
/*    DEP_PROGRESS, */
    DEP_PUT_DATA,
    DEP_GET_DATA,
    DEP_CTL,
    DEP_LAST  /* always the last element. it shoud not be used */
} dep_cmd_action_t;

union dep_cmd_u {
    struct {
        remote_dep_wire_get_t task;
        int                   peer;
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
        parsec_taskpool_t    *taskpool;
        parsec_data_copy_t   *source;
        parsec_data_copy_t   *destination;
        parsec_datatype_t     datatype;
        int64_t               displ_s;
        int64_t               displ_r;
        int                   count;
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

typedef struct parsec_comm_callback_s parsec_comm_callback_t;

static int
remote_dep_mpi_save_put_cb(parsec_execution_stream_t* es,
                           parsec_comm_callback_t* cb, MPI_Status* status);
static void remote_dep_mpi_put_start(parsec_execution_stream_t* es, dep_cmd_item_t* item);
static int remote_dep_mpi_put_end_cb(parsec_execution_stream_t* es,
                                     parsec_comm_callback_t* cb, MPI_Status* status);
#if 0 != RDEP_MSG_EAGER_LIMIT
static void remote_dep_mpi_put_eager( parsec_execution_stream_t* es,
                                      dep_cmd_item_t* item);
#endif  /* 0 != RDEP_MSG_EAGER_LIMIT */
static int remote_dep_mpi_save_activate_cb(parsec_execution_stream_t* es,
                                           parsec_comm_callback_t* cb, MPI_Status* status);
static void remote_dep_mpi_get_start(parsec_execution_stream_t* es, parsec_remote_deps_t* deps);
static void remote_dep_mpi_get_end( parsec_execution_stream_t* es, int idx, parsec_remote_deps_t* deps );
static int
remote_dep_mpi_get_end_cb(parsec_execution_stream_t* es,
                          parsec_comm_callback_t* cb, MPI_Status* status);
static void remote_dep_mpi_new_taskpool( parsec_execution_stream_t* es, dep_cmd_item_t *item );
static void remote_dep_mpi_release_delayed_deps( parsec_execution_stream_t* es,
                                                 dep_cmd_item_t *item );

extern char*
remote_dep_cmd_to_string(remote_dep_wire_activate_t* origin,
                         char* str,
                         size_t len)
{
    parsec_task_t task;

    task.taskpool = parsec_taskpool_lookup( origin->taskpool_id );
    if( NULL == task.taskpool ) return snprintf(str, len, "UNKNOWN_of_TASKPOOL_%d", origin->taskpool_id), str;
    task.task_class   = task.taskpool->task_classes_array[origin->task_class_id];
    if( NULL == task.task_class ) return snprintf(str, len, "UNKNOWN_of_TASKCLASS_%d", origin->task_class_id), str;
    memcpy(&task.locals, origin->locals, sizeof(parsec_assignment_t) * task.task_class->nb_locals);
    task.priority     = 0xFFFFFFFF;
    return parsec_task_snprintf(str, len, &task);
}

static pthread_t dep_thread_id;
parsec_dequeue_t dep_cmd_queue;
parsec_list_t    dep_cmd_fifo;             /* ordered non threaded fifo */
parsec_list_t    dep_activates_fifo;       /* ordered non threaded fifo */
parsec_list_t    dep_activates_noobj_fifo; /* non threaded fifo */
parsec_list_t    dep_put_fifo;             /* ordered non threaded fifo */

/* help manage the messages in the same category, where a category is either messages
 * to the same destination, or with the same action key.
 */
static dep_cmd_item_t** parsec_mpi_same_pos_items;
static int parsec_mpi_same_pos_items_size = 0;

static void *remote_dep_dequeue_main(parsec_context_t* context);
static int mpi_initialized = 0;
#if defined(PARSEC_REMOTE_DEP_USE_THREADS)
static pthread_mutex_t mpi_thread_mutex;
static pthread_cond_t mpi_thread_condition;
#endif

static parsec_execution_stream_t parsec_comm_es = {
    .th_id = 0,
    .core_id = -1,
    .socket_id = -1,
#if defined(PARSEC_PROF_TRACE)
    .es_profile = NULL,
#endif /* PARSEC_PROF_TRACE */
    .scheduler_object = NULL,
#if defined(PARSEC_SIM)
    .largest_simulation_date = 0,
#endif
#if defined(PARSEC_PROF_PINS)
    .pins_events_cb = {{0}},
#endif  /* defined(PARSEC_PROF_PINS) */
#if defined(PARSEC_PROF_RUSAGE_EU)
#if defined(PARSEC_HAVE_GETRUSAGE) || !defined(__bgp__)
    ._es_rusage = {{0}},
#endif /* PARSEC_HAVE_GETRUSAGE */
#endif
    .virtual_process = NULL,
    .context_mempool = NULL,
    .datarepo_mempools = {0}
};

/**
 * Store the user provided communicator in the PaRSEC context. We need to make a
 * copy to make sure the communicator does not dissapear before the communication
 * engine starts up.
 */
static int remote_dep_set_ctx(parsec_context_t* context, intptr_t opaque_comm_ctx )
{
    MPI_Comm comm;
    int rc;

    /* We can only change the communicator if the communication engine is not active */
    if( 1 < parsec_communication_engine_up ) {
        parsec_warning("Cannot change PaRSEC's MPI communicator while the engine is running [ignored]");
        return PARSEC_ERROR;
    }
    /* Are we trying to set a congruent communicator a second time? */
    assert(-1 != opaque_comm_ctx /* -1 reserved for non-initialized */);
    if( -1 != context->comm_ctx ) {
        MPI_Comm_compare((MPI_Comm)context->comm_ctx, (MPI_Comm)opaque_comm_ctx, &rc);
        if( (MPI_IDENT == rc) || (MPI_CONGRUENT == rc) ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "Set the same or a congruent communicator. Nothing to do");
            return PARSEC_SUCCESS;
        }
        MPI_Comm_free((MPI_Comm*)&context->comm_ctx);
    }
    rc = MPI_Comm_dup((MPI_Comm)opaque_comm_ctx, &comm);
    context->comm_ctx = (intptr_t)comm;
    return (MPI_SUCCESS == rc) ? PARSEC_SUCCESS : PARSEC_ERROR;
}

static int remote_dep_dequeue_init(parsec_context_t* context)
{
    pthread_attr_t thread_attr;
    int is_mpi_up = 0;
    int thread_level_support;

    assert(mpi_initialized == 0);

    remote_dep_mpi_params(context);

    MPI_Initialized(&is_mpi_up);
    if( 0 == is_mpi_up ) {
        /**
         * MPI is not up, so we will consider this as a single node run. Fall
         * back to the no-MPI case.
         */
        context->nb_nodes = 1;
        parsec_communication_engine_up = -1;  /* No communications supported */
        /*TODO: restore the original behavior when modular datatype engine is
         * available */
        parsec_fatal("MPI was not initialized. This version of PaRSEC was compiled with MPI datatype supports and *needs* MPI to execute.\n"
                     "\t* Please initialized MPI in the application (MPI_Init/MPI_Init_thread) prior to initializing PaRSEC.\n"
                     "\t* Alternatively, compile a version of PaRSEC without MPI (-DPARSEC_DIST_WITH_MPI=OFF in ccmake)\n");
        return 1;
    }
    parsec_communication_engine_up = 0;  /* we have communication capabilities */

    MPI_Query_thread( &thread_level_support );
    if( thread_level_support == MPI_THREAD_SINGLE ||
        thread_level_support == MPI_THREAD_FUNNELED ) {
        parsec_warning("MPI was not initialized with the appropriate level of thread support.\n"
                      "\t* Current level is %s, while MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE is needed\n"
                      "\t* to guarantee correctness of the PaRSEC runtime.\n",
                thread_level_support == MPI_THREAD_SINGLE ? "MPI_THREAD_SINGLE" : "MPI_THREAD_FUNNELED" );
    }
    if( -1 == context->comm_ctx ) {
        MPI_Comm comm;
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        context->comm_ctx = (intptr_t)comm;
        assert(-1 != context->comm_ctx /* -1 reserved for non-initialized */);
    }
    MPI_Comm_size( (MPI_Comm)context->comm_ctx, (int*)&(context->nb_nodes));

    if(parsec_param_comm_thread_multiple) {
        if( thread_level_support >= MPI_THREAD_MULTIPLE ) {
            context->flags |= PARSEC_CONTEXT_FLAG_COMM_MT;
        }
        else if(parsec_param_comm_thread_multiple != -1) {
            parsec_warning("Requested multithreaded access to the communication engine, but MPI is not initialized with MPI_THREAD_MULTIPLE.\n"
                        "\t* PaRSEC will continue with the funneled thread communication engine model.\n");
        }
#if RDEP_MSG_EAGER_LIMIT != 0
        if( (context->flags & PARSEC_CONTEXT_FLAG_COMM_MT) && parsec_param_eager_limit ) {
            parsec_warning("Using eager and thread multiple MPI messaging is not implemented yet. Disabling Eager.");
            parsec_param_eager_limit = 0;
        }
#endif
    }
#if defined(PARSEC_HAVE_MPI_OVERTAKE)
    parsec_mca_param_reg_int_name("runtime", "comm_mpi_overtake", "Lets MPI allow overtaking of messages (if applicable). (0: no, 1: yes)",
                                  false, false, parsec_param_enable_mpi_overtake, &parsec_param_enable_mpi_overtake);
#endif

    /**
     * Finalize the initialization of the upper level structures
     * Worst case: one of the DAGs is going to use up to
     * MAX_PARAM_COUNT times nb_nodes dependencies.
     */
    remote_deps_allocation_init(context->nb_nodes, MAX_PARAM_COUNT);

    PARSEC_OBJ_CONSTRUCT(&dep_cmd_queue, parsec_dequeue_t);
    PARSEC_OBJ_CONSTRUCT(&dep_cmd_fifo, parsec_list_t);

    /* From now on the communication capabilities are enabled */
    parsec_communication_engine_up = 1;
    if(context->nb_nodes == 1) {
        /* We're all by ourselves. In case we need to use MPI to handle data copies
         * between different formats let's setup local MPI support.
         */
        remote_dep_mpi_on(context);

        goto up_and_running;
    }

    /* Build the condition used to drive the MPI thread */
    pthread_mutex_init( &mpi_thread_mutex, NULL );
    pthread_cond_init( &mpi_thread_condition, NULL );

    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);

   /**
    * We need to synchronize with the newly spawned thread. We will use the
    * condition for this. If we lock the mutex prior to spawning the MPI thread,
    * and then go in a condition wait, the MPI thread can lock the mutex, and
    * then call condition signal. This insure proper synchronization. Similar
    * mechanism will be used to turn on and off the MPI thread.
    */
    pthread_mutex_lock(&mpi_thread_mutex);

    pthread_create(&dep_thread_id,
                   &thread_attr,
                   (void* (*)(void*))remote_dep_dequeue_main,
                   (void*)context);

    /* Wait until the MPI thread signals it's awakening */
    pthread_cond_wait( &mpi_thread_condition, &mpi_thread_mutex );
  up_and_running:
    mpi_initialized = 1;  /* up and running */

    return context->nb_nodes;
}

static int remote_dep_dequeue_fini(parsec_context_t* context)
{
    if( 0 == mpi_initialized ) return 0;
    (void)context;

    /**
     * We suppose the off function was called before. Then we will append a
     * shutdown command in the MPI thread queue, and wake the MPI thread. Upon
     * processing of the pending command the MPI thread will exit, we will be
     * able to catch this by locking the mutex.  Once we know the MPI thread is
     * gone, cleaning up will be straighforward.
     */
    if( 1 < parsec_communication_engine_up ) {
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
        void *ret;

        item->action = DEP_CTL;
        item->cmd.ctl.enable = -1;  /* turn off and return from the MPI thread */
        item->priority = 0;
        parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*) item);

        /* I am supposed to own the lock. Wake the MPI thread */
        pthread_cond_signal(&mpi_thread_condition);
        pthread_mutex_unlock(&mpi_thread_mutex);
        pthread_join(dep_thread_id, &ret);
        assert((parsec_context_t*)ret == context);
    }
    else if ( parsec_communication_engine_up == 1 ) {
        remote_dep_mpi_fini(context);
    }

    assert(NULL == parsec_dequeue_pop_front(&dep_cmd_queue));
    PARSEC_OBJ_DESTRUCT(&dep_cmd_queue);
    assert(NULL == parsec_dequeue_pop_front(&dep_cmd_fifo));
    PARSEC_OBJ_DESTRUCT(&dep_cmd_fifo);
    mpi_initialized = 0;

    return 0;
}

/* The possible values for parsec_communication_engine_up are: 0 if no
 * communication capabilities are enabled, 1 if we are in a single node scenario
 * and the main thread will check the communications on a regular basis, 2 if
 * the order is enqueued but the thread is not yet on, and 3 if the thread is
 * running.
 */
static int remote_dep_dequeue_on(parsec_context_t* context)
{
    /* If we are the only participant in this execution, we should not have to
     * communicate with any other process. However, we might have to execute all
     * local data copies, which requires MPI.
     */
    if( 0 >= parsec_communication_engine_up ) return -1;
    if( context->nb_nodes == 1 ) return 1;

    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: comm engine signalled ON on process %d/%d",
                         context->my_rank, context->nb_nodes);

    /* At this point I am supposed to own the mutex */
    parsec_communication_engine_up = 2;
    pthread_cond_signal(&mpi_thread_condition);
    pthread_mutex_unlock(&mpi_thread_mutex);
    /* The waking up of the communication thread happen asynchronously, once the thread
     * receives the signal. At that point it acquires the mpi_thread_mutex and set the
     * global variable parsec_communication_engine_up to 3.
     */

    /**
     * We need to wait for the communication thread to perform the mpi_setup
     * as it will fill-up my_rank on the context.
     */
    while( 3 != parsec_communication_engine_up ) sched_yield();

    (void)context;
    return 1;
}

static int remote_dep_dequeue_off(parsec_context_t* context)
{
    if(parsec_communication_engine_up < 2) return -1;  /* The start order has not been issued */

    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    item->action = DEP_CTL;
    item->cmd.ctl.enable = 0;  /* turn OFF the MPI thread */
    item->priority = 0;
    /* wait until the communication thread is up and running */
    while( 3 != parsec_communication_engine_up ) sched_yield();
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: comm engine signalled OFF on process %d/%d",
                         context->my_rank, context->nb_nodes);
    parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*) item);

    /* wait until we own the PaRSEC MPI synchronization mutex */
    pthread_mutex_lock(&mpi_thread_mutex);
    assert( 1 == parsec_communication_engine_up );
    (void)context;  /* silence warning */
    return 0;
}

#include "parsec/bindthread.h"

#if defined PARSEC_PROF_TRACE
static void remote_dep_mpi_profiling_init(void);
#else
#define remote_dep_mpi_profiling_init() do {} while(0)
#endif

static void* remote_dep_dequeue_main(parsec_context_t* context)
{
    int whatsup;

    remote_dep_bind_thread(context);
    PARSEC_PAPI_SDE_THREAD_INIT();

    /* Now synchronize with the main thread */
    pthread_mutex_lock(&mpi_thread_mutex);
    pthread_cond_signal(&mpi_thread_condition);

    /* This is the main loop. Wait until being woken up by the main thread, do
     * the MPI stuff until we get the OFF or FINI commands. Then react the them.
     */
    do {
        /* Let's wait until we are awaken */
        pthread_cond_wait(&mpi_thread_condition, &mpi_thread_mutex);
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: comm engine ON on process %d/%d",
                             context->my_rank, context->nb_nodes);
        /* The MPI thread is owning the lock */
        assert( parsec_communication_engine_up == 2 );
        remote_dep_mpi_on(context);
        /* acknoledge the activation */
        parsec_communication_engine_up = 3;
        whatsup = remote_dep_dequeue_nothread_progress(&parsec_comm_es, -1 /* loop till explicitly asked to return */);
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: comm engine OFF on process %d/%d",
                             context->my_rank, context->nb_nodes);
        parsec_communication_engine_up = 1;  /* went to sleep */
    } while(-1 != whatsup);

    /* Release all resources */
    remote_dep_mpi_fini(context);
    PARSEC_PAPI_SDE_THREAD_FINI();

    return (void*)context;
}

static int remote_dep_dequeue_new_taskpool(parsec_taskpool_t* tp)
{
    if(!mpi_initialized) return 0;
    remote_dep_inc_flying_messages(tp);
    dep_cmd_item_t* item = (dep_cmd_item_t*)calloc(1, sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    item->action = DEP_NEW_TASKPOOL;
    item->priority = 0;
    item->cmd.new_taskpool.tp = tp;
    parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*)item);
    return 1;
}

int
remote_dep_dequeue_delayed_dep_release(parsec_remote_deps_t *deps)
{
    if(!mpi_initialized) return 0;
    dep_cmd_item_t* item = (dep_cmd_item_t*)calloc(1, sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    item->action = DEP_DTD_DELAYED_RELEASE;
    item->priority = 0;
    item->cmd.release.deps = deps;
    parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*)item);
    return 1;
}

static int remote_dep_dequeue_send(parsec_execution_stream_t* es, int rank,
                                   parsec_remote_deps_t* deps)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    item->action   = DEP_ACTIVATE;
    item->priority = deps->max_priority;
    item->cmd.activate.peer             = rank;
    item->cmd.activate.task.deps        = (remote_dep_datakey_t)deps;
    item->cmd.activate.task.output_mask = 0;
    item->cmd.activate.task.tag         = 0;
    /* if MPI is multithreaded do not thread-shift the send activate */
    if( parsec_comm_es.virtual_process->parsec_context->flags & PARSEC_CONTEXT_FLAG_COMM_MT ) {
        parsec_list_item_singleton(&item->pos_list); /* NOTE: this disables aggregation in MT cases. */
        remote_dep_nothread_send(es, &item);
    }
    else {
        parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*)item);
    }
    return 1;
}

void parsec_remote_dep_memcpy(parsec_execution_stream_t* es,
                             parsec_taskpool_t* tp,
                             parsec_data_copy_t *dst,
                             parsec_data_copy_t *src,
                             parsec_dep_data_description_t* data)
{
    assert( dst );

    /* if MPI is multithreaded do not thread-shift the sendrecv */
    if( es->virtual_process->parsec_context->flags & PARSEC_CONTEXT_FLAG_COMM_MT ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream,
                             "COPY [in content] LOCAL DATA from %p to %p count %d",
                             (char*)PARSEC_DATA_COPY_GET_PTR(src) + data->displ,
                             (char*)PARSEC_DATA_COPY_GET_PTR(dst) + 0,
                             data->count);
        MPI_Sendrecv((char*)PARSEC_DATA_COPY_GET_PTR(src) + data->displ,
                     data->count, data->layout, 0, es->th_id,
                     (char*)PARSEC_DATA_COPY_GET_PTR(dst) + 0,
                     data->count, data->layout, 0, es->th_id,
                     dep_self, MPI_STATUS_IGNORE);
        return;
    }

    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream,
                         "create MEMCPY request from %p to %p count %d",
                         (char*)PARSEC_DATA_COPY_GET_PTR(src) + data->displ,
                         (char*)PARSEC_DATA_COPY_GET_PTR(dst) + 0,
                         data->count);
    dep_cmd_item_t* item = (dep_cmd_item_t*)calloc(1, sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    item->action = DEP_MEMCPY;
    item->priority = 0;
    item->cmd.memcpy.taskpool = tp;
    item->cmd.memcpy.source       = src;
    item->cmd.memcpy.destination  = dst;
    item->cmd.memcpy.datatype     = data->layout;
    item->cmd.memcpy.displ_s      = data->displ;
    item->cmd.memcpy.displ_r      = 0;
    item->cmd.memcpy.count        = data->count;

    PARSEC_OBJ_RETAIN(src);
    remote_dep_inc_flying_messages(tp);

    parsec_dequeue_push_back(&dep_cmd_queue, (parsec_list_item_t*) item);
}

static inline parsec_data_copy_t*
remote_dep_copy_allocate(parsec_dep_data_description_t* data)
{
    parsec_data_copy_t* dc;
    if( NULL == data->arena ) {
        assert(0 == data->count);
        return NULL;
    }
    dc = parsec_arena_get_copy(data->arena, data->count, 0);
    dc->coherency_state = PARSEC_DATA_COHERENCY_EXCLUSIVE;
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tMalloc new remote tile %p size %" PRIu64 " count = %" PRIu64 " displ = %" PRIi64 "",
            dc, data->arena->elem_size, data->count, data->displ);
    return dc;
}
#define is_inplace(ctx,dep) NULL
#define is_read_only(ctx,dep) NULL

/**
 * This function is called from the task successors iterator. It exists for a
 * single purpose: to retrieve the datatype involved with the operation. Thus,
 * once a datatype has been succesfully retrieved it must cancel the iterator
 * progress in order to return ASAP the datatype to the communication engine.
 */
parsec_ontask_iterate_t
remote_dep_mpi_retrieve_datatype(parsec_execution_stream_t *eu,
                                 const parsec_task_t *newcontext,
                                 const parsec_task_t *oldcontext,
                                 const parsec_dep_t* dep,
                                 parsec_dep_data_description_t* out_data,
                                 int src_rank, int dst_rank, int dst_vpid,
                                 void *param)
{
    (void)eu; (void)oldcontext; (void)dst_vpid; (void)newcontext; (void)out_data;
    if( dst_rank != eu->virtual_process->parsec_context->my_rank )
        return PARSEC_ITERATE_CONTINUE;

    parsec_remote_deps_t *deps               = (parsec_remote_deps_t*)param;
    struct remote_dep_output_param_s* output = &deps->output[dep->dep_datatype_index];
    const parsec_task_class_t* fct           = newcontext->task_class;
    uint32_t flow_mask                       = (1U << dep->flow->flow_index) | 0x80000000;  /* in flow */
    /* Extract the datatype, count and displacement from the target task */
    if( PARSEC_HOOK_RETURN_DONE == fct->get_datatype(eu, newcontext, &flow_mask, &output->data) ) {
        /* something is wrong, we are unable to extract the expected datatype
         from the receiver task. At this point it is difficult to stop the
         algorithm, so let's assume the send datatype is to be used instead.*/
        output->data = *out_data;
    }

    parsec_data_t* data_arena = is_read_only(oldcontext, dep);
    if(NULL == data_arena) {
        output->deps_mask &= ~(1U << dep->dep_index); /* unmark all data that are RO we already hold from previous tasks */
    } else {
        output->deps_mask |= (1U << dep->dep_index); /* mark all data that are not RO */
        data_arena = is_inplace(oldcontext, dep);  /* Can we do it inplace */
    }
    output->data.data = NULL;

    if( deps->max_priority < newcontext->priority ) deps->max_priority = newcontext->priority;
    deps->incoming_mask |= (1U << dep->dep_datatype_index);
    deps->root           = src_rank;
    return PARSEC_ITERATE_STOP;
}

/**
 * Retrieve the datatypes involved in this communication. In addition the flag
 * PARSEC_ACTION_RECV_INIT_REMOTE_DEPS set the priority to the maximum priority
 * of all the children.
 */
static int
remote_dep_get_datatypes(parsec_execution_stream_t* es,
                         parsec_remote_deps_t* origin,
                         int storage_id, int *position)
{
    parsec_task_t task;
    uint32_t i, j, k, local_mask = 0;

    assert(NULL == origin->taskpool);
    origin->taskpool = parsec_taskpool_lookup(origin->msg.taskpool_id);
    if( NULL == origin->taskpool )
        return -1; /* the parsec taskpool doesn't exist yet */

    task.taskpool   = origin->taskpool;
    /* Do not set the task.task_class here, because it might trigger a race condition in DTD */
    task.priority = 0;  /* unknown yet */

    /* This function is divided into DTD and PTG's logic */
    if( PARSEC_TASKPOOL_TYPE_DTD == origin->taskpool->taskpool_type ) {
        parsec_dtd_taskpool_t *dtd_tp = NULL;
        parsec_dtd_task_t *dtd_task = NULL;

        dtd_tp = (parsec_dtd_taskpool_t *)origin->taskpool;

        /* if( NULL == task.task_class ), this case will be taken care of automatically */

        /* We need to convert from a dep_datatype_index mask into a dep_index
         * mask. However, in order to be able to use the above iterator we need to
         * be able to identify the dep_index for each particular datatype index, and
         * call the iterate_successors on each of the dep_index sets.
         */
        int return_defer = 0;
        for(k = 0; origin->msg.output_mask>>k; k++) {
            if(!(origin->msg.output_mask & (1U<<k))) continue;

            uint64_t key = (uint64_t)origin->msg.locals[0].value<<32 | (1U<<k);
            local_mask = 0;
            local_mask |= (1U<<k);

            parsec_hash_table_lock_bucket(dtd_tp->task_hash_table, (parsec_key_t)key);
            dtd_task = parsec_dtd_find_task( dtd_tp, key );

            if( NULL == dtd_task ) {
                return_defer = 1;

                /* AM buffers are reused by the comm engine once the activation
                 * has been conveyed to upper layer. In case of DTD we might receive msg to
                 * activate a task that the local node (the recipient of the activation)
                 * have not discovered yet. In that case we need to store the buffer,
                 * but note, we only need to store it the first time we are receiving this
                 * activation. PARSEC_DTD_SKIP_SAVING indicates whether this is the first
                 * time or not. Since, this function is called from other places (when
                 * we later try to activate a task for which we have already received
                 * an activation for) we do not need to store the buffer and we send
                 * PARSEC_DTD_SKIP_SAVING as an indicaton of that.
                 */
                if( storage_id != PARSEC_DTD_SKIP_SAVING) {
                    char* packed_buffer;
                    /* Copy the eager data to some temp storage */
                    packed_buffer = malloc(origin->msg.length);
                    memcpy(packed_buffer, dep_activate_buff[storage_id] + *position, origin->msg.length);
                    *position += origin->msg.length;  /* move to the next order */
                    origin->taskpool = (parsec_taskpool_t*)packed_buffer;  /* temporary storage */
                }
                parsec_dtd_track_remote_dep( dtd_tp, key, origin );
            }

            parsec_hash_table_unlock_bucket(dtd_tp->task_hash_table, (parsec_key_t)key);

            if(return_defer) {
                return -2;
            }

            PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tRetrieve datatype with mask 0x%x (remote_dep_get_datatypes)", local_mask);
            task.locals[k] = origin->msg.locals[k];
            task.task_class = dtd_task->super.task_class;
            origin->msg.task_class_id = dtd_task->super.task_class->task_class_id;
            task.task_class->iterate_successors(es, (parsec_task_t *)dtd_task,
                                               local_mask,
                                               remote_dep_mpi_retrieve_datatype,
                                               origin);
        }
    } else {
        task.task_class = task.taskpool->task_classes_array[origin->msg.task_class_id];

        for(i = 0; i < task.task_class->nb_locals; i++)
            task.locals[i] = origin->msg.locals[i];

        /* We need to convert from a dep_datatype_index mask into a dep_index
         * mask. However, in order to be able to use the above iterator we need to
         * be able to identify the dep_index for each particular datatype index, and
         * call the iterate_successors on each of the dep_index sets.
         */
        for(k = 0; origin->msg.output_mask>>k; k++) {
            if(!(origin->msg.output_mask & (1U<<k))) continue;
            for(local_mask = i = 0; NULL != task.task_class->out[i]; i++ ) {
                if(!(task.task_class->out[i]->flow_datatype_mask & (1U<<k))) continue;
                for(j = 0; NULL != task.task_class->out[i]->dep_out[j]; j++ )
                    if(k == task.task_class->out[i]->dep_out[j]->dep_datatype_index)
                        local_mask |= (1U << task.task_class->out[i]->dep_out[j]->dep_index);
                if( 0 != local_mask ) break;  /* we have our local mask, go get the datatype */
            }

            PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tRetrieve datatype with mask 0x%x (remote_dep_get_datatypes)", local_mask);
            task.task_class->iterate_successors(es, &task,
                                                local_mask,
                                                remote_dep_mpi_retrieve_datatype,
                                                origin);
        }
    }

    /**
     * At this point the msg->output_mask contains the root mask, and should be
     * keep as is and be propagated down the communication pattern. On the
     * origin->incoming_mask we have the mask of all local data to be retrieved
     * from the predecessor.
     */
    origin->outgoing_mask = origin->incoming_mask;  /* safekeeper */
    return 0;
}

/**
 * Trigger the local reception of a remote task data. Upon completion of all
 * pending receives related to a remote task completion, we call the
 * release_deps to enable all local tasks and then start the activation
 * propagation.
 */
static parsec_remote_deps_t*
remote_dep_release_incoming(parsec_execution_stream_t* es,
                            parsec_remote_deps_t* origin,
                            remote_dep_datakey_t complete_mask)
{
    parsec_task_t task;
    const parsec_flow_t* target;
    int i, pidx;
    uint32_t action_mask = 0;

    /* Update the mask of remaining dependencies to avoid releasing the same outputs twice */
    assert((origin->incoming_mask & complete_mask) == complete_mask);
    origin->incoming_mask ^= complete_mask;

    task.taskpool = origin->taskpool;
    task.task_class = task.taskpool->task_classes_array[origin->msg.task_class_id];
    task.priority = origin->priority;
    for(i = 0; i < task.task_class->nb_locals;
        task.locals[i] = origin->msg.locals[i], i++);
    for(i = 0; i < task.task_class->nb_flows;
        task.data[i].data_in = task.data[i].data_out = NULL, task.data[i].data_repo = NULL, i++);

    for(i = 0; complete_mask>>i; i++) {
        assert(i < MAX_PARAM_COUNT);
        if( !((1U<<i) & complete_mask) ) continue;
        pidx = 0;
        target = task.task_class->out[pidx];
        while( !((1U<<i) & target->flow_datatype_mask) ) {
            target = task.task_class->out[++pidx];
            assert(NULL != target);
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tDATA %p(%s) released from %p[%d] flow idx %d",
                origin->output[i].data.data, target->name, origin, i, target->flow_index);
        task.data[target->flow_index].data_repo = NULL;
        task.data[target->flow_index].data_in   = origin->output[i].data.data;
        task.data[target->flow_index].data_out  = origin->output[i].data.data;
    }

#ifdef PARSEC_DIST_COLLECTIVES
    /* Corresponding comment below on the propagation part */
    if(0 == origin->incoming_mask && PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type) {
        remote_dep_inc_flying_messages(task.taskpool);
        (void)parsec_atomic_fetch_inc_int32(&origin->pending_ack);
    }
#endif  /* PARSEC_DIST_COLLECTIVES */

    if(PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type) {
        /* We need to convert from a dep_datatype_index mask into a dep_index mask */
        for(int i = 0; NULL != task.task_class->out[i]; i++ ) {
            target = task.task_class->out[i];
            if( !(complete_mask & target->flow_datatype_mask) ) continue;
            for(int j = 0; NULL != target->dep_out[j]; j++ )
                if(complete_mask & (1U << target->dep_out[j]->dep_datatype_index))
                    action_mask |= (1U << target->dep_out[j]->dep_index);
        }
    } else if(PARSEC_TASKPOOL_TYPE_DTD == origin->taskpool->taskpool_type) {
        action_mask = complete_mask;
    } else {
        assert(0);
    }
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tTranslate mask from 0x%lx to 0x%x (remote_dep_release_incoming)",
            complete_mask, action_mask);
    (void)task.task_class->release_deps(es, &task,
                                        action_mask | PARSEC_ACTION_RELEASE_LOCAL_DEPS,
                                        NULL);
    assert(0 == (origin->incoming_mask & complete_mask));

    if(0 != origin->incoming_mask)  /* not done receiving */
        return origin;

    /**
     * All incoming data are now received, start the propagation. We first
     * release the local dependencies, thus we must ensure the communication
     * engine is not prevented from completing the propagation (the code few
     * lines above). Once the propagation is started we can release the
     * references on the allocated data and on the dependency.
     */
    uint32_t mask = origin->outgoing_mask;
    origin->outgoing_mask = 0;

#if defined(PARSEC_DIST_COLLECTIVES)
    if( PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type ) /* indicates it is a PTG taskpool */
        parsec_remote_dep_propagate(es, &task, origin);
#endif  /* PARSEC_DIST_COLLECTIVES */
    /**
     * Release the dependency owned by the communication engine for all data
     * internally allocated by the engine.
     */
    for(i = 0; mask>>i; i++) {
        assert(i < MAX_PARAM_COUNT);
        if( !((1U<<i) & mask) ) continue;
        if( NULL != origin->output[i].data.data )  /* except CONTROLs */
            PARSEC_DATA_COPY_RELEASE(origin->output[i].data.data);
    }
#if defined(PARSEC_DIST_COLLECTIVES)
    if(PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type) {
        remote_dep_complete_and_cleanup(&origin, 1);
    } else {
        remote_deps_free(origin);
    }
#else
    remote_deps_free(origin);
#endif  /* PARSEC_DIST_COLLECTIVES */

    return NULL;
}

#ifndef PARSEC_REMOTE_DEP_USE_THREADS
static int remote_dep_dequeue_nothread_init(parsec_context_t* context)
{
    parsec_dequeue_construct(&dep_cmd_queue);
    parsec_list_construct(&dep_cmd_fifo);
    if(-1 == context->comm_ctx) {
        MPI_Comm comm;
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        context->comm_ctx = (intptr_t)comm;
        assert(-1 != context->comm_ctx /* -1 reserved for non-initialized */);
    }
    return remote_dep_mpi_setup(context);
}

static int remote_dep_dequeue_nothread_fini(parsec_context_t* context)
{
    remote_dep_mpi_fini(context);
    parsec_list_destruct(&dep_cmd_fifo);
    parsec_dequeue_destruct(&dep_cmd_queue);
    return 0;
}
#endif

static int
remote_dep_dequeue_nothread_progress(parsec_execution_stream_t* es,
                                     int cycles)
{
    parsec_context_t* context = es->virtual_process->parsec_context;
    parsec_list_item_t *items;
    dep_cmd_item_t *item, *same_pos = NULL;
    parsec_list_t temp_list;
    int ret = 0, how_many, position, executed_tasks = 0;

    PARSEC_OBJ_CONSTRUCT(&temp_list, parsec_list_t);
 check_pending_queues:
    if( cycles >= 0 )
        if( 0 == cycles--) return executed_tasks;  /* report how many events were progressed */

    /* Move a number of transfers from the shared dequeue into our ordered lifo. */
    how_many = 0;
    while( NULL != (item = (dep_cmd_item_t*) parsec_dequeue_try_pop_front(&dep_cmd_queue)) ) {
        if( DEP_CTL == item->action ) {
            /* A DEP_CTL is a barrier that must not be crossed, flush the
             * ordered fifo and don't add anything until it is consumed */
            if( parsec_list_nolock_is_empty(&dep_cmd_fifo) && parsec_list_nolock_is_empty(&temp_list) )
                goto handle_now;
            parsec_dequeue_push_front(&dep_cmd_queue, (parsec_list_item_t*)item);
            break;
        }
        how_many++;
        same_pos = NULL;
        /* Find the position in the array of the first possible item in the same category */
        position = (DEP_ACTIVATE == item->action) ? item->cmd.activate.peer : (context->nb_nodes + item->action);

        parsec_list_item_singleton(&item->pos_list);
        same_pos = parsec_mpi_same_pos_items[position];
        if((NULL != same_pos) && (same_pos->priority >= item->priority)) {
            /* insert the item in the peer list */
            parsec_list_item_ring_push_sorted(&same_pos->pos_list, &item->pos_list, dep_mpi_pos_list);
        } else {
            if(NULL != same_pos) {
                /* this is the new head of the list. */
                parsec_list_item_ring_push(&same_pos->pos_list, &item->pos_list);
                /* Remove previous elem from the priority list. The element
                 might be either in the dep_cmd_fifo if it is old enough to be
                 pushed there, or in the temp_list waiting to be moved
                 upstream. Pay attention from which queue it is removed. */
#if defined(PARSEC_DEBUG_PARANOID)
                parsec_list_nolock_remove((struct parsec_list_t*)same_pos->super.belong_to, (parsec_list_item_t*)same_pos);
#else
                parsec_list_nolock_remove(NULL, (parsec_list_item_t*)same_pos);
#endif
                parsec_list_item_singleton((parsec_list_item_t*)same_pos);
            }
            parsec_mpi_same_pos_items[position] = item;
            /* And add ourselves in the temp list */
            parsec_list_nolock_push_front(&temp_list, (parsec_list_item_t*)item);
        }
        if(how_many > parsec_param_nb_tasks_extracted)
            break;
    }
    if( !parsec_list_nolock_is_empty(&temp_list) ) {
        /* Sort the temporary list */
        parsec_list_nolock_sort(&temp_list, dep_cmd_prio);
        /* Remove the ordered items from the list, and clean the list */
        items = parsec_list_nolock_unchain(&temp_list);
        /* Insert them into the locally ordered cmd_fifo */
        parsec_list_nolock_chain_sorted(&dep_cmd_fifo, items, dep_cmd_prio);
    }
    /* Extract the head of the list and point the array to the correct value */
    if(NULL == (item = (dep_cmd_item_t*)parsec_list_nolock_pop_front(&dep_cmd_fifo)) ) {
        do {
            ret = remote_dep_mpi_progress(es);
        } while(ret);

        if( !ret
         && ((comm_yield == 2)
          || (comm_yield == 1
           && !parsec_list_nolock_is_empty(&dep_activates_fifo)
           && !parsec_list_nolock_is_empty(&dep_put_fifo))) ) {
            struct timespec ts;
            ts.tv_sec = 0; ts.tv_nsec = comm_yield_ns;
            nanosleep(&ts, NULL);
        }
        goto check_pending_queues;
    }
    assert(DEP_CTL != item->action);
    executed_tasks++;  /* count all the tasks executed during this call */
  handle_now:
    position = (DEP_ACTIVATE == item->action) ? item->cmd.activate.peer : (context->nb_nodes + item->action);
    switch(item->action) {
    case DEP_CTL:
        ret = item->cmd.ctl.enable;
        PARSEC_OBJ_DESTRUCT(&temp_list);
        PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "rank %d DISABLE MPI communication engine", parsec_debug_rank);
        free(item);
        return ret;  /* FINI or OFF */
    case DEP_NEW_TASKPOOL:
        remote_dep_mpi_new_taskpool(es, item);
        break;
    case DEP_DTD_DELAYED_RELEASE:
        remote_dep_mpi_release_delayed_deps(es, item);
        break;
    case DEP_ACTIVATE:
        remote_dep_nothread_send(es, &item);
        same_pos = item;
        goto have_same_pos;
    case DEP_PUT_DATA:
#if 0 != RDEP_MSG_EAGER_LIMIT
        remote_dep_mpi_put_eager(es, item);
        same_pos = NULL;
#else
        assert("This should never be called!");
#endif  /* 0 != RDEP_MSG_EAGER_LIMIT */
        goto have_same_pos;
    case DEP_MEMCPY:
        remote_dep_nothread_memcpy(es, item);
        break;
    default:
        assert(0 && item->action); /* Not a valid action */
        break;
    }

    /* Correct the other structures */
    same_pos = (dep_cmd_item_t*)parsec_list_item_ring_chop(&item->pos_list);
    if( NULL != same_pos)
        same_pos = container_of(same_pos, dep_cmd_item_t, pos_list);
    free(item);
  have_same_pos:
    if( NULL != same_pos) {
        parsec_list_nolock_push_front(&temp_list, (parsec_list_item_t*)same_pos);
        /* if we still have pending messages of the same type, stay here for an extra loop */
        if( cycles >= 0 ) cycles++;
    }
    parsec_mpi_same_pos_items[position] = same_pos;

    goto check_pending_queues;
}

/******************************************************************************
 * ALL MPI SPECIFIC CODE GOES HERE
 ******************************************************************************/
enum {
    REMOTE_DEP_ACTIVATE_TAG = 0,
    REMOTE_DEP_GET_DATA_TAG,
    REMOTE_DEP_MAX_CTRL_TAG
} parsec_remote_dep_tag_t;

#ifdef PARSEC_PROF_TRACE
static parsec_thread_profiling_t* MPIctl_prof;
static parsec_thread_profiling_t* MPIsnd_prof;
static parsec_thread_profiling_t* MPIrcv_prof;
static int MPI_Activate_sk, MPI_Activate_ek;
static int64_t get = 0;
static int MPI_Data_ctl_sk, MPI_Data_ctl_ek;
static int MPI_Data_plds_sk, MPI_Data_plds_ek;
static int MPI_Data_pldr_sk, MPI_Data_pldr_ek;
static int activate_cb_trace_sk, activate_cb_trace_ek;
static int put_cb_trace_sk, put_cb_trace_ek;

/**
 * The structure describe the MPI events saves into the profiling stream. The following
 * string represent it's description so that an external package can decrypt the
 * binary format of the stream.
 */
typedef struct {
    int rank_src;  // 0
    int rank_dst;  // 4
    uint64_t tid;  // 8
    uint32_t tpid;  // 16
    uint32_t tcid;  // 20
} parsec_profile_remote_dep_mpi_info_t; // 24 bytes

static char parsec_profile_remote_dep_mpi_info_to_string[] = "src{int32_t};dst{int32_t};tid{int64_t};tpid{int32_t};tcid{int32_t}";

static void remote_dep_mpi_profiling_init(void)
{
    parsec_profiling_add_dictionary_keyword( "MPI_ACTIVATE", "fill:#FF0000",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Activate_sk, &MPI_Activate_ek);
    parsec_profiling_add_dictionary_keyword( "MPI_DATA_CTL", "fill:#000077",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Data_ctl_sk, &MPI_Data_ctl_ek);
    parsec_profiling_add_dictionary_keyword( "MPI_DATA_PLD_SND", "fill:#B08080",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Data_plds_sk, &MPI_Data_plds_ek);
    parsec_profiling_add_dictionary_keyword( "MPI_DATA_PLD_RCV", "fill:#80B080",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Data_pldr_sk, &MPI_Data_pldr_ek);

    parsec_profiling_add_dictionary_keyword( "ACTIVATE_CB", "fill:#FF0000",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &activate_cb_trace_sk, &activate_cb_trace_ek);
    parsec_profiling_add_dictionary_keyword( "PUT_CB", "fill:#FF0000",
                                            sizeof(parsec_profile_remote_dep_mpi_info_t),
                                            parsec_profile_remote_dep_mpi_info_to_string,
                                            &put_cb_trace_sk, &put_cb_trace_ek);

    MPIctl_prof = parsec_profiling_thread_init( 2*1024*1024, "MPI ctl");
    MPIsnd_prof = parsec_profiling_thread_init( 2*1024*1024, "MPI isend");
    MPIrcv_prof = parsec_profiling_thread_init( 2*1024*1024, "MPI irecv");
    parsec_comm_es.es_profile = MPIctl_prof;
}

static void remote_dep_mpi_profiling_fini(void)
{
    /* TODO: we need to clean the profiling threads memory */
    MPIsnd_prof = NULL;
    MPIrcv_prof = NULL;
    MPIctl_prof = NULL;
}

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
#define remote_dep_mpi_profiling_init() do {} while(0)
#define remote_dep_mpi_profiling_fini() do {} while(0)
#endif  /* PARSEC_PROF_TRACE */

typedef int (*parsec_comm_callback_f)(parsec_execution_stream_t*,
                                     parsec_comm_callback_t*,  /**< the associated callback structure */
                                     MPI_Status* status);     /**< the corresponding status */
struct parsec_comm_callback_s {
    parsec_comm_callback_f fct;
    long                  storage1;
    long                  storage2;
};

static parsec_comm_callback_t *array_of_callbacks;
static MPI_Request            *array_of_requests;
static int                    *array_of_indices;
static MPI_Status             *array_of_statuses;

/* TODO: fix heterogeneous restriction by using proper mpi datatypes */
#define dep_dtt MPI_BYTE
#define dep_count sizeof(remote_dep_wire_activate_t)
#define dep_extent dep_count
#define DEP_SHORT_BUFFER_SIZE (dep_extent+RDEP_MSG_SHORT_LIMIT)
#define datakey_dtt MPI_LONG
#define datakey_count 3
static remote_dep_wire_get_t* dep_get_buff;

/* Pointers are converted to long to be used as keys to fetch data in the get
 * rdv protocol. Make sure we can carry pointers correctly.
 */
#ifdef PARSEC_HAVE_LIMITS_H
#include <limits.h>
#endif
#if ULONG_MAX < UINTPTR_MAX
#error "unsigned long is not large enough to hold a pointer!"
#endif

/* note: tags are necessary to order communication between pairs. They are used to
 * correctly handle data transfers, as each data provider will provide a tag which
 * combined with the source ensure message matching consistency. As MPI requires the
 * max tag to be positive, initializing it to a negative value allows us to check
 * if the layer has been initialized or not.
 */
static int MAX_MPI_TAG = -1, mca_tag_ub = -1;
#define MIN_MPI_TAG (REMOTE_DEP_MAX_CTRL_TAG+1)
static volatile int __VAL_NEXT_TAG = MIN_MPI_TAG;
#if INT_MAX == INT32_MAX
#define next_tag_cas(t, o, n) parsec_atomic_cas_int32(t, o, n)
#elif INT_MAX == INT64_MAX
#define next_tag_cas(t, o, n) parsec_atomic_cas_int64(t, o, n)
#else
#error "next_tag_cas written to support sizeof(int) of 4 or 8"
#endif
static inline int next_tag(int k) {
    int __tag, __tag_o, __next_tag;
reread:
    __tag = __tag_o = __VAL_NEXT_TAG;
    if( __tag > (MAX_MPI_TAG-k) ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "rank %d tag rollover: min %d < %d (+%d) < max %d", parsec_debug_rank,
                MIN_MPI_TAG, __tag, k, MAX_MPI_TAG);
        __tag = MIN_MPI_TAG;
    }
    __next_tag = __tag+k;

    if( parsec_comm_es.virtual_process->parsec_context->flags & PARSEC_CONTEXT_FLAG_COMM_MT ) {
        if(!next_tag_cas(&__VAL_NEXT_TAG, __tag_o, __next_tag)) {
            goto reread;
        }
    }
    else {
        __VAL_NEXT_TAG = __next_tag;
    }
    return __tag;
}

/**
 * The following 2 functions take care of all the steps necessary to initialize the
 * invariable part of the communication engine such as the const dependencies
 * to MPI (max tag and other global info), or local objects.
 */
static int remote_dep_mpi_init_once(parsec_context_t* context)
{
    int mpi_tag_ub_exists, *ub;

    assert(-1 == MAX_MPI_TAG);
    PARSEC_OBJ_CONSTRUCT(&dep_activates_fifo, parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&dep_activates_noobj_fifo, parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&dep_put_fifo, parsec_list_t);

    assert(MPI_COMM_NULL == dep_self);
    MPI_Comm_dup(MPI_COMM_SELF, &dep_self);
    assert(MPI_COMM_NULL == dep_comm);

    /*
     * Based on MPI 1.1 the MPI_TAG_UB should only be defined
     * on MPI_COMM_WORLD.
     */
#if defined(PARSEC_HAVE_MPI_20)
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &mpi_tag_ub_exists);
#else
    MPI_Attr_get(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &mpi_tag_ub_exists);
#endif  /* defined(PARSEC_HAVE_MPI_20) */

    parsec_mca_param_reg_int_name("mpi", "tag_ub",
                                  "The upper bound of the TAG used by the MPI communication engine. Bounded by the MPI_TAG_UB attribute on the MPI implementation MPI_COMM_WORLD. (-1 for MPI default)",
                                  false, false, -1, &mca_tag_ub);

    if( !mpi_tag_ub_exists ) {
        MAX_MPI_TAG = (-1 == mca_tag_ub) ? INT_MAX : mca_tag_ub;
        parsec_warning("Your MPI implementation does not define MPI_TAG_UB and thus violates the standard (MPI-2.2, page 29, line 30). The max tag is therefore set using the MCA mpi_tag_ub (current value %d).\n", MAX_MPI_TAG);
    } else {
        MAX_MPI_TAG = ((-1 == mca_tag_ub) || (mca_tag_ub > *ub)) ? *ub : mca_tag_ub;
    }
    if( MAX_MPI_TAG < INT_MAX ) {
        parsec_debug_verbose(3, parsec_comm_output_stream,
                             "MPI:\tYour MPI implementation defines the maximal TAG value to %d (0x%08x),"
                             " which might be too small should you have more than %d pending remote dependencies",
                             MAX_MPI_TAG, (unsigned int)MAX_MPI_TAG, MAX_MPI_TAG / MAX_DEP_OUT_COUNT);
    }

    remote_dep_mpi_profiling_init();
    (void)context;
    return 0;
}

/**
 * The communication engine is now completely disabled. All internal resources
 * are released, and no future communications are possible.
 * Anything initialized in init_once must be disposed off here
 */
static int remote_dep_mpi_fini(parsec_context_t* context)
{
    assert( -1 != MAX_MPI_TAG );
    remote_dep_mpi_cleanup(context);

    /* Remove the static handles */
    MPI_Comm_free(&dep_self); /* dep_self becomes MPI_COMM_NULL */

    /* Release the context communicators if any */
    if( -1 != context->comm_ctx) {
        MPI_Comm_free((MPI_Comm*)&context->comm_ctx);
        context->comm_ctx = -1; /* We use -1 for the opaque comm_ctx, rather than the MPI specific MPI_COMM_NULL */
    }

    PARSEC_OBJ_DESTRUCT(&dep_activates_fifo);
    PARSEC_OBJ_DESTRUCT(&dep_activates_noobj_fifo);
    PARSEC_OBJ_DESTRUCT(&dep_put_fifo);
    MAX_MPI_TAG = -1;  /* mark the layer as uninitialized */
    remote_dep_mpi_profiling_fini();

    (void)context;
    return 0;
}

static int remote_dep_mpi_setup(parsec_context_t* context)
{
    parsec_comm_callback_t* cb;
    int i, rc;

    if( -1 == MAX_MPI_TAG )
        if( 0 != (rc = remote_dep_mpi_init_once(context)) ) {
            parsec_debug_verbose(3, parsec_comm_output_stream, "MPI: Failed to correctly retrieve the max TAG."
                                 " PaRSEC cannot continue using MPI\n");
            return rc;
        }

    /* Did anything changed that would require a build of the management structures? */
    assert(-1 != context->comm_ctx);
    if(dep_comm == (MPI_Comm)context->comm_ctx) {
        return 0;
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "rank %d ENABLE MPI communication engine",
                         parsec_debug_rank);
    if(MPI_COMM_NULL != dep_comm) {
        parsec_debug_verbose(3, parsec_comm_output_stream, "MPI: Rearming the dep_comm and dep_self.");
        /* Cleanup prior setup */
        remote_dep_mpi_cleanup(context);
    }
    assert(-1 != context->comm_ctx);
    dep_comm = (MPI_Comm) context->comm_ctx;

#if defined(PARSEC_HAVE_MPI_OVERTAKE)
    if( parsec_param_enable_mpi_overtake ) {
        MPI_Info no_order;
        MPI_Info_create(&no_order);
        MPI_Info_set(no_order, "mpi_assert_allow_overtaking", "true");
        MPI_Comm_set_info(dep_comm, no_order);
        MPI_Info_free(&no_order);
    }
#endif

    MPI_Comm_size(dep_comm, &(context->nb_nodes));
    MPI_Comm_rank(dep_comm, &(context->my_rank));

    parsec_mpi_same_pos_items_size = context->nb_nodes + (int)DEP_LAST;
    parsec_mpi_same_pos_items = (dep_cmd_item_t**)calloc(parsec_mpi_same_pos_items_size,
                                                         sizeof(dep_cmd_item_t*));
    /* Extend the number of pending activations if we have a large number of peers */
    if( context->nb_nodes > (10*parsec_comm_activations_max) )
        parsec_comm_activations_max = context->nb_nodes / 10;
    if( context->nb_nodes > (10*parsec_comm_data_get_max) )
        parsec_comm_data_get_max = context->nb_nodes / 10;
    DEP_NB_REQ = (parsec_comm_activations_max + parsec_comm_data_get_max +
                  parsec_comm_gets_max + parsec_comm_puts_max);

    array_of_callbacks = (parsec_comm_callback_t*)calloc(DEP_NB_REQ, sizeof(parsec_comm_callback_t));
    array_of_requests  = (MPI_Request*)calloc(DEP_NB_REQ, sizeof(MPI_Request));
    array_of_indices   = (int*)calloc(DEP_NB_REQ, sizeof(int));
    array_of_statuses  = (MPI_Status*)calloc(DEP_NB_REQ, sizeof(MPI_Status));
    for(i = 0; i < DEP_NB_REQ; i++)
        array_of_requests[i] = MPI_REQUEST_NULL;

    /* Create all the persistent receives (activation and GET orders) and start them */
    dep_activate_buff = (char**)calloc(parsec_comm_activations_max, sizeof(char*));
    dep_activate_buff[0] = (char*)calloc(parsec_comm_activations_max, DEP_SHORT_BUFFER_SIZE*sizeof(char));
    for(i = 0; i < parsec_comm_activations_max; i++) {
        dep_activate_buff[i] = dep_activate_buff[0] + i * DEP_SHORT_BUFFER_SIZE*sizeof(char);
        MPI_Recv_init(dep_activate_buff[i], DEP_SHORT_BUFFER_SIZE, MPI_PACKED,
                      MPI_ANY_SOURCE, REMOTE_DEP_ACTIVATE_TAG, dep_comm,
                      &array_of_requests[parsec_comm_last_active_req]);
        cb = &array_of_callbacks[parsec_comm_last_active_req];
        cb->fct      = remote_dep_mpi_save_activate_cb;
        cb->storage1 = parsec_comm_last_active_req;
        cb->storage2 = i;
        MPI_Start(&array_of_requests[parsec_comm_last_active_req]);
        parsec_comm_last_active_req++;
    }

    dep_get_buff = (remote_dep_wire_get_t*)calloc(parsec_comm_data_get_max, sizeof(remote_dep_wire_get_t));
    for(i = 0; i < parsec_comm_data_get_max; i++) {
        MPI_Recv_init(&dep_get_buff[i], datakey_count, datakey_dtt,
                      MPI_ANY_SOURCE, REMOTE_DEP_GET_DATA_TAG, dep_comm,
                      &array_of_requests[parsec_comm_last_active_req]);
        cb = &array_of_callbacks[parsec_comm_last_active_req];
        cb->fct      = remote_dep_mpi_save_put_cb;
        cb->storage1 = parsec_comm_last_active_req;
        cb->storage2 = i;
        MPI_Start(&array_of_requests[parsec_comm_last_active_req]);
        parsec_comm_last_active_req++;
    }

    return 0;
}

#include "parsec/utils/mca_param.h"

static void remote_dep_mpi_params(parsec_context_t* context) {
    (void)context;
#if RDEP_MSG_SHORT_LIMIT != 0
    parsec_mca_param_reg_sizet_name("runtime", "comm_short_limit", "Controls the maximum size of a short message. Short messages contain both the control message notifying the completion of a task and the associated data that fit completely in that buffer length. The maximum size of a short message should be lower than the network MTU.",
                                  false, false, parsec_param_short_limit, &parsec_param_short_limit);
    if(parsec_param_short_limit > RDEP_MSG_SHORT_LIMIT) {
        parsec_warning("User requested a short message limit of %d which is greater than compiled in limit %d; value reset to compiled limit %d", parsec_param_short_limit, RDEP_MSG_SHORT_LIMIT, RDEP_MSG_SHORT_LIMIT);
        parsec_param_short_limit = RDEP_MSG_SHORT_LIMIT;
    }
#endif
#if RDEP_MSG_EAGER_LIMIT != 0
    parsec_mca_param_reg_sizet_name("runtime", "comm_eager_limit", "Controls the maximum size of a message that uses the eager protocol. Eager messages are sent eagerly before a 2-sided synchronization and may cause flow control and memory contentions at the receiver, but have a better latency.",
                                  false, false, parsec_param_eager_limit, &parsec_param_eager_limit);
#endif
    parsec_mca_param_reg_int_name("runtime", "comm_aggregate", "Aggregate multiple dependencies in the same short message (1=true,0=false).",
                                  false, false, parsec_param_enable_aggregate, &parsec_param_enable_aggregate);
}

void
remote_dep_mpi_initialize_execution_stream(parsec_context_t *context)
{
    memcpy(&parsec_comm_es, context->virtual_processes[0]->execution_streams[0], sizeof(parsec_execution_stream_t));
}

/**
 * This function should be called to release all internal structures
 * needed to handle the internals of the communication engine. Upon
 * return however, if the engine is to be reused it has to be reinitialized.
 * In general this function is called either upon finalization of the PaRSEC
 * engine or by the communication thread early during the startup process, in
 * order to prepare for message exchange.
 * BEWARE: this function release the communicator, it has to be called collectively.
 */
static int remote_dep_mpi_cleanup(parsec_context_t* context)
{
    int i, flag;
    MPI_Status status;

    /* Cancel and release all persistent requests */
    for(i = 0; i < parsec_comm_activations_max + parsec_comm_data_get_max; i++) {
        MPI_Cancel(&array_of_requests[i]);
        MPI_Test(&array_of_requests[i], &flag, &status);
        MPI_Request_free(&array_of_requests[i]);
        assert( MPI_REQUEST_NULL == array_of_requests[i] );
    }
    parsec_comm_last_active_req -= (parsec_comm_activations_max + parsec_comm_data_get_max);
    assert(0 == parsec_comm_last_active_req);

    free(array_of_callbacks); array_of_callbacks = NULL;
    free(array_of_requests);  array_of_requests  = NULL;
    free(array_of_indices);   array_of_indices   = NULL;
    free(array_of_statuses);  array_of_statuses  = NULL;

    free(parsec_mpi_same_pos_items); parsec_mpi_same_pos_items = NULL;
    parsec_mpi_same_pos_items_size = 0;

    free(dep_get_buff); dep_get_buff = NULL;
    free(dep_activate_buff[0]);
    free(dep_activate_buff); dep_activate_buff = NULL;

    /* Force a reconstruction of the internal dep_comm but without free it, in order
     * to avoid releasing the context->comm_ctx.
     */
    dep_comm = MPI_COMM_NULL;

    (void)context;
    return 0;
}

static int remote_dep_mpi_on(parsec_context_t* context)
{
    remote_dep_mpi_setup(context);
#ifdef PARSEC_PROF_TRACE
    /* put a start marker on each line */
    TAKE_TIME(MPIctl_prof, MPI_Activate_sk, 0);
    TAKE_TIME(MPIsnd_prof, MPI_Activate_sk, 0);
    TAKE_TIME(MPIrcv_prof, MPI_Activate_sk, 0);
    MPI_Barrier(dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, 0);
    TAKE_TIME(MPIsnd_prof, MPI_Activate_ek, 0);
    TAKE_TIME(MPIrcv_prof, MPI_Activate_ek, 0);
#endif
    (void)context;
    return 0;
}

/**
 * Given a remote_dep_wire_activate message it packs as much as possible
 * into the provided buffer. If possible (short allowed and enough room
 * in the buffer) some of the arguments will also be packed. Beware, the
 * remote_dep_wire_activate message itself must be updated with the
 * correct length before packing.
 *
 * @returns 1 if the message can't be packed due to lack of space, or 0
 * otherwise.
 */
static int remote_dep_mpi_pack_dep(int peer,
                                   dep_cmd_item_t* item,
                                   char* packed_buffer,
                                   int length,
                                   int* position)
{
    parsec_remote_deps_t *deps = (parsec_remote_deps_t*)item->cmd.activate.task.deps;
    remote_dep_wire_activate_t* msg = &deps->msg;
    int k, dsize, saved_position = *position;
    uint32_t peer_bank, peer_mask, expected = 0;
#if defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, 128);
#endif

    peer_bank = peer / (sizeof(uint32_t) * 8);
    peer_mask = 1U << (peer % (sizeof(uint32_t) * 8));

    MPI_Pack_size(dep_count, dep_dtt, dep_comm, &dsize);
    if( (length - (*position)) < dsize ) {  /* no room. bail out */
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "Can't pack at %d/%d. Bail out!", *position, length);
        return 1;
    }
    /* Don't pack yet, we need to update the length field before packing */
    *position += dsize;
    assert((0 != msg->output_mask) &&   /* this should be preset */
           (msg->output_mask & deps->outgoing_mask) == deps->outgoing_mask);
    msg->length = 0;
    item->cmd.activate.task.output_mask = 0;  /* clean start */
    /* Treat for special cases: CTL, Short, etc... */
    for(k = 0; deps->outgoing_mask >> k; k++) {
        if( !((1U << k) & deps->outgoing_mask )) continue;
        if( !(deps->output[k].rank_bits[peer_bank] & peer_mask) ) continue;

        /* Remove CTL from the message we expect to send */
#if defined(PARSEC_PROF_DRY_DEP)
        deps->output[k].data.arena = NULL; /* make all data a control */
#endif
        if(NULL == deps->output[k].data.arena) {
            PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, " CTL\t%s\tparam %d\tdemoted to be a control", tmp, k);
            continue;
        }
        assert(deps->output[k].data.count > 0);
        if(parsec_param_short_limit) {
            /* Embed data (up to short size) with the activate msg */
            MPI_Pack_size(deps->output[k].data.count, deps->output[k].data.layout,
                          dep_comm, &dsize);
            if((length - (*position)) >= dsize) {
                MPI_Pack((char*)PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data) + deps->output[k].data.displ,
                         deps->output[k].data.count, deps->output[k].data.layout,
                         packed_buffer, length, position, dep_comm);
                PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, " EGR\t%s\tparam %d\tshort piggyback in the activate msg (%d/%d)",
                        tmp, k, *position, length);
                msg->length += dsize;
                continue;  /* go to the next */
            } else if( 0 != saved_position ) {
                PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "DATA\t%s\tparam %d\texceed buffer length. Start again from here next iteration",
                        tmp, k);
                *position = saved_position;
                return 1;
            }
            /* the data doesn't fit in the buffer. */
        }
        expected++;
        item->cmd.activate.task.output_mask |= (1U<<k);
        PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "DATA\t%s\tparam %d\tdeps %p send on demand (increase deps counter by %d [%d])",
                tmp, k, deps, expected, deps->pending_ack);
    }
    if(expected)
        (void)parsec_atomic_fetch_add_int32(&deps->pending_ack, expected);  /* Keep track of the inflight data */
    /* We can only have up to k data sends related to this remote_dep (include the order itself) */
    item->cmd.activate.task.tag = next_tag(k);
    msg->tag = item->cmd.activate.task.tag;
#if defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER)
    parsec_debug_verbose(6, parsec_comm_output_stream, "MPI:\tTO\t%d\tActivate\t% -8s\n"
          "    \t\t\twith datakey %lx\tmask %lx\t(tag=%d) short mask %lu length %d",
          peer, tmp, msg->deps, msg->output_mask, msg->tag,
          msg->output_mask ^ item->cmd.activate.task.output_mask, msg->length);
#endif
    /* And now pack the updated message (msg->length and msg->output_mask) itself. */
    MPI_Pack(msg, dep_count, dep_dtt, packed_buffer, length, &saved_position, dep_comm);
    return 0;
}

/**
 * Perform a memcopy with datatypes by doing a local sendrecv.
 */
static int remote_dep_nothread_memcpy(parsec_execution_stream_t* es,
                                      dep_cmd_item_t *item)
{
    dep_cmd_t* cmd = &item->cmd;
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream,
                         "COPY LOCAL DATA from %p to %p count %d",
                         (char*)PARSEC_DATA_COPY_GET_PTR(cmd->memcpy.source     ) + cmd->memcpy.displ_s,
                         (char*)PARSEC_DATA_COPY_GET_PTR(cmd->memcpy.destination) + cmd->memcpy.displ_r,
                         cmd->memcpy.count);

    int rc = MPI_Sendrecv((char*)PARSEC_DATA_COPY_GET_PTR(cmd->memcpy.source     ) + cmd->memcpy.displ_s,
                          cmd->memcpy.count, cmd->memcpy.datatype, 0, 0,
                          (char*)PARSEC_DATA_COPY_GET_PTR(cmd->memcpy.destination) + cmd->memcpy.displ_r,
                          cmd->memcpy.count, cmd->memcpy.datatype, 0, 0,
                          dep_self, MPI_STATUS_IGNORE);
    PARSEC_DATA_COPY_RELEASE(cmd->memcpy.source);
    remote_dep_dec_flying_messages(item->cmd.memcpy.taskpool);
    (void)es;
    return (MPI_SUCCESS == rc ? 0 : -1);
}

/**
 * Starting with a particular item pack as many remote_dep_wire_activate
 * messages with the same destination (from the item ring associated with
 * pos_list) into a buffer. Upon completion the entire buffer is send to the
 * remote peer, the completed messages are released and the header is updated to
 * the next unsent message.
 */
static int remote_dep_nothread_send(parsec_execution_stream_t* es,
                                    dep_cmd_item_t **head_item)
{
    parsec_remote_deps_t *deps;
    dep_cmd_item_t *item = *head_item;
    parsec_list_item_t* ring = NULL;
    char packed_buffer[DEP_SHORT_BUFFER_SIZE];
    int peer, position = 0;

    peer = item->cmd.activate.peer;  /* this doesn't change */
    deps = (parsec_remote_deps_t*)item->cmd.activate.task.deps;
    TAKE_TIME_WITH_INFO(es->es_profile, MPI_Activate_sk, 0,
                        es->virtual_process->parsec_context->my_rank,
                        peer, deps->msg);
  pack_more:
    assert(peer == item->cmd.activate.peer);
    deps = (parsec_remote_deps_t*)item->cmd.activate.task.deps;

    parsec_list_item_singleton((parsec_list_item_t*)item);
    if( 0 == remote_dep_mpi_pack_dep(peer, item, packed_buffer,
                                     DEP_SHORT_BUFFER_SIZE, &position) ) {
        /* space left on the buffer. Move to the next item with the same destination */
        dep_cmd_item_t* next = (dep_cmd_item_t*)parsec_list_item_ring_chop(&item->pos_list);
        if( NULL == ring ) ring = (parsec_list_item_t*)item;
        else parsec_list_item_ring_push(ring, (parsec_list_item_t*)item);
        if( NULL != next ) {
            item = container_of(next, dep_cmd_item_t, pos_list);
            assert(DEP_ACTIVATE == item->action);
            if( parsec_param_enable_aggregate )
                goto pack_more;
        } else item = NULL;
    }
    *head_item = item;
    assert(NULL != ring);

    MPI_Send((void*)packed_buffer, position, MPI_PACKED, peer, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
    TAKE_TIME(es->es_profile, MPI_Activate_ek, 0);
    DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(peer, (void*)&deps->msg, &deps->msg);

    do {
        item = (dep_cmd_item_t*)ring;
        ring = parsec_list_item_ring_chop(ring);
        deps = (parsec_remote_deps_t*)item->cmd.activate.task.deps;

#if RDEP_MSG_EAGER_LIMIT != 0
        if( 0 != item->cmd.activate.task.output_mask ) {
            remote_dep_mpi_put_eager(es, item);
        } else
#endif   /* RDEP_MSG_EAGER_LIMIT != 0 */
            free(item);  /* only large messages are left */

        remote_dep_complete_and_cleanup(&deps, 1);
    } while( NULL != ring );
    (void)es;
    return 0;
}

static int remote_dep_mpi_progress(parsec_execution_stream_t* es)
{
    MPI_Status *status;
    int ret = 0, idx, outcount, pos;
    parsec_comm_callback_t* cb;

    if( !PARSEC_THREAD_IS_MASTER(es) ) return 0;

    do {
        MPI_Testsome(parsec_comm_last_active_req, array_of_requests,
                     &outcount, array_of_indices, array_of_statuses);
        if(0 == outcount) goto feed_more_work;  /* can we push some more work? */

        /* Trigger the callbacks */
        for( idx = 0; idx < outcount; idx++ ) {

            cb = &array_of_callbacks[array_of_indices[idx]];
            status = &(array_of_statuses[idx]);

            cb->fct(es, cb, status);
            ret++;
        }

        /* Compact the pending requests in order to minimize the testsome waiting time.
         * Parsing the array_of_indices in the reverse order insure a smooth and fast
         * compacting.
         */
        for( idx = outcount-1; idx >= 0; idx-- ) {
            pos = array_of_indices[idx];
            if(MPI_REQUEST_NULL != array_of_requests[pos])
                continue;  /* The callback replaced the completed request, keep going */
            /* Get the last active callback to replace the empty one */
            parsec_comm_last_active_req--;
            if( parsec_comm_last_active_req > pos ) {
                array_of_requests[pos]  = array_of_requests[parsec_comm_last_active_req];
                array_of_callbacks[pos] = array_of_callbacks[parsec_comm_last_active_req];
            }
            array_of_requests[parsec_comm_last_active_req] = MPI_REQUEST_NULL;
        }

      feed_more_work:
        if((parsec_comm_gets < parsec_comm_gets_max) && !parsec_list_nolock_is_empty(&dep_activates_fifo)) {
            parsec_remote_deps_t* deps = (parsec_remote_deps_t*)parsec_list_nolock_pop_front(&dep_activates_fifo);
            remote_dep_mpi_get_start(es, deps);
        }
        if((parsec_comm_puts < parsec_comm_puts_max) && !parsec_list_nolock_is_empty(&dep_put_fifo)) {
            dep_cmd_item_t* item = (dep_cmd_item_t*)parsec_list_nolock_pop_front(&dep_put_fifo);
            remote_dep_mpi_put_start(es, item);
        }
        if(0 == outcount) return ret;
    } while(1);
}

#if RDEP_MSG_EAGER_LIMIT != 0
/**
 * Compute the mask of all dependencies associated with a defined deps that can
 * be embedded in the outgoing message. This takes in account the control data
 * (with zero length), the short data up to the allowed max amount of the
 * message as well as the eager protocol (data that will follow eagerly without
 * a need for rendez-vous).
 */
static remote_dep_datakey_t
remote_dep_mpi_eager_which(const parsec_remote_deps_t* deps,
                           remote_dep_datakey_t output_mask)
{
    if( 0 == parsec_param_eager_limit &&
        0 == parsec_param_short_limit )  /* both disabled via MCA */
        return 0;

    for(int k = 0; output_mask>>k; k++) {
        if( !(output_mask & (1U<<k)) ) continue;            /* No dependency */
        if( NULL == deps->output[k].data.arena ) continue;  /* CONTROL dependency */
        size_t extent = deps->output[k].data.arena->elem_size * deps->output[k].data.count;

        if( (extent <= parsec_param_eager_limit) || (extent <= parsec_param_short_limit) ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tPEER\tNA\t%5s MODE  k=%d\tsize=%d <= %d\t(tag=base+%d)",
                    (extent <= (RDEP_MSG_SHORT_LIMIT) ? "Short" : "Eager"),
                    k, extent, RDEP_MSG_SHORT_LIMIT, k);
            continue;
        }
        output_mask ^= (1U<<k);
    }
    return output_mask;
}

static void remote_dep_mpi_put_eager(parsec_execution_stream_t* es,
                                     dep_cmd_item_t* item)
{
    remote_dep_wire_get_t* task = &item->cmd.activate.task;
    parsec_remote_deps_t* deps = (parsec_remote_deps_t*)task->deps;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN);
#endif

    item->cmd.activate.task.output_mask = remote_dep_mpi_eager_which(deps, task->output_mask);
    if( 0 == item->cmd.activate.task.output_mask ) {
        PARSEC_DEBUG_VERBOSE(100, parsec_comm_output_stream, "PUT_EAGER no data for item %p, freeing", item);
        free(item);  /* nothing to do, no reason to keep it */
        return;
    }

    if( es != &parsec_comm_es ) {
        /* The activate part is done by the caller thread. However, the
         * comm_thread will take care of the short PUTS in its context
         * to avoid thread synchronization cost on accesses to dep_put_fifo.
         * So, we insert the ACTIVATE as a PUT_DATA in the front of the 
         * cmd_queue for later handling in another call to this function
         * but appropriately thread-shifted. */
        item->action = DEP_PUT_DATA;
        PARSEC_DEBUG_VERBOSE(100, parsec_comm_output_stream, "PUT_DATA item %p enqueued", item);
        parsec_dequeue_push_front(&dep_cmd_queue, (parsec_list_item_t*)item);
    }
    else {
        /* Check if we can process it right now */
        if( parsec_comm_puts < parsec_comm_puts_max ) {
            remote_dep_mpi_put_start(es, item);
            return;
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: Put Eager DELAYED for %s from %d tag %u which 0x%x (deps %p)",
            tmp, item->cmd.activate.peer, task->tag, task->output_mask, deps);
        parsec_list_nolock_push_sorted(&dep_put_fifo, (parsec_list_item_t*)item, dep_cmd_prio);
    }
}
#endif  /* RDEP_MSG_EAGER_LIMIT != 0 */

static int
remote_dep_mpi_save_put_cb(parsec_execution_stream_t* es,
                           parsec_comm_callback_t* cb,
                           MPI_Status* status)
{
    remote_dep_wire_get_t* task;
    parsec_remote_deps_t *deps;
    dep_cmd_item_t* item;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    item = (dep_cmd_item_t*) malloc(sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(&item->super, parsec_list_item_t);
    item->action = DEP_GET_DATA;
    item->cmd.activate.peer = status->MPI_SOURCE;

    task = &(item->cmd.activate.task);
    memcpy(task, &dep_get_buff[cb->storage2], sizeof(remote_dep_wire_get_t));
    deps = (parsec_remote_deps_t*) (uintptr_t) task->deps;
    assert(0 != deps->pending_ack);
    assert(0 != deps->outgoing_mask);
    item->priority = deps->max_priority;

    /* Get the highest priority PUT operation */
    parsec_list_nolock_push_sorted(&dep_put_fifo, (parsec_list_item_t*)item, dep_cmd_prio);
    if( parsec_comm_puts < parsec_comm_puts_max ) {
        item = (dep_cmd_item_t*)parsec_list_nolock_pop_front(&dep_put_fifo);
        remote_dep_mpi_put_start(es, item);
    } else {
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI: Put DELAYED for %s from %d tag %u which 0x%x (deps %p)",
                remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), item->cmd.activate.peer,
                task->tag, task->output_mask, (void*)deps);
    }
    /* Let's re-enable the pending request in the same position */
    MPI_Start(&array_of_requests[cb->storage1]);
    return 0;
}

static void
remote_dep_mpi_put_start(parsec_execution_stream_t* es,
                         dep_cmd_item_t* item)
{
    remote_dep_wire_get_t* task = &(item->cmd.activate.task);
#if !defined(PARSEC_PROF_DRY_DEP)
    parsec_remote_deps_t* deps = (parsec_remote_deps_t*) (uintptr_t) task->deps;
    int k, nbdtt, tag = task->tag;
    parsec_comm_callback_t* cb;
    void* dataptr;
    MPI_Datatype dtt;
#endif  /* !defined(PARSEC_PROF_DRY_DEP) */
#if defined(PARSEC_DEBUG_NOISIER)
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif

    (void)es;
    DEBUG_MARK_CTL_MSG_GET_RECV(item->cmd.activate.peer, (void*)task, task);

#if !defined(PARSEC_PROF_DRY_DEP)
    assert(task->output_mask);
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tPUT mask=%lx deps 0x%lx", task->output_mask, task->deps);

    for(k = 0; task->output_mask>>k; k++) {
        assert(k < MAX_PARAM_COUNT);
        if(!((1U<<k) & task->output_mask)) continue;

        if(parsec_comm_puts == parsec_comm_puts_max) {
            PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\treach PUT limit for deps 0x%lx. Reschedule.", deps);
            parsec_list_nolock_push_front(&dep_put_fifo, (parsec_list_item_t*)item);
            return;
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\t[idx %d mask(0x%x / 0x%x)] %p, %p", k, (1U<<k), task->output_mask,
                deps->output[k].data.data, PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data));
        dataptr = PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data);
        dtt     = deps->output[k].data.layout;
        nbdtt   = deps->output[k].data.count;
#if defined(PARSEC_DEBUG_NOISIER)
        MPI_Type_get_name(dtt, type_name, &len);
        PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tTO\t%d\tPut START\tunknown \tk=%d\twith deps 0x%lx at %p type %s\t(tag=%d displ = %ld)",
               item->cmd.activate.peer, k, task->deps, dataptr, type_name, tag+k, deps->output[k].data.displ);
#endif

        TAKE_TIME_WITH_INFO(MPIsnd_prof, MPI_Data_plds_sk, k,
                            es->virtual_process->parsec_context->my_rank,
                            item->cmd.activate.peer, deps->msg);
        task->output_mask ^= (1U<<k);
        MPI_Isend((char*)dataptr + deps->output[k].data.displ, nbdtt, dtt,
                  item->cmd.activate.peer, tag + k, dep_comm,
                  &array_of_requests[parsec_comm_last_active_req]);
        cb = &array_of_callbacks[parsec_comm_last_active_req];
        cb->fct      = remote_dep_mpi_put_end_cb;
        cb->storage1 = (long)deps;
        cb->storage2 = k;
        parsec_comm_last_active_req++;
        parsec_comm_puts++;
        assert(parsec_comm_last_active_req <= DEP_NB_REQ);
        DEBUG_MARK_DTA_MSG_START_SEND(item->cmd.activate.peer, dataptr, tag+k);
    }
#endif  /* !defined(PARSEC_PROF_DRY_DEP) */
    if(0 == task->output_mask) {
        PARSEC_DEBUG_VERBOSE(100, parsec_comm_output_stream, "PUT_START output_maks completed for item %p, freeing", item);
        free(item);
    }
}

static int
remote_dep_mpi_put_end_cb(parsec_execution_stream_t* es,
                          parsec_comm_callback_t* cb,
                          MPI_Status* status)
{
    parsec_remote_deps_t* deps = (parsec_remote_deps_t*)cb->storage1;

    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tTO\tna\tPut END  \tunknown \tk=%d\twith deps %p\tparams %lx\t(tag=%d) data ptr %p",
            cb->storage2, deps, cb->storage2, status->MPI_TAG,
            deps->output[cb->storage2].data.data); (void)status;
    DEBUG_MARK_DTA_MSG_END_SEND(status->MPI_TAG);
    TAKE_TIME(MPIsnd_prof, MPI_Data_plds_ek, cb->storage2);
    remote_dep_complete_and_cleanup(&deps, 1);
    parsec_comm_puts--;
    (void)es;
    return 0;
}

/**
 * An activation message has been received, and the remote_dep_wire_activate_t
 * part has already been extracted into the deps->msg. This function handles the
 * rest of the receiver logic, extract the possible short and control data from
 * the buffer, post all the eager protocol receives and all other local
 * cleanups.
 */
static void remote_dep_mpi_recv_activate(parsec_execution_stream_t* es,
                                         parsec_remote_deps_t* deps,
                                         char* packed_buffer,
                                         int length,
                                         int* position)
{
    remote_dep_datakey_t complete_mask = 0;
    int k, dsize, tag = (int)deps->msg.tag; (void)tag;
#if defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN);
#endif
#if RDEP_MSG_EAGER_LIMIT != 0
    remote_dep_datakey_t eager_which = remote_dep_mpi_eager_which(deps, deps->incoming_mask);
#if !defined(PARSEC_PROF_DRY_DEP)
    MPI_Request reqs[MAX_PARAM_COUNT];
    int nb_reqs = 0, flag;
#endif  /* !defined(PARSEC_PROF_DRY_DEP) */
#endif  /* RDEP_MSG_EAGER_LIMIT != 0 */

#if defined(PARSEC_DEBUG) || defined(PARSEC_DEBUG_NOISIER)
    parsec_debug_verbose(6, parsec_comm_output_stream, "MPI:\tFROM\t%d\tActivate\t% -8s\n"
          "\twith datakey %lx\tparams %lx length %d (pack buf %d/%d) prio %d",
           deps->from, tmp, deps->msg.deps, deps->incoming_mask,
           deps->msg.length, *position, length, deps->max_priority);
#endif
    for(k = 0; deps->incoming_mask>>k; k++) {
        if(!(deps->incoming_mask & (1U<<k))) continue;
        /* Check for CTL and data that do not carry payload */
        if((NULL == deps->output[k].data.arena) || (0 == deps->output[k].data.count)) {
            PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tHERE\t%d\tGet NONE\t% -8s\tk=%d\twith datakey %lx at <NA> type CONTROL",
                    deps->from, tmp, k, deps->msg.deps);
            deps->output[k].data.data = NULL;
            complete_mask |= (1U<<k);
            continue;
        }

        if( parsec_param_short_limit && (length > *position) ) {
            /* Check if the data is short-embedded in the activate */
            MPI_Pack_size(deps->output[k].data.count, deps->output[k].data.layout,
                          dep_comm, &dsize);
            if((length - (*position)) >= dsize) {
                assert(NULL == deps->output[k].data.data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
                if(NULL == deps->output[k].data.data) {
                    deps->output[k].data.data = remote_dep_copy_allocate(&deps->output[k].data);
                }
#ifndef PARSEC_PROF_DRY_DEP
                PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, " EGR\t%s\tparam %d\tshort from the activate msg (%d/%d)",
                        tmp, k, dsize, length - *position);
                MPI_Unpack(packed_buffer, length, position,
                           (char*)PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data) + deps->output[k].data.displ,
                           deps->output[k].data.count, deps->output[k].data.layout, dep_comm);
#endif
                complete_mask |= (1U<<k);
                continue;
            }
        }
#if RDEP_MSG_EAGER_LIMIT != 0
       /* Check if we have EAGER deps to satisfy quickly */
        if( eager_which & (1U<<k) ) {

            assert(NULL == deps->output[k].data.data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
            if(NULL == deps->output[k].data.data) {
                deps->output[k].data.data = remote_dep_copy_allocate(&deps->output[k].data);
            }
            PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tFROM\t%d\tGet EAGER\t% -8s\tk=%d\twith datakey %lx at %p\t(tag=%d)",
                    deps->from, tmp, k, deps->msg.deps, deps->output[k].data.data, tag+k);
#ifndef PARSEC_PROF_DRY_DEP
            MPI_Irecv((char*)PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data) + deps->output[k].data.displ,
                      deps->output[k].data.count, deps->output[k].data.layout,
                      deps->from, tag + k, dep_comm, &reqs[nb_reqs]);
            nb_reqs++;
            MPI_Testall(nb_reqs, reqs, &flag, MPI_STATUSES_IGNORE);  /* a little progress */
#endif
            complete_mask |= (1U<<k);
            continue;
        }
#endif
        PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tFROM\t%d\tGet DATA\t% -8s\tk=%d\twith datakey %lx tag=%d (to be posted)",
                deps->from, tmp, k, deps->msg.deps, tag+k);
    }
#if (RDEP_MSG_EAGER_LIMIT != 0) && !defined(PARSEC_PROF_DRY_DEP)
    if (nb_reqs) {
#if 0
        dep_cmd_item_t* item;
        /* don't recursively call remote_dep_mpi_progress(es); but we still
         * need to advance the eager PUTs and activates */
        do {
            item = parsec_dequeue_try_pop_front(&dep_cmd_queue);
            if( NULL != item ) {
                if( DEP_PUT_DATA == item->action ) {
                    remote_dep_mpi_put_eager(es, item);
                }
                else {
                    parsec_dequeue_push_front(&dep_cmd_queue, item);
                }
            }
            MPI_Testall(nb_reqs, reqs, &flag, MPI_STATUSES_IGNORE);
        } while(!flag);
#else
        MPI_Waitall(nb_reqs, reqs, MPI_STATUSES_IGNORE);
#endif
    }
#endif  /* (RDEP_MSG_EAGER_LIMIT != 0) && !defined(PARSEC_PROF_DRY_DEP) */
    assert(length == *position);

    /* Release all the already satisfied deps without posting the RDV */
    if(complete_mask) {
#if defined(PARSEC_DEBUG_NOISIER)
        for(int k = 0; complete_mask>>k; k++)
            if((1U<<k) & complete_mask)
                PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tHERE\t%d\tGet PREEND\t% -8s\tk=%d\twith datakey %lx at %p ALREADY SATISFIED\t(tag=%d)",
                        deps->from, tmp, k, deps->msg.deps, deps->output[k].data.data, tag+k );
#endif
        /* If this is the only call then force the remote deps propagation */
        deps = remote_dep_release_incoming(es, deps, complete_mask);
    }

    /* Store the request in the rdv queue if any unsatisfied dep exist at this point */
    if(NULL != deps) {
        assert(0 != deps->incoming_mask);
        assert(0 != deps->msg.output_mask);
        parsec_list_nolock_push_sorted(&dep_activates_fifo, (parsec_list_item_t*)deps, rdep_prio);
    }

    /* Check if we have any pending GET orders */
    if((parsec_comm_gets < parsec_comm_gets_max) && !parsec_list_nolock_is_empty(&dep_activates_fifo)) {
        deps = (parsec_remote_deps_t*)parsec_list_nolock_pop_front(&dep_activates_fifo);
        remote_dep_mpi_get_start(es, deps);
    }
}

static int
remote_dep_mpi_save_activate_cb(parsec_execution_stream_t* es,
                                parsec_comm_callback_t* cb,
                                MPI_Status* status)
{
    PARSEC_PINS(es, ACTIVATE_CB_BEGIN, NULL);
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    int position = 0, length, rc;
    parsec_remote_deps_t* deps = NULL;

    MPI_Get_count(status, MPI_PACKED, &length);
    while(position < length) {
        deps = remote_deps_allocate(&parsec_remote_dep_context.freelist);
        MPI_Unpack(dep_activate_buff[cb->storage2], length, &position,
                   &deps->msg, dep_count, dep_dtt, dep_comm);
        deps->from = status->MPI_SOURCE;

        /* Retrieve the data arenas and update the msg.incoming_mask to reflect
         * the data we should be receiving from the predecessor.
         */
        rc = remote_dep_get_datatypes(es, deps, cb->storage2, &position);

        if( -1 == rc ) {
            /* the corresponding tp doesn't exist, yet. Put it in unexpected */
            char* packed_buffer;
            PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tFROM\t%d\tActivate NoTPool\t% -8s\tk=%d\twith datakey %lx\tparams %lx",
                    deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                    cb->storage2, deps->msg.deps, deps->msg.output_mask);
            /* Copy the short data to some temp storage */
            packed_buffer = malloc(deps->msg.length);
            memcpy(packed_buffer, dep_activate_buff[cb->storage2] + position, deps->msg.length);
            position += deps->msg.length;  /* move to the next order */
            deps->taskpool = (parsec_taskpool_t*)packed_buffer;  /* temporary storage */
            parsec_list_nolock_push_back(&dep_activates_noobj_fifo, (parsec_list_item_t*)deps);
            continue;
        } else {
            assert(deps->taskpool != NULL);
            if( -2 == rc ) { /* DTD problems, defer activating this remote dep */
                assert(deps->incoming_mask != deps->msg.output_mask);
                continue;
            }
        }

        PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tFROM\t%d\tActivate\t% -8s\tk=%d\twith datakey %lx\tparams %lx",
               status->MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
               cb->storage2, deps->msg.deps, deps->msg.output_mask);
        /* Import the activation message and prepare for the reception */
        remote_dep_mpi_recv_activate(es, deps, dep_activate_buff[cb->storage2],
                                     position + deps->msg.length, &position);
        assert( parsec_param_enable_aggregate || (position == length));
    }
    assert(position == length);
    /* Let's re-enable the pending request in the same position */
    MPI_Start(&array_of_requests[cb->storage1]);
    PARSEC_PINS(es, ACTIVATE_CB_END, NULL);
    return 0;
}

static void remote_dep_mpi_new_taskpool( parsec_execution_stream_t* es,
                                         dep_cmd_item_t *dep_cmd_item )
{
    parsec_list_item_t *item;
    parsec_taskpool_t* obj = dep_cmd_item->cmd.new_taskpool.tp;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    for(item = PARSEC_LIST_ITERATOR_FIRST(&dep_activates_noobj_fifo);
        item != PARSEC_LIST_ITERATOR_END(&dep_activates_noobj_fifo);
        item = PARSEC_LIST_ITERATOR_NEXT(item) ) {
        parsec_remote_deps_t* deps = (parsec_remote_deps_t*)item;
        if( deps->msg.taskpool_id == obj->taskpool_id ) {
            char* buffer = (char*)deps->taskpool;  /* get back the buffer from the "temporary" storage */
            int rc, position = 0;
            deps->taskpool = NULL;
            rc = remote_dep_get_datatypes(es, deps, PARSEC_DTD_SKIP_SAVING, &position); assert( -1 != rc );
            assert(deps->taskpool != NULL);
            PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tFROM\t%d\tActivate NEWOBJ\t% -8s\twith datakey %lx\tparams %lx",
                    deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                    deps->msg.deps, deps->msg.output_mask);

            item = parsec_list_nolock_remove(&dep_activates_noobj_fifo, item);

            /* In case of DTD execution, receiving rank might not have discovered
             * the task responsible for this message. So we have to put this message
             * in a hash table so that we can activate it, when this rank discovers it.
             */
            if( -2 == rc ) { /* DTD problems, defer activating this remote dep */
                deps->taskpool = (parsec_taskpool_t*) buffer;
                continue;
            }


            remote_dep_mpi_recv_activate(es, deps, buffer, deps->msg.length, &position);
            free(buffer);
            (void)rc;
        }
    }
    remote_dep_dec_flying_messages(obj);
}

/* In DTD runs, remote nodes might ask us to activate tasks that has not been
 * discovered in the local node yet. We delay activation of those tasks and
 * push the dep in a hash table. As soon as we discover the remote task, for
 * which an activation is already pending, we issue a command to activate that
 * dep, This function does the necessary steps to continue the activation of
 * the remote task.
 */
static void
remote_dep_mpi_release_delayed_deps( parsec_execution_stream_t* es,
                                     dep_cmd_item_t *item )
{
    PARSEC_PINS(es, ACTIVATE_CB_BEGIN, NULL);
    parsec_remote_deps_t *deps = item->cmd.release.deps;
    int rc, position = 0;
    char* buffer = (char*)deps->taskpool;  /* get back the buffer from the "temporary" storage */
    deps->taskpool = NULL;

    rc = remote_dep_get_datatypes(es, deps, PARSEC_DTD_SKIP_SAVING, &position);

    assert(rc != -2);
    (void)rc;

    assert(deps != NULL);
    remote_dep_mpi_recv_activate(es, deps, buffer, deps->msg.length, &position);
    free(buffer);
    PARSEC_PINS(es, ACTIVATE_CB_END, NULL);
}

static void remote_dep_mpi_get_start(parsec_execution_stream_t* es,
                                     parsec_remote_deps_t* deps)
{
    remote_dep_wire_activate_t* task = &(deps->msg);
    int from = deps->from, k, count, nbdtt;
    remote_dep_wire_get_t msg;
    MPI_Datatype dtt;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN], type_name[MPI_MAX_OBJECT_NAME];
    int len;
    remote_dep_cmd_to_string(task, tmp, MAX_TASK_STRLEN);
#endif

    for(k = count = 0; deps->incoming_mask >> k; k++)
        if( ((1U<<k) & deps->incoming_mask) ) count++;
    if( (parsec_comm_gets + count) > parsec_comm_gets_max ) {
        assert(deps->msg.output_mask != 0);
        assert(deps->incoming_mask != 0);
        parsec_list_nolock_push_front(&dep_activates_fifo, (parsec_list_item_t*)deps);
        return;
    }
    (void)es;
    DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, (void*)task, task);

    msg.output_mask = deps->incoming_mask;  /* Only get what I need */
    msg.deps        = task->deps;
    msg.tag         = task->tag;

    for(k = 0; deps->incoming_mask >> k; k++) {
        if( !((1U<<k) & deps->incoming_mask) ) continue;

        /* prepare the local receiving data */
        assert(NULL == deps->output[k].data.data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
        if(NULL == deps->output[k].data.data) {
            deps->output[k].data.data = remote_dep_copy_allocate(&deps->output[k].data);
        }
#ifdef PARSEC_PROF_DRY_DEP
        (void)dtt; (void)nbdtt; (void)msg; (void)from;
        /* Removing the corresponding bit prevent the sending of the GET_DATA request */
        remote_dep_mpi_get_end(es, k, deps);
        deps->incoming_mask ^= (1U<<k);
#else
        dtt   = deps->output[k].data.layout;
        nbdtt = deps->output[k].data.count;
#  if defined(PARSEC_DEBUG_NOISIER)
        MPI_Type_get_name(dtt, type_name, &len);
        PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tTO\t%d\tGet START\t% -8s\tk=%d\twith datakey %lx at %p type %s count %d displ %ld extent %d\t(tag=%d)",
                from, tmp, k, task->deps, PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data), type_name, nbdtt,
                deps->output[k].data.displ, deps->output[k].data.arena->elem_size * nbdtt, msg.tag+k);
#  endif
        TAKE_TIME_WITH_INFO(MPIrcv_prof, MPI_Data_pldr_sk, k, from,
                            es->virtual_process->parsec_context->my_rank, deps->msg);
        DEBUG_MARK_DTA_MSG_START_RECV(from, deps->output[k].data.data, msg.tag+k);
        MPI_Irecv((char*)PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data) + deps->output[k].data.displ, nbdtt,
                  dtt, from, msg.tag + k, dep_comm,
                  &array_of_requests[parsec_comm_last_active_req]);
        parsec_comm_callback_t* cb = &array_of_callbacks[parsec_comm_last_active_req];
        cb->fct      = remote_dep_mpi_get_end_cb;
        cb->storage1 = (long)deps;
        cb->storage2 = k;
        parsec_comm_last_active_req++;
        parsec_comm_gets++;
        assert(parsec_comm_last_active_req <= DEP_NB_REQ);
#endif
    }
#if !defined(PARSEC_PROF_DRY_DEP)
    if(msg.output_mask) {
        TAKE_TIME_WITH_INFO(MPIctl_prof, MPI_Data_ctl_sk, get,
                            from, es->virtual_process->parsec_context->my_rank, (*task));
        MPI_Send(&msg, datakey_count, datakey_dtt, from,
                 REMOTE_DEP_GET_DATA_TAG, dep_comm);
        TAKE_TIME(MPIctl_prof, MPI_Data_ctl_ek, get++);
        DEBUG_MARK_CTL_MSG_GET_SENT(from, (void*)&msg, &msg);
    }
#endif  /* !defined(PARSEC_PROF_DRY_DEP) */
}

static void remote_dep_mpi_get_end(parsec_execution_stream_t* es,
                                   int idx,
                                   parsec_remote_deps_t* deps)
{
    /* The ref on the data will be released below */
    remote_dep_release_incoming(es, deps, (1U<<idx));
}

static int
remote_dep_mpi_get_end_cb(parsec_execution_stream_t* es,
                          parsec_comm_callback_t* cb,
                          MPI_Status* status)
{
    parsec_remote_deps_t* deps = (parsec_remote_deps_t*)cb->storage1;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MPI:\tFROM\t%d\tGet END  \t% -8s\tk=%d\twith datakey na        \tparams %lx\t(tag=%d)",
            status->MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
            (int)cb->storage2, deps->incoming_mask, status->MPI_TAG); (void)status;
    DEBUG_MARK_DTA_MSG_END_RECV(status->MPI_TAG);
    TAKE_TIME(MPIrcv_prof, MPI_Data_pldr_ek, (int)cb->storage2);
    remote_dep_mpi_get_end(es, (int)cb->storage2, deps);
    parsec_comm_gets--;
    return 0;
}
