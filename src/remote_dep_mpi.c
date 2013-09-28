/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* /!\  THIS FILE IS NOT INTENDED TO BE COMPILED ON ITS OWN
 *      It should be included from remote_dep.c if HAVE_MPI is defined
 */
#include "dague_config.h"

#include <mpi.h>
#include "profiling.h"
#include "arena.h"
#include "list.h"

#define DAGUE_REMOTE_DEP_USE_THREADS

typedef union dep_cmd_u dep_cmd_t;

static int remote_dep_mpi_init(dague_context_t* context);
static int remote_dep_mpi_fini(dague_context_t* context);
static int remote_dep_mpi_on(dague_context_t* context);
static int remote_dep_mpi_send_dep(dague_execution_unit_t* eu_context, int rank, remote_dep_wire_activate_t* msg);
static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context);
static int remote_dep_get_datatypes(dague_remote_deps_t* origin);
static int remote_dep_release(dague_execution_unit_t* eu_context, dague_remote_deps_t* origin);

static int remote_dep_nothread_send(dague_execution_unit_t* eu_context, int rank, dague_remote_deps_t* deps);
static int remote_dep_nothread_memcpy(dep_cmd_t* cmd);

static int remote_dep_bind_thread(dague_context_t* context);

static int remote_dep_dequeue_send(int rank, dague_remote_deps_t* deps);
static int remote_dep_dequeue_new_object(dague_object_t* obj);
#ifdef DAGUE_REMOTE_DEP_USE_THREADS
static int remote_dep_dequeue_init(dague_context_t* context);
static int remote_dep_dequeue_fini(dague_context_t* context);
static int remote_dep_dequeue_on(dague_context_t* context);
static int remote_dep_dequeue_off(dague_context_t* context);
/*static int remote_dep_dequeue_progress(dague_execution_unit_t* eu_context);*/
#   define remote_dep_init(ctx) remote_dep_dequeue_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_dequeue_on(ctx)
#   define remote_dep_off(ctx)  remote_dep_dequeue_off(ctx)
#   define remote_dep_new_object(obj) remote_dep_dequeue_new_object(obj)
#   define remote_dep_send(rank, deps) remote_dep_dequeue_send(rank, deps)
#   define remote_dep_progress(ctx) ((void)ctx,0)

#else
static int remote_dep_dequeue_nothread_init(dague_context_t* context);
static int remote_dep_dequeue_nothread_fini(dague_context_t* context);
#   define remote_dep_init(ctx) remote_dep_dequeue_nothread_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_nothread_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_mpi_on(ctx)
#   define remote_dep_off(ctx)  0
#   define remote_dep_new_object(obj) remote_dep_dequeue_new_object(obj)
#   define remote_dep_send(rank, deps) remote_dep_dequeue_send(rank, deps)
#   define remote_dep_progress(ctx) remote_dep_dequeue_nothread_progress(ctx)
#endif
static int remote_dep_dequeue_nothread_progress(dague_execution_unit_t* eu_context);

#include "dequeue.h"

#define DEP_NB_CONCURENT 3
static int dague_mpi_activations = 1 * DEP_NB_CONCURENT;
static int dague_mpi_transfers  = 2 * DEP_NB_CONCURENT;

typedef enum dep_cmd_action_t
{
    DEP_ACTIVATE,
    DEP_RELEASE,
/*    DEP_PROGRESS,
    DEP_PUT_DATA,
    DEP_GET_DATA,*/
    DEP_NEW_OBJECT,
    DEP_CTL,
    DEP_MEMCPY,
} dep_cmd_action_t;

union dep_cmd_u
{
    struct {
        int rank;
        dague_remote_deps_t* deps;
    } activate;
    struct {
        dague_remote_deps_t* deps;
    } release;
    struct {
        int enable;
    } ctl;
    struct {
        dague_object_t* obj;
    } new_object;
    struct {
        dague_object_t      *dague_object;
        dague_arena_chunk_t *source;
        void                *destination;
        dague_datatype_t     datatype;
        int64_t              displ_s;
        int64_t              displ_r;
        int                  count;
    } memcpy;
};

typedef struct dep_cmd_item_t
{
    dague_list_item_t super;
    dep_cmd_action_t  action;
    int               priority;
    dep_cmd_t         cmd;
} dep_cmd_item_t;
#define dep_cmd_prio (offsetof(dep_cmd_item_t, priority))

typedef struct dague_dep_wire_get_fifo_elem_t {
    dague_list_item_t           item;
    remote_dep_wire_get_t       task;
    int                         priority;
    int                         peer;
} dague_dep_wire_get_fifo_elem_t;
#define dep_wire_get_prio (offsetof(dague_dep_wire_get_fifo_elem_t, priority))
#define rdep_prio (offsetof(dague_remote_deps_t, max_priority))

static void remote_dep_mpi_save_put( dague_execution_unit_t* eu_context, int i, MPI_Status* status );
static void remote_dep_mpi_put_start( dague_execution_unit_t* eu_context, dague_dep_wire_get_fifo_elem_t* item, int i );
static void remote_dep_mpi_put_end( dague_execution_unit_t* eu_context, int i, int k, MPI_Status* status );
static void remote_dep_mpi_put_short( dague_execution_unit_t* eu_context, remote_dep_wire_activate_t* msg, int rank );
static void remote_dep_mpi_save_activate( dague_execution_unit_t* eu_context, int i, MPI_Status* status );
static void remote_dep_mpi_get_start( dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i );
static void remote_dep_mpi_get_end( dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i, int k );
static void remote_dep_mpi_new_object( dague_execution_unit_t* eu_context, dague_object_t* obj );

#ifdef DAGUE_DEBUG_VERBOSE1
static char* remote_dep_cmd_to_string(remote_dep_wire_activate_t* origin, char* str, size_t len)
{
    unsigned int i, index = 0;
    dague_object_t* object;
    const dague_function_t* function;

    object = dague_object_lookup( origin->object_id );
    function = object->functions_array[origin->function_id];

    index += snprintf( str + index, len - index, "%s", function->name );
    if( index >= len ) return str;
    for( i = 0; i < function->nb_parameters; i++ ) {
        index += snprintf( str + index, len - index, "_%d",
                           origin->locals[function->params[i]->context_index].value );
        if( index >= len ) return str;
    }
    return str;
}
#endif

static pthread_t dep_thread_id;
dague_dequeue_t dep_cmd_queue;
dague_list_t    dep_cmd_fifo;            /* ordered non threaded fifo */
dague_list_t    dep_activates_fifo;      /* ordered non threaded fifo */
dague_list_t    dep_activates_noobj_fifo;/* non threaded fifo */
dague_list_t    dep_put_fifo;            /* ordered non threaded fifo */
dague_remote_deps_t** dep_pending_recv_array;
dague_dep_wire_get_fifo_elem_t** dep_pending_put_array;

static void *remote_dep_dequeue_main(dague_context_t* context);
static int mpi_initialized = 0;
#if defined(DAGUE_REMOTE_DEP_USE_THREADS)
static pthread_mutex_t mpi_thread_mutex;
static pthread_cond_t mpi_thread_condition;
#endif

static int remote_dep_dequeue_init(dague_context_t* context)
{
    pthread_attr_t thread_attr;
    int is_mpi_up = 0;

    assert(mpi_initialized == 0);

    MPI_Initialized(&is_mpi_up);
    if( 0 == is_mpi_up ) {
        /**
         * MPI is not up, so we will consider this as a single
         * node run. Fall back to the no-MPI case.
         */
        context->nb_nodes = 1;
        DEBUG3(("MPI is not initialized. Fall back to a single node execution\n"));
        return 1;
    }
    MPI_Comm_size(MPI_COMM_WORLD, (int*)&(context->nb_nodes));
    if(1 == context->nb_nodes ) return 1;

    /**
     * Finalize the initialization of the upper level structures
     * Worst case: one of the DAGs is going to use up to
     * MAX_PARAM_COUNT times nb_nodes dependencies.
     */
    remote_deps_allocation_init(context->nb_nodes, MAX_PARAM_COUNT);

    dague_dequeue_construct(&dep_cmd_queue);
    dague_list_construct(&dep_cmd_fifo);

    /* Build the condition used to drive the MPI thread */
    pthread_mutex_init( &mpi_thread_mutex, NULL );
    pthread_cond_init( &mpi_thread_condition, NULL );

    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);

   /**
    * We need to synchronize with the newly spawned thread. We will use
    * the condition for this. If we lock the mutex prior to spawning the
    * MPI thread, and then go in a condition wait, the MPI thread can
    * lock the mutex, and then call condition signal. This insure
    * proper synchronization. Similar mechanism will be used to turn
    * on and off the MPI thread.
    */
    pthread_mutex_lock(&mpi_thread_mutex);

    pthread_create(&dep_thread_id,
                   &thread_attr,
                   (void* (*)(void*))remote_dep_dequeue_main,
                   (void*)context);

    /* Wait until the MPI thread signals it's awakening */
    pthread_cond_wait( &mpi_thread_condition, &mpi_thread_mutex );
    mpi_initialized = 1;  /* up and running */

    return context->nb_nodes;
}

static int remote_dep_dequeue_fini(dague_context_t* context)
{
    if( (1 == context->nb_nodes) || (0 == mpi_initialized) ) return 0;

    /**
     * We suppose the off function was called before. Then
     * we will append a shutdown command in the MPI thread queue,
     * and wake the MPI thread. Upon processing of the pending
     * command the MPI thread will exit, we will be able to catch
     * this by locking the mutex.
     * Once we know the MPI thread is gone, cleaning up will be
     * straighforward.
     */
    {
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        DAGUE_LIST_ITEM_CONSTRUCT(item);
        void *ret;

        item->action = DEP_CTL;
        item->cmd.ctl.enable = -1;  /* turn off the MPI thread */
        item->priority = 0;
        dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);

        /* I am supposed to own the lock. Wake the MPI thread */
        pthread_cond_signal(&mpi_thread_condition);
        pthread_mutex_unlock(&mpi_thread_mutex);
        pthread_join(dep_thread_id, &ret);
        assert((dague_context_t*)ret == context);
    }

    dague_dequeue_destruct(&dep_cmd_queue);
    dague_list_destruct(&dep_cmd_fifo);

    return 0;
}

static volatile int mpi_thread_marker = 0;
static int remote_dep_dequeue_on(dague_context_t* context)
{
    if(1 == context->nb_nodes) return 0;
    /* At this point I am supposed to own the mutex */
    mpi_thread_marker = 1;
    pthread_cond_signal(&mpi_thread_condition);
    pthread_mutex_unlock(&mpi_thread_mutex);
    return 1;
}

static int remote_dep_dequeue_off(dague_context_t* context)
{
    if(1 == context->nb_nodes) return 0;

    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    DAGUE_LIST_ITEM_CONSTRUCT(item);
    item->action = DEP_CTL;
    item->cmd.ctl.enable = 0;  /* turn OFF the MPI thread */
    item->priority = 0;
    while( 1 == mpi_thread_marker ) sched_yield();
    dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);

    pthread_mutex_lock(&mpi_thread_mutex);
    return 0;
}

#define YIELD_TIME 5000
#include "bindthread.h"

static int do_nano = 0;

static void* remote_dep_dequeue_main(dague_context_t* context)
{
    int whatsup;

    remote_dep_bind_thread(context);

    remote_dep_mpi_init(context);
    /* Now synchroniza with the main thread */
    pthread_mutex_lock(&mpi_thread_mutex);
    pthread_cond_signal(&mpi_thread_condition);

    /* This is the main loop. Wait until being woken up by the main thread,
     * do the MPI stuff until we get the OFF or FINI commands. Then
     * react the them.
     */
    do {
        /* Now let's block */
        pthread_cond_wait(&mpi_thread_condition, &mpi_thread_mutex);
        /* acknoledge the activation */
        mpi_thread_marker = 0;
        /* The MPI thread is owning the lock */
        remote_dep_mpi_on(context);
        whatsup = remote_dep_dequeue_nothread_progress(context->virtual_processes[0]->execution_units[0]);
    } while(-1 != whatsup);
    /* Release all resources */
    remote_dep_mpi_fini(context);
    return (void*)context;
}

static int remote_dep_dequeue_new_object(dague_object_t* obj)
{
    if(!mpi_initialized) return 0;
    dep_cmd_item_t* item = (dep_cmd_item_t*)calloc(1, sizeof(dep_cmd_item_t));
    DAGUE_LIST_ITEM_CONSTRUCT(item);
    item->action = DEP_NEW_OBJECT;
    item->priority = 0;
    item->cmd.new_object.obj = obj;
    dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*)item);
    return 1;
}

static int remote_dep_dequeue_send(int rank, dague_remote_deps_t* deps)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    DAGUE_LIST_ITEM_CONSTRUCT(item);
    item->action = DEP_ACTIVATE;
    item->priority = deps->max_priority;
    item->cmd.activate.rank = rank;
    item->cmd.activate.deps = deps;
    dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*)item);
    return 1;
}

void dague_remote_dep_memcpy(dague_execution_unit_t* eu_context,
                             dague_object_t* dague_object,
                             void* dst,
                             dague_dep_data_description_t* data)
{
    assert( dst );
    dep_cmd_item_t* item = (dep_cmd_item_t*)calloc(1, sizeof(dep_cmd_item_t));
    DAGUE_LIST_ITEM_CONSTRUCT(item);
    item->action = DEP_MEMCPY;
    item->priority = 0;
    item->cmd.memcpy.dague_object = dague_object;
    item->cmd.memcpy.source       = (dague_arena_chunk_t*)data->ptr;
    item->cmd.memcpy.destination  = dst;
    item->cmd.memcpy.datatype     = data->layout;
    item->cmd.memcpy.displ_s      = data->displ;
    item->cmd.memcpy.displ_r      = 0;
    item->cmd.memcpy.count        = data->count;
    AREF(item->cmd.memcpy.source);
    remote_dep_inc_flying_messages(dague_object, eu_context->virtual_process->dague_context);
    dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);
}

/**
 * Retrieve the datatypes involved in this communication. In addition
 * the flag DAGUE_ACTION_RECV_INIT_REMOTE_DEPS set the
 * origin->max_priority to the maximum priority of all the children.
 */
static int remote_dep_get_datatypes(dague_remote_deps_t* origin)
{
    dague_execution_context_t exec_context;

    exec_context.dague_object = dague_object_lookup( origin->msg.object_id );
    if( NULL == exec_context.dague_object )
        return -1; /* the dague object doesn't exist yet */
    assert( NULL == origin->dague_object );
    origin->dague_object = exec_context.dague_object;
    exec_context.function = exec_context.dague_object->functions_array[origin->msg.function_id];

    for(int i = 0; i < exec_context.function->nb_locals; i++)
        exec_context.locals[i] = origin->msg.locals[i];

    return exec_context.function->release_deps(NULL, &exec_context,
                                               DAGUE_ACTION_RECV_INIT_REMOTE_DEPS | origin->msg.which,
                                               origin);
}

static int remote_dep_release(dague_execution_unit_t* eu_context, dague_remote_deps_t* origin)
{
    int actions = DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_RELEASE_REMOTE_DEPS;
    dague_execution_context_t exec_context;
    const dague_flow_t* target;
    int ret, i, whereto;

    exec_context.dague_object = dague_object_lookup( origin->msg.object_id );
#if defined(DAGUE_DEBUG)
    exec_context.priority = 0;
#endif
    assert(exec_context.dague_object); /* Future: for composition, store this in a list to be considered upon creation of the DO*/
    exec_context.function = exec_context.dague_object->functions_array[origin->msg.function_id];
    for( i = 0; i < exec_context.function->nb_locals; i++)
        exec_context.locals[i] = origin->msg.locals[i];

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != (target = exec_context.function->out[i])); i++) {
        whereto = target->flow_index;
        exec_context.data[whereto].data_repo = NULL;
        exec_context.data[whereto].data      = NULL;
        if(origin->msg.deps & (1 << i)) {
            DEBUG3(("MPI:\tDATA %p released from %p[%d]\n", ADATA(origin->output[i].data), origin, i));
            exec_context.data[whereto].data = origin->output[i].data.ptr;
#if defined(DAGUE_DEBUG) && defined(DAGUE_DEBUG_VERBOSE3)
            if(origin->output[i].data.arena) { /* no prints for CTL! */
                char tmp[MAX_TASK_STRLEN];
                void* _data = ADATA(exec_context.data[whereto].data);
                DEBUG3(("MPI:\t%s: recv %p -> [0] %9.5f [1] %9.5f [2] %9.5f\n",
                       dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &exec_context),
                       _data, ((double*)_data)[0], ((double*)_data)[1], ((double*)_data)[2]));
            }
#endif
        }
    }
    ret = exec_context.function->release_deps(eu_context, &exec_context,
                                              actions |
                                              origin->msg.deps,
                                              origin);
    origin->msg.which ^= origin->msg.deps;
    origin->msg.deps = 0;
    return ret;
}

#ifndef DAGUE_REMOTE_DEP_USE_THREADS
static int remote_dep_dequeue_nothread_init(dague_context_t* context)
{
    dague_dequeue_construct(&dep_cmd_queue);
    dague_list_construct(&dep_cmd_fifo);
    return remote_dep_mpi_init(context);
}

static int remote_dep_dequeue_nothread_fini(dague_context_t* context)
{
    remote_dep_mpi_fini(context);
    dague_list_destruct(&dep_cmd_fifo);
    return 0;
}
#endif

static int remote_dep_dequeue_nothread_progress(dague_execution_unit_t* eu_context)
{
    dep_cmd_item_t* item;
    int ret = 0;

 check_pending_queues:
    /**
     * Move as many elements as possible from the dequeue into our ordered lifo.
     */
    while( NULL != (item = (dep_cmd_item_t*) dague_dequeue_try_pop_front(&dep_cmd_queue)) ) {
        if( DEP_CTL == item->action ) {
            /* A DEP_CTL is a barrier that must not be crossed, flush the
             * ordered fifo and don't add anything until it is consumed */
            if( !dague_ulist_is_empty(&dep_cmd_fifo) ) {
                dague_dequeue_push_front(&dep_cmd_queue, (dague_list_item_t*)item);
                break;
            } else goto handle_now;
        }

        dague_ulist_push_sorted(&dep_cmd_fifo, (dague_list_item_t*)item, dep_cmd_prio);
    }
    item = (dep_cmd_item_t*)dague_ulist_fifo_pop(&dep_cmd_fifo);

    if(NULL == item ) {
        do {
            ret = remote_dep_mpi_progress(eu_context);
        } while(ret);

        if(do_nano && !ret) {
            struct timespec ts;
            ts.tv_sec = 0; ts.tv_nsec = YIELD_TIME;
            nanosleep(&ts, NULL);
        }
        goto check_pending_queues;
    }
handle_now:
    switch(item->action) {
    case DEP_CTL:
        ret = item->cmd.ctl.enable;
        free(item);
        return ret;  /* FINI or OFF */
    case DEP_NEW_OBJECT:
        remote_dep_mpi_new_object(eu_context, item->cmd.new_object.obj);
        break;
    case DEP_ACTIVATE:
        remote_dep_nothread_send(eu_context, item->cmd.activate.rank, item->cmd.activate.deps);
        break;
    case DEP_MEMCPY:
        remote_dep_nothread_memcpy(&item->cmd);
        remote_dep_dec_flying_messages(item->cmd.memcpy.dague_object, eu_context->virtual_process->dague_context);
        break;
    default:
        assert(0 && item->action); /* Not a valid action */
        break;
    }
    free(item);
    goto check_pending_queues;
}


static int remote_dep_nothread_send( dague_execution_unit_t* eu_context,
                                     int rank,
                                     dague_remote_deps_t* deps)
{
    int k;
    int rank_bank = rank / (sizeof(uint32_t) * 8);
    uint32_t rank_mask = 1 << (rank % (sizeof(uint32_t) * 8));
    int output_count = deps->output_count;
    remote_dep_wire_activate_t msg = deps->msg;

    msg.deps = (uintptr_t)deps;
    for( k = 0; output_count; k++ ) {
        output_count -= deps->output[k].count_bits;
        if(deps->output[k].rank_bits[rank_bank] & rank_mask) {
#if defined(DAGUE_PROF_DRY_DEP)
            deps->output[k].data.arena = NULL; /* make all data a control */
#endif
            msg.which |= (1<<k);
        }
    }
    remote_dep_mpi_send_dep(eu_context, rank, &msg);
    return 0;
}

static int remote_dep_nothread_memcpy(dep_cmd_t* cmd)
{

    /* TODO: split the mpi part */
    int rc = MPI_Sendrecv(ADATA(cmd->memcpy.source) + cmd->memcpy.displ_s, cmd->memcpy.count, cmd->memcpy.datatype, 0, 0,
                          cmd->memcpy.destination + cmd->memcpy.displ_r, cmd->memcpy.count, cmd->memcpy.datatype, 0, 0,
                          MPI_COMM_SELF, MPI_STATUS_IGNORE);
    AUNREF(cmd->memcpy.source);
    return (MPI_SUCCESS == rc ? 0 : -1);
}





/******************************************************************************
 * ALL MPI SPECIFIC CODE GOES HERE
 ******************************************************************************/
enum {
    REMOTE_DEP_ACTIVATE_TAG,
    REMOTE_DEP_GET_DATA_TAG,
    REMOTE_DEP_PUT_DATA_TAG,
    REMOTE_DEP_MAX_CTRL_TAG
} dague_remote_dep_tag_t;

#ifdef DAGUE_PROF_TRACE
static dague_thread_profiling_t* MPIctl_prof;
static dague_thread_profiling_t* MPIsnd_prof[DEP_NB_CONCURENT];
static dague_thread_profiling_t* MPIrcv_prof[DEP_NB_CONCURENT];
static unsigned long act = 0;
static int MPI_Activate_sk, MPI_Activate_ek;
static unsigned long get = 0;
static int MPI_Data_ctl_sk, MPI_Data_ctl_ek;
static int MPI_Data_plds_sk, MPI_Data_plds_ek;
static int MPI_Data_pldr_sk, MPI_Data_pldr_ek;

typedef struct {
    int rank_src;
    int rank_dst;
    char func[16];
} dague_profile_remote_dep_mpi_info_t;

static char dague_profile_remote_dep_mpi_info_to_string[] = "";

static void remote_dep_mpi_profiling_init(void)
{
    int i;

    dague_profiling_add_dictionary_keyword( "MPI_ACTIVATE", "fill:#FF0000",
                                            sizeof(dague_profile_remote_dep_mpi_info_t),
                                            dague_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Activate_sk, &MPI_Activate_ek);
    dague_profiling_add_dictionary_keyword( "MPI_DATA_CTL", "fill:#000077",
                                            sizeof(dague_profile_remote_dep_mpi_info_t),
                                            dague_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Data_ctl_sk, &MPI_Data_ctl_ek);
    dague_profiling_add_dictionary_keyword( "MPI_DATA_PLD_SND", "fill:#B08080",
                                            sizeof(dague_profile_remote_dep_mpi_info_t),
                                            dague_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Data_plds_sk, &MPI_Data_plds_ek);
    dague_profiling_add_dictionary_keyword( "MPI_DATA_PLD_RCV", "fill:#80B080",
                                            sizeof(dague_profile_remote_dep_mpi_info_t),
                                            dague_profile_remote_dep_mpi_info_to_string,
                                            &MPI_Data_pldr_sk, &MPI_Data_pldr_ek);

    MPIctl_prof = dague_profiling_thread_init( 2*1024*1024, "MPI ctl");
    for(i = 0; i < DEP_NB_CONCURENT; i++) {
        MPIsnd_prof[i] = dague_profiling_thread_init( 2*1024*1024, "MPI isend(req=%d)", i);
        MPIrcv_prof[i] = dague_profiling_thread_init( 2*1024*1024, "MPI irecv(req=%d)", i);
    }
}

#define TAKE_TIME_WITH_INFO(PROF, KEY, I, src, dst, ctx) do {           \
        dague_profile_remote_dep_mpi_info_t __info;                     \
        dague_execution_context_t __exec_context;                       \
        dague_object_t *__object = dague_object_lookup( ctx.object_id ); \
        __exec_context.function = __object->functions_array[ ctx.function_id ]; \
        __exec_context.dague_object = __object;                         \
        memcpy(&__exec_context.locals, ctx.locals, MAX_LOCAL_COUNT * sizeof(assignment_t)); \
        dague_snprintf_execution_context(__info.func, 16, &__exec_context);      \
        __info.rank_src = src;                                          \
        __info.rank_dst = dst;                                          \
        dague_profiling_trace((PROF), (KEY), (I), PROFILE_OBJECT_ID_NULL, &__info); \
    } while(0)

#define TAKE_TIME(PROF, KEY, I) dague_profiling_trace((PROF), (KEY), (I), PROFILE_OBJECT_ID_NULL, NULL);
#else
#define TAKE_TIME_WITH_INFO(PROF, KEY, I, src, dst, ctx) do {} while(0)
#define TAKE_TIME(PROF, KEY, I) do {} while(0)
#define remote_dep_mpi_profiling_init() do {} while(0)
#endif  /* DAGUE_PROF_TRACE */

#if defined(DAGUE_STATS)

#   define DAGUE_STATACC_ACCUMULATE_MSG(counter, count, datatype) do {\
        int _sa_size; \
        MPI_Pack_size(count, datatype, dep_comm, &_sa_size); \
        DAGUE_STATACC_ACCUMULATE(counter, 1); \
        DAGUE_STATACC_ACCUMULATE(counter_bytes_sent, _sa_size); \
    }
#else
#   define DAGUE_STATACC_ACCUMULATE_MSG(counter, count, datatype)
#endif /* DAGUE_STATS */


/* TODO: smart use of dague context instead of ugly globals */
static MPI_Comm dep_comm;
#define DEP_NB_REQ (2 * DEP_NB_CONCURENT + 2 * (DEP_NB_CONCURENT * MAX_PARAM_COUNT))
static MPI_Request  array_of_requests[DEP_NB_REQ];
static int          array_of_indices[DEP_NB_REQ];
static MPI_Status   array_of_statuses[DEP_NB_REQ];
static MPI_Request* dep_activate_req    = &array_of_requests[0 * DEP_NB_CONCURENT];
static MPI_Request* dep_get_req         = &array_of_requests[1 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_snd_req     = &array_of_requests[2 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_rcv_req     = &array_of_requests[2 * DEP_NB_CONCURENT + DEP_NB_CONCURENT * MAX_PARAM_COUNT];

/* TODO: fix heterogeneous restriction by using proper mpi datatypes */
#define dep_dtt MPI_BYTE
#define dep_count sizeof(remote_dep_wire_activate_t)
#define dep_extent dep_count
#define DEP_EAGER_BUFFER_SIZE (dep_extent+RDEP_MSG_EAGER_LIMIT)
static char dep_activate_buff[DEP_NB_CONCURENT][DEP_EAGER_BUFFER_SIZE];
//static dague_remote_deps_t* dep_activate_buff[DEP_NB_CONCURENT];
#define datakey_dtt MPI_LONG
#define datakey_count 3
static remote_dep_wire_get_t dep_get_buff[DEP_NB_CONCURENT];

/* Pointers are converted to long to be used as keys to fetch data in the get
 * rdv protocol. Make sure we can carry pointers correctly.
 */
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif
#if ULONG_MAX < UINTPTR_MAX
#error "unsigned long is not large enough to hold a pointer!"
#endif
/* note: tags are necessary, because multiple activate requests are not
 * fifo, relative to one another, during the waitsome loop */
static int MAX_MPI_TAG;
#define MIN_MPI_TAG (REMOTE_DEP_MAX_CTRL_TAG+1)
static int VAL_NEXT_TAG = MIN_MPI_TAG;
static inline int next_tag(int k) {
    int tag = VAL_NEXT_TAG;
    if( MAX_MPI_TAG < tag+k )
        VAL_NEXT_TAG = MIN_MPI_TAG;
    else
        VAL_NEXT_TAG += k;
    return tag;
}

static int remote_dep_mpi_init(dague_context_t* context)
{
    int i, mpi_tag_ub_exists, *ub;

    dague_list_construct(&dep_activates_fifo);
    dague_list_construct(&dep_activates_noobj_fifo);
    dague_list_construct(&dep_put_fifo);

    MPI_Comm_dup(MPI_COMM_WORLD, &dep_comm);
    /*
     * Based on MPI 1.1 the MPI_TAG_UB should only be defined
     * on MPI_COMM_WORLD.
     */
#if defined(HAVE_MPI_20)
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &mpi_tag_ub_exists);
#else
    MPI_Attr_get(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &mpi_tag_ub_exists);
#endif  /* defined(HAVE_MPI_20) */
    if( !mpi_tag_ub_exists ) {
        MAX_MPI_TAG = INT_MAX;
        WARNING(("Your MPI implementation does not define MPI_TAG_UB and thus violates the standard (MPI-2.2, page 29, line 30); Lets assume any integer value is a valid MPI Tag.\n"));
    } else {
        MAX_MPI_TAG = *ub;
#if defined( DAGUE_DEBUG_VERBOSE1 )
        if( MAX_MPI_TAG < INT_MAX ) {
            WARNING(("MPI:\tYour MPI implementation defines the maximal TAG value to %d (0x%08x), which might be too small should you have more than %d simultaneous remote dependencies\n",
                    MAX_MPI_TAG, (unsigned int)MAX_MPI_TAG, MAX_MPI_TAG / MAX_PARAM_COUNT));
        }
#endif
    }

    MPI_Comm_size(dep_comm, &(context->nb_nodes));
    MPI_Comm_rank(dep_comm, &(context->my_rank));
    for(i = 0; i < DEP_NB_REQ; i++) {
        array_of_requests[i] = MPI_REQUEST_NULL;
    }

    /* Create all the pending receives and start them */
    for(i = 0; i < DEP_NB_CONCURENT; i++) {
        MPI_Recv_init(&dep_activate_buff[i], DEP_EAGER_BUFFER_SIZE, MPI_PACKED, MPI_ANY_SOURCE, REMOTE_DEP_ACTIVATE_TAG, dep_comm, &dep_activate_req[i]);
        MPI_Recv_init(&dep_get_buff[i], datakey_count, datakey_dtt, MPI_ANY_SOURCE, REMOTE_DEP_GET_DATA_TAG, dep_comm, &dep_get_req[i]);
        MPI_Start(&dep_activate_req[i]);
        MPI_Start(&dep_get_req[i]);
    }

    dep_pending_recv_array = (dague_remote_deps_t**)calloc(DEP_NB_CONCURENT, sizeof(dague_remote_deps_t*));
    dep_pending_put_array = (dague_dep_wire_get_fifo_elem_t**)calloc(DEP_NB_CONCURENT, sizeof(dague_dep_wire_get_fifo_elem_t*));
    remote_dep_mpi_profiling_init();
    return 0;
}

static int remote_dep_mpi_fini(dague_context_t* context)
{
    int i, flag;
    MPI_Status status;

    for(i = 0; i < DEP_NB_CONCURENT; i++) {
        MPI_Cancel(&dep_activate_req[i]); MPI_Test(&dep_activate_req[i], &flag, &status); MPI_Request_free(&dep_activate_req[i]);
        MPI_Cancel(&dep_get_req[i]); MPI_Test(&dep_get_req[i], &flag, &status); MPI_Request_free(&dep_get_req[i]);
    }
    for(i = 0; i < DEP_NB_REQ; i++) {
        assert(MPI_REQUEST_NULL == array_of_requests[i]);
    }
    free( dep_pending_put_array );
    free( dep_pending_recv_array );
    dague_list_destruct(&dep_activates_fifo);
    dague_list_destruct(&dep_activates_noobj_fifo);
    dague_list_destruct(&dep_put_fifo);
    MPI_Comm_free(&dep_comm);
    (void)context;
    return 0;
}

static int remote_dep_mpi_on(dague_context_t* context)
{
#ifdef DAGUE_PROF_TRACE
    int i;
    /* put a start marker on each line */
    TAKE_TIME(MPIctl_prof, MPI_Activate_sk, 0);
    for(i = 0; i < DEP_NB_CONCURENT; i++) {
        TAKE_TIME(MPIsnd_prof[i], MPI_Activate_sk, 0);
        TAKE_TIME(MPIrcv_prof[i], MPI_Activate_sk, 0);
    }
    MPI_Barrier(dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, 0);
    for(i = 0; i < DEP_NB_CONCURENT; i++) {
        TAKE_TIME(MPIsnd_prof[i], MPI_Activate_ek, 0);
        TAKE_TIME(MPIrcv_prof[i], MPI_Activate_ek, 0);
    }
#endif
    (void)context;
    return 0;
}

/* Send the activate tag */
static int remote_dep_mpi_send_dep(dague_execution_unit_t* eu_context, int rank, remote_dep_wire_activate_t* msg)
{
    dague_remote_deps_t* deps = (dague_remote_deps_t*) msg->deps;
    dague_object_t* obj = deps->dague_object;
#ifdef DAGUE_DEBUG_VERBOSE1
    char tmp[MAX_TASK_STRLEN];
#endif
#if !defined(DAGUE_PROF_TRACE)
    (void)eu_context;
#endif
    msg->tag = next_tag(MAX_PARAM_COUNT); /* todo: waste less tags to diminish collision probability */
    DEBUG(("MPI:\tTO\t%d\tActivate\t% -8s\ti=na\twith datakey %lx\tmask %lx\t(tag=%d)\n", rank, remote_dep_cmd_to_string(msg, tmp, MAX_TASK_STRLEN), msg->deps, msg->which, msg->tag));

    /* Treat for special cases: CTL, Eeager, etc... */
    char packed_buffer[DEP_EAGER_BUFFER_SIZE];
    int packed = 0;
    MPI_Pack(msg, dep_count, dep_dtt, packed_buffer, DEP_EAGER_BUFFER_SIZE, &packed, dep_comm);
    for(int k=0; msg->which>>k; k++) {
        if(0 == (msg->which & (1<<k))) continue;

        /* Remove CTL from the message we expect to send */
        if(NULL == deps->output[k].data.arena) {
            DEBUG2((" CTL\t%s\tparam %d\tdemoted to be a control\n",remote_dep_cmd_to_string(&deps->msg, tmp, 128), k));
            msg->which ^= (1<<k);
            remote_dep_complete_and_cleanup(deps, 1);
            continue;
        }
        assert(deps->output[k].data.count > 0);
        /* Embed as many Eager as possible with the activate msg */
        int dsize;
        MPI_Pack_size(deps->output[k].data.count, deps->output[k].data.layout, dep_comm, &dsize);
        if((DEP_EAGER_BUFFER_SIZE - packed) > (size_t)dsize) {
            DEBUG2((" EGR\t%s\tparam %d\teager piggyback in the activate message\n",remote_dep_cmd_to_string(&deps->msg, tmp, 128), k));
            msg->which ^= (1<<k);
            MPI_Pack(ADATA(deps->output[k].data.ptr) + deps->output[k].data.displ,
                     deps->output[k].data.count, deps->output[k].data.layout,
                     packed_buffer, DEP_EAGER_BUFFER_SIZE, &packed, dep_comm);
            remote_dep_complete_and_cleanup(deps, 1);
        }
    }

    TAKE_TIME_WITH_INFO(MPIctl_prof, MPI_Activate_sk, act, eu_context->virtual_process->dague_context->my_rank, rank, (*msg));
    DAGUE_STATACC_ACCUMULATE_MSG(counter_control_messages_sent, packed, MPI_PACKED);
    MPI_Send((void*)packed_buffer, packed, MPI_PACKED, rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, act++);
    DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(rank, (void*)msg, msg);

    if(0 == msg->which) {
        remote_dep_dec_flying_messages(obj, eu_context->virtual_process->dague_context);
    } else {
        remote_dep_mpi_put_short(eu_context, msg, rank);
    }
    return 1;
}


static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context)
{
#ifdef DAGUE_DEBUG_VERBOSE2
    char tmp[MAX_TASK_STRLEN];
#endif
    MPI_Status *status;
    int ret = 0, index, i, k, outcount;

    if( !DAGUE_THREAD_IS_MASTER(eu_context) ) return 0;

    do {
        MPI_Testsome(DEP_NB_REQ, array_of_requests, &outcount, array_of_indices, array_of_statuses);
        if(0 == outcount) break;  /* nothing ready right now */
        for( index = 0; index < outcount; index++ ) {
            i = array_of_indices[index];
            status = &(array_of_statuses[index]);

            if(i < dague_mpi_activations) {
                assert(REMOTE_DEP_ACTIVATE_TAG == status->MPI_TAG);
                remote_dep_mpi_save_activate( eu_context, i, status );
                MPI_Start(&dep_activate_req[i]);
            } else if(i < dague_mpi_transfers) {
                i -= dague_mpi_activations; /* shift i */
                assert(REMOTE_DEP_GET_DATA_TAG == status->MPI_TAG);
                remote_dep_mpi_save_put( eu_context, i, status );
                MPI_Start(&dep_get_req[i]);
            } else {
                i -= dague_mpi_transfers;  /* shift i */
                assert(i >= 0);
                if(i < (DEP_NB_CONCURENT * MAX_PARAM_COUNT)) {
                    /* We finished sending the data, allow for more requests 
                     * to be processed */
                    k = i % MAX_PARAM_COUNT;
                    i = i / MAX_PARAM_COUNT;
                    remote_dep_mpi_put_end(eu_context, i, k, status);
                } else {
                    /* We received a data, call the matching release_dep */
                    dague_remote_deps_t* deps;
                    i -= (DEP_NB_CONCURENT * MAX_PARAM_COUNT);
                    assert(i >= 0);
                    k = i%MAX_PARAM_COUNT;
                    i = i/MAX_PARAM_COUNT;
                    deps = (dague_remote_deps_t*) dep_pending_recv_array[i];
                    DEBUG2(("MPI:\tFROM\t%d\tGet END  \t% -8s\ti=%d,k=%d\twith datakey na        \tparams %lx\t(tag=%d)\n",
                            status->MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), i, k, deps->msg.which, status->MPI_TAG));
                    DEBUG_MARK_DTA_MSG_END_RECV(status->MPI_TAG);
                    TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_ek, i+k);
                    remote_dep_mpi_get_end(eu_context, deps, i, k);
                    ret++;
                }
            }
        }
    } while(1);
    return ret;
}


static remote_dep_datakey_t remote_dep_mpi_short_which(remote_dep_wire_activate_t* msg)
{
#ifdef DAGUE_DEBUG_VERBOSE3
        char tmp[MAX_TASK_STRLEN];
#endif
    dague_remote_deps_t* deps = (dague_remote_deps_t*)msg->deps;
    remote_dep_datakey_t short_which = 0;
    for(int k = 0; msg->which>>k; k++) {
        assert(k < MAX_PARAM_COUNT);
        if( !(msg->which & (1<<k)) ) continue;
        if( NULL == deps->output[k].data.arena ) continue;
        size_t extent = deps->output[k].data.arena->elem_size * deps->output[k].data.count;
        if( extent <= (RDEP_MSG_EAGER_LIMIT) ) {
            short_which |= 1<<k;
            DEBUG3(("MPI:\tPEER\tNA\tEager MODE  \t% -8s\tk=%d\tsize=%d <= %d\t(tag=%d)\n",
                    remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), k, extent,
                    RDEP_MSG_EAGER_LIMIT, msg->tag+k));
            continue;
        }
        if( extent <= (RDEP_MSG_SHORT_LIMIT) ) {
            short_which |= 1<<k;
            DEBUG3(("MPI:\tPEER\tNA\tShort MODE  \t% -8s\tk=%d\tsize=%d <= %d\t(tag=%d)\n",
                    remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), k, extent,
                    RDEP_MSG_SHORT_LIMIT, msg->tag+k));
            continue;
        }
    }
    return short_which;
}




static void remote_dep_mpi_put_short( dague_execution_unit_t* eu_context, remote_dep_wire_activate_t* msg, int rank )
{
#ifdef DAGUE_DEBUG_VERBOSE3
    char tmp[MAX_TASK_STRLEN];
#endif
    remote_dep_datakey_t short_which = remote_dep_mpi_short_which(msg);
    if( short_which ) {
        dague_remote_deps_t* deps = (dague_remote_deps_t*)msg->deps;
        dague_dep_wire_get_fifo_elem_t* wireget;
        wireget = (dague_dep_wire_get_fifo_elem_t*)malloc(sizeof(dague_dep_wire_get_fifo_elem_t));
        DAGUE_LIST_ITEM_CONSTRUCT(wireget);
        wireget->priority = deps->max_priority;
        wireget->peer = rank;
        wireget->task.deps = msg->deps;
        wireget->task.which = short_which;
        wireget->task.tag = msg->tag;
        /* Check if we can process it right now */
        for(int i = 0; i < DEP_NB_CONCURENT; i++ ) {
            if( NULL == dep_pending_put_array[i] ) {
                remote_dep_mpi_put_start(eu_context, wireget, i);
                return;
            }
        }
        DEBUG3(("MPI: Put Short DELAYED for %s from %d tag %u which 0x%x (deps %p)\n",
                remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), wireget->peer, msg->tag, short_which, (void*)deps));

        dague_ulist_push_front(&dep_put_fifo, (dague_list_item_t*)wireget);
#if 0
        /* we can't process it now, push it first in queue, and
         * progress rdv to make room */
        dague_list_item_t* item = (dague_list_item_t*)wireget;
        while( (dague_list_item_t*)wireget == item ) {
            dague_ulist_push_front(&dep_put_fifo, item);
            remote_dep_mpi_progress(eu_context);
            item = dague_ulist_pop_front(&dep_put_fifo);
        }
        if( NULL != item ) { /* return the item to the list */
            dague_ulist_push_front(&dep_put_fifo, item);
        }
#endif
    }
}

static void remote_dep_mpi_save_put( dague_execution_unit_t* eu_context, int i, MPI_Status* status )
{
#ifdef DAGUE_DEBUG_VERBOSE3
    char tmp[MAX_TASK_STRLEN];
#endif
    dague_dep_wire_get_fifo_elem_t* item;
    remote_dep_wire_get_t* task;
    dague_remote_deps_t *deps;

    item = (dague_dep_wire_get_fifo_elem_t*)malloc(sizeof(dague_dep_wire_get_fifo_elem_t));
    DAGUE_LIST_ITEM_CONSTRUCT(item);
    task = &(item->task);
    memcpy( task, &dep_get_buff[i], sizeof(remote_dep_wire_get_t) );
    deps = (dague_remote_deps_t*) (uintptr_t) task->deps;
    item-> priority = deps->max_priority;
    item->peer = status->MPI_SOURCE;
    dague_ulist_push_sorted(&dep_put_fifo, (dague_list_item_t*)item, dep_wire_get_prio);
    /* Check if we can push any new puts */
    for( i = 0; i < DEP_NB_CONCURENT; i++ ) {
        if( NULL == dep_pending_put_array[i] ) {
            item = (dague_dep_wire_get_fifo_elem_t*)dague_ulist_fifo_pop(&dep_put_fifo);
            remote_dep_mpi_put_start(eu_context, item, i );
            return;
        }
    }
    DEBUG3(("MPI: Put DELAYED for %s from %d tag %u which 0x%x (deps %p)\n",
       remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), item->peer, task->tag, task->which, (void*)deps));
}

static void remote_dep_mpi_put_start(dague_execution_unit_t* eu_context, dague_dep_wire_get_fifo_elem_t* item, int i)
{
    remote_dep_wire_get_t* task = &(item->task);
    dague_remote_deps_t* deps = (dague_remote_deps_t*) (uintptr_t) task->deps;
    int tag = task->tag;
    void* data;
    MPI_Datatype dtt;
    int nbdtt;
#ifdef DAGUE_DEBUG_VERBOSE2
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif

    (void)eu_context;
    DEBUG_MARK_CTL_MSG_GET_RECV(item->peer, (void*)task, task);

    assert(task->which);
    DEBUG3(("MPI:\tPUT which=%lx\n", task->which));
    for(int k = 0; task->which>>k; k++) {
        assert(k < MAX_PARAM_COUNT);
        if(!((1<<k) & task->which)) continue;
        DEBUG3(("MPI:\t%p[%d] %p, %p\n", deps, k, deps->output[k].data.ptr,
                ADATA(deps->output[k].data.ptr)));
        data = ADATA(deps->output[k].data.ptr);
        dtt = deps->output[k].data.layout;
        nbdtt = deps->output[k].data.count;
#ifdef DAGUE_DEBUG_VERBOSE2
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG2(("MPI:\tTO\t%d\tPut START\tunknown \tj=%d,k=%d\twith datakey %lx at %p type %s\t(tag=%d displ = %ld)\n",
               item->peer, i, k, task->deps, data, type_name, tag+k, deps->output[k].displ));
#endif

        TAKE_TIME_WITH_INFO(MPIsnd_prof[i], MPI_Data_plds_sk, i,
                            eu_context->virtual_process->dague_context->my_rank, item->peer, deps->msg);
        MPI_Isend(data + deps->output[k].data.displ,
                  nbdtt, dtt, item->peer, tag + k, dep_comm, &dep_put_snd_req[i*MAX_PARAM_COUNT+k]);
        DEBUG_MARK_DTA_MSG_START_SEND(item->peer, data, tag+k);
    }
    dep_pending_put_array[i] = item;
}

static void
remote_dep_mpi_put_end(dague_execution_unit_t* eu_context,
                       int i, int k,
                       MPI_Status* status)
{
    dague_dep_wire_get_fifo_elem_t* item = dep_pending_put_array[i];
    assert(NULL != item);
    remote_dep_wire_get_t* task = &(item->task);
    dague_remote_deps_t* deps = (dague_remote_deps_t*)(uintptr_t)task->deps;

    DEBUG2(("MPI:\tTO\tna\tPut END  \tunknown \tj=%d,k=%d\twith datakey %lx\tparams %lx\t(tag=%d)\n",
           i, k, deps, task->which, status->MPI_TAG)); (void)status;
    DEBUG_MARK_DTA_MSG_END_SEND(status->MPI_TAG);
    AUNREF(deps->output[k].data.ptr);
    TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_ek, i);
    task->which ^= (1<<k);
    /* Are we done yet ? */

    if( 0 == task->which ) {
        remote_dep_dec_flying_messages(deps->dague_object,
                                       eu_context->virtual_process->dague_context);
    }
    remote_dep_complete_and_cleanup(deps, 1);
    if( 0 == task->which ) {
        free(item);
        dep_pending_put_array[i] = NULL;
        item = (dague_dep_wire_get_fifo_elem_t*)dague_ulist_fifo_pop(&dep_put_fifo);
        if( NULL != item ) {
            remote_dep_mpi_put_start(eu_context, item, i );
        }
    }
}

static void remote_dep_mpi_recv_activate( dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, char* packed_buffer, int unpacked )
{
#ifdef DAGUE_DEBUG_VERBOSE2
    char tmp[MAX_TASK_STRLEN];
#endif
    int tag = (int)deps->msg.tag;
    remote_dep_datakey_t datakey = deps->msg.deps;
    deps->msg.deps = (remote_dep_datakey_t)deps;
    remote_dep_datakey_t short_which = remote_dep_mpi_short_which(&deps->msg);
    deps->msg.deps = 0; /* now, it contains the mask of deps presatisfied */

    for(int k = 0; deps->msg.which>>k; k++) {
        if(!(deps->msg.which & (1<<k))) continue;
        /* Check for all CTL messages, that do not carry payload */
        if(NULL == deps->output[k].data.arena) {
            DEBUG2(("MPI:\tHERE\t%d\tGet NONE\t% -8s\ti=NA,k=%d\twith datakey %lx at <NA> type CONTROL extent 0\t(tag=%d)\n", deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), k, datakey, tag+k));
            deps->output[k].data.ptr = (void*)2; /* the first non zero even value */
            deps->msg.deps |= 1<<k;
            continue;
        }

        /* Check if the data is EAGER embedded in the activate */
        int dsize;
        MPI_Pack_size(deps->output[k].data.count, deps->output[k].data.layout, dep_comm, &dsize);
        if((DEP_EAGER_BUFFER_SIZE - unpacked) > (size_t)dsize) {
            assert(NULL == deps->output[k].data.ptr); /* we do not support in-place tiles now, make sure it doesn't happen yet */
            if(NULL == deps->output[k].data.ptr) {
                deps->output[k].data.ptr = dague_arena_get(deps->output[k].data.arena, deps->output[k].data.count);
                DEBUG3(("MPI:\tMalloc new remote tile %p size %zu count = %d displ = %ld\n",
                        deps->output[k].data, deps->output[k].data.arena->elem_size,
                        deps->output[k].count, deps->output[k].data.displ));
                assert(deps->output[k].data.ptr != NULL);
            }
#ifndef DAGUE_PROF_DRY_DEP
            DEBUG2((" EGR\t%s\tparam %d\teager piggyback from the activate message\n",remote_dep_cmd_to_string(&deps->msg, tmp, 128), k));
            MPI_Unpack(packed_buffer, DEP_EAGER_BUFFER_SIZE, &unpacked,
                       ADATA(deps->output[k].data.ptr) + deps->output[k].data.displ,
                       deps->output[k].data.count, deps->output[k].data.layout, dep_comm);
#endif
            deps->msg.deps |= 1<<k;
            continue;
        }

        /* Check if we have SHORT deps to satisfy quickly */
        if( short_which & (1<<k) ) {

            assert(NULL == deps->output[k].data.ptr); /* we do not support in-place tiles now, make sure it doesn't happen yet */
            if(NULL == deps->output[k].data.ptr) {
                deps->output[k].data.ptr = dague_arena_get(deps->output[k].data.arena, deps->output[k].data.count);
                DEBUG3(("MPI:\tMalloc new remote tile %p size %zu count = %d\n",
                        deps->output[k].data.ptr, deps->output[k].data.arena->elem_size, deps->output[k].data.count));
                assert(deps->output[k].data.ptr != NULL);
            }
            DEBUG2(("MPI:\tFROM\t%d\tGet SHORT\t% -8s\ti=NA,k=%d\twith datakey %lx at %p\t(tag=%d)\n",
                   deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), k, deps->msg.deps, ADATA(deps->output[k].data), tag+k));
#ifndef DAGUE_PROF_DRY_DEP
            MPI_Request req; int flag = 0;
            MPI_Irecv(ADATA(deps->output[k].data.ptr) + deps->output[k].data.displ,
                      deps->output[k].data.count, deps->output[k].data.layout,
                      deps->from, tag+k, dep_comm, &req);
            do {
                MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
                if(flag) break;
                remote_dep_mpi_progress(eu_context);
            } while(!flag);
#endif
            deps->msg.deps |= 1<<k;
            continue;
        }
    }

    /* Release all the already satisfied deps without posting the RDV */
    if(deps->msg.deps) {
#ifdef DAGUE_DEBUG_VERBOSE2
        for(int k = 0; deps->msg.deps>>k; k++)
            if((1<<k) & deps->msg.deps)
                DEBUG2(("MPI:\tHERE\t%d\tGet PREEND\t% -8s\ti=NA,k=%d\twith datakey %lx at %p ALREADY SATISFIED\t(tag=%d)\n",
                       deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), k, datakey, ADATA(deps->output[k].data.ptr), tag+k ));
#endif
        remote_dep_release(eu_context, deps);
    }

    /* Store the request in the rdv queue if any unsatisfied dep exist at this
     * point */
    if(deps->msg.which) {
        deps->msg.deps = datakey;
        dague_ulist_push_sorted(&dep_activates_fifo, (dague_list_item_t*)deps, rdep_prio);
    }
    else
    {
        dague_lifo_push(&dague_remote_dep_context.freelist, (dague_list_item_t*)deps);
    }

    /* Check if we have some ordered rdv get to treat */
    for(int i = 0; i < DEP_NB_CONCURENT; i++ ) {
        if( NULL == dep_pending_recv_array[i] ) {
            deps = (dague_remote_deps_t*)dague_ulist_fifo_pop(&dep_activates_fifo);
            if(deps) remote_dep_mpi_get_start(eu_context, deps, i );
            break;
        }
    }
}

static void remote_dep_mpi_save_activate( dague_execution_unit_t* eu_context, int i, MPI_Status* status )
{
#ifdef DAGUE_DEBUG_VERBOSE1
    char tmp[MAX_TASK_STRLEN];
#endif
    int unpacked = 0;
    dague_remote_deps_t* deps = remote_deps_allocate(&dague_remote_dep_context.freelist);
    MPI_Unpack(dep_activate_buff[i], DEP_EAGER_BUFFER_SIZE, &unpacked,
               &deps->msg, dep_count, dep_dtt, dep_comm);
    deps->from = status->MPI_SOURCE;
    DEBUG(("MPI:\tFROM\t%d\tActivate\t% -8s\ti=%d\twith datakey %lx\tparams %lx\n",
           status->MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
           i, deps->msg.deps, deps->msg.which));

    if( -1 == remote_dep_get_datatypes(deps) )
    {   /* the corresponding dague_object doesn't exist, yet. Put it in unexpected */
        char* packed_buffer;
        DEBUG2(("MPI:\tFROM\t%d\tActivate NOOBJ\t% -8s\ti=%d\twith datakey %lx\tparams %lx\n",
                       deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                       i, deps->msg.deps, deps->msg.which));
        /* Copy the eager data to some temp storage; TODO: use count instead of max size */
        packed_buffer = malloc(sizeof(dague_remote_deps_t)+DEP_EAGER_BUFFER_SIZE);
        memcpy(packed_buffer, deps, sizeof(dague_remote_deps_t));
        memcpy(packed_buffer+sizeof(dague_remote_deps_t), dep_activate_buff[i], DEP_EAGER_BUFFER_SIZE);
        dague_ulist_fifo_push(&dep_activates_noobj_fifo, (dague_list_item_t*)packed_buffer);
        remote_deps_free(deps);
        return;
    }
    /* Retrieve the data arenas and update the msg.which to reflect all the data
     * we should be receiving from the father. If some of the dependencies have
     * been dropped, force their release.
     */
    remote_dep_mpi_recv_activate(eu_context, deps, dep_activate_buff[i], unpacked);
}

static void remote_dep_mpi_new_object( dague_execution_unit_t* eu_context, dague_object_t* obj )
{
#if defined(DAGUE_DEBUG_VERBOSE2)
    char tmp[MAX_TASK_STRLEN];
#endif
    DAGUE_ULIST_ITERATOR(&dep_activates_noobj_fifo, item,
    ({
        dague_remote_deps_t* ideps = (dague_remote_deps_t*)item;
        if( ideps->msg.object_id == obj->object_id ) {
            char* buffer = sizeof(dague_remote_deps_t) + (char*)item;
            int rc, unpacked = 0;
            dague_remote_deps_t* deps = remote_deps_allocate(&dague_remote_dep_context.freelist);
            MPI_Unpack(buffer, DEP_EAGER_BUFFER_SIZE, &unpacked, 
                       &deps->msg, dep_count, dep_dtt, dep_comm);
            deps->from = ideps->from;
            rc = remote_dep_get_datatypes(deps); assert( -1 != rc );
            DEBUG2(("MPI:\tFROM\t%d\tActivate NEWOBJ\t% -8s\ti=NA\twith datakey %lx\tparams %lx\n",
                    deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                    deps->msg.deps, deps->msg.which));
            remote_dep_mpi_recv_activate(eu_context, deps, buffer, unpacked);
            item = dague_ulist_remove(&dep_activates_noobj_fifo, item);
            free(item);
            (void)rc;
        }
    }));
}

static void remote_dep_mpi_get_start(dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i)
{
#ifdef DAGUE_DEBUG_VERBOSE2
    char tmp[MAX_TASK_STRLEN], type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif
    MPI_Datatype dtt;
    int nbdtt;
    remote_dep_wire_get_t msg;
    remote_dep_wire_activate_t* task = &(deps->msg);
    int from = deps->from;
    void* data;

    (void)eu_context;
    DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, (void*)task, task);

    msg.which = task->which;
    msg.deps  = task->deps;
    msg.tag   = task->tag;

    for(int k = 0; msg.which >> k; k++) {
        if( !((1<<k) & msg.which) ) continue;
        dtt   = deps->output[k].data.layout;
        nbdtt = deps->output[k].data.count;
        data  = deps->output[k].data.ptr;
        assert(NULL == data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
        if(NULL == data) {
            data = dague_arena_get(deps->output[k].data.arena, deps->output[k].data.count);
            DEBUG3(("MPI:\tMalloc new remote tile %p size %zu\n", data,
                    deps->output[k].data.arena->elem_size * deps->output[k].data.count));
            assert(data != NULL);
            deps->output[k].data.ptr = data;
        }
#ifdef DAGUE_PROF_DRY_DEP
        (void)dtt;
        (void)nbdtt;
        (void)dep_put_rcv_req;
        msg.which &= ~(1<<k);
        remote_dep_mpi_get_end(eu_context, deps, i, k);
#else
#  ifdef DAGUE_DEBUG_VERBOSE2
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG2(("MPI:\tTO\t%d\tGet START\t% -8s\ti=%d,k=%d\twith datakey %lx at %p type %s count %d displ %ld extent %d\t(tag=%d)\n",
                from, remote_dep_cmd_to_string(task, tmp, MAX_TASK_STRLEN), i, k, task->deps, ADATA(data),
                type_name, nbdtt, deps->output[k].data.displ, deps->output[k].data.arena->elem_size * nbdtt, msg.tag+k));
#  endif
        TAKE_TIME_WITH_INFO(MPIrcv_prof[i], MPI_Data_pldr_sk, i+k, from,
                            eu_context->virtual_process->dague_context->my_rank, deps->msg);
        DEBUG_MARK_DTA_MSG_START_RECV(from, data, msg.tag+k);
        MPI_Irecv(ADATA(data) + deps->output[k].data.displ, nbdtt,
                  dtt, from, msg.tag+k, dep_comm,
                  &dep_put_rcv_req[i*MAX_PARAM_COUNT+k]);
#endif
    }
    if(msg.which)
    {
        TAKE_TIME_WITH_INFO(MPIctl_prof, MPI_Data_ctl_sk, get,
                            from, eu_context->virtual_process->dague_context->my_rank, (*task));
        DAGUE_STATACC_ACCUMULATE_MSG(counter_control_messages_sent, datakey_count, datakey_dtt);
        MPI_Send(&msg, datakey_count, datakey_dtt, from,
                 REMOTE_DEP_GET_DATA_TAG, dep_comm);
        assert(NULL == dep_pending_recv_array[i]);
        dep_pending_recv_array[i] = deps;
        TAKE_TIME(MPIctl_prof, MPI_Data_ctl_ek, get++);
        DEBUG_MARK_CTL_MSG_GET_SENT(from, (void*)&msg, &msg);
    }

    deps->msg.deps = 0; /* now this is the mask of finished deps */
}

static void remote_dep_mpi_get_end(dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i, int k)
{
    deps->msg.deps = 1<<k;
    remote_dep_release(eu_context, deps);
    AUNREF(deps->output[k].data.ptr);
    if(deps->msg.which == deps->msg.deps) {
        dague_lifo_push(&dague_remote_dep_context.freelist, (dague_list_item_t*)deps);
        dep_pending_recv_array[i] = NULL;
        if( !dague_ulist_is_empty(&dep_activates_fifo) ) {
            deps = (dague_remote_deps_t*)dague_ulist_fifo_pop(&dep_activates_fifo);
            if( NULL != deps ) {
                remote_dep_mpi_get_start(eu_context, deps, i );
            }
        }
    }
}


/* Bind the communication thread on an unused core if possible */
int remote_dep_bind_thread(dague_context_t* context){

    do_nano=1;

#if defined(HAVE_HWLOC) && defined(HAVE_HWLOC_BITMAP)
    char *str = NULL;
    if( context->comm_th_core >= 0 ) {
        /* Bind to the specified core */
        if(dague_bindthread(context->comm_th_core, -1) == context->comm_th_core) {
            STATUS(("Communication thread bound to physical core %d\n",  context->comm_th_core));

            /* Check if this core is not used by a computation thread */
            if( hwloc_bitmap_isset(context->index_core_free_mask, context->comm_th_core) )
                do_nano = 0;
        } else {
            /* There is no guarantee the thread doesn't share the core. Let do_nano to 1. */
            WARNING(("Request to bind the communication thread on core %d failed.\n", context->comm_th_core));
        }
    } else if( context->comm_th_core == -2 ) {
        /* Bind to the specified mask */
        hwloc_cpuset_t free_common_cores;

        /* reduce the mask to unused cores if any */
        free_common_cores=hwloc_bitmap_alloc();
        hwloc_bitmap_and(free_common_cores, context->index_core_free_mask, context->comm_th_index_mask);

        if( !hwloc_bitmap_iszero(free_common_cores) ) {
            hwloc_bitmap_copy(context->comm_th_index_mask, free_common_cores);

            do_nano = 0;
        }
        hwloc_bitmap_asprintf(&str, context->comm_th_index_mask);
        hwloc_bitmap_free(free_common_cores);
        if( dague_bindthread_mask(context->comm_th_index_mask) >= 0 ) {
            DEBUG(("Communication thread bound on the index mask %s\n", str));
        } else {
            WARNING(("Communication thread requested to be bound on the cpu mask %s \n", str));
            do_nano = 1;
        }
    } else {
        /* no binding specified
         * - bind on available cores if any,
         * - let float otherwise
         */

        if( !hwloc_bitmap_iszero(context->index_core_free_mask) ) {
            if( dague_bindthread_mask(context->index_core_free_mask) > -1 ){
                hwloc_bitmap_asprintf(&str, context->index_core_free_mask);
                DEBUG(("Communication thread bound on the cpu mask %s\n", str));
                free(str);
                do_nano = 0;
            }
        }
    }
#else /* NO HAVE_HWLOC */
    /* If we don't have hwloc, try to bind the thread on the core #nbcore as the
     * default strategy disributed the computation threads from core 0 to nbcore-1 */
    int nb_total_comp_threads = 0;
    int p;
    for(p = 0; p < context->nb_vp; p++) {
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;
    }
    int boundto = dague_bindthread(nb_total_comp_threads, -1);
    if (boundto != nb_total_comp_threads) {
        DEBUG(("Communication thread floats\n"));
    } else {
        do_nano = 0;
        DEBUG(("Communication thread bound to physical core %d\n", boundto));
    }
#endif /* NO HAVE_HWLOC */
    return 0;
}
