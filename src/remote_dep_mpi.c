/*
 * Copyright (c) 2009-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* /!\  THIS FILE IS NOT INTENDED TO BE COMPILED ON ITS OWN
 *      It should be included from remote_dep.c if HAVE_MPI is defined
 */
#include <dague_config.h>

#include <mpi.h>
#include "profiling.h"
#include "list.h"
#include "data.h"

#define DAGUE_REMOTE_DEP_USE_THREADS

typedef struct dep_cmd_item_s dep_cmd_item_t;
typedef union dep_cmd_u dep_cmd_t;

static int remote_dep_mpi_init(dague_context_t* context);
static int remote_dep_mpi_fini(dague_context_t* context);
static int remote_dep_mpi_on(dague_context_t* context);
static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context);
static int remote_dep_get_datatypes(dague_execution_unit_t* eu_context, dague_remote_deps_t* origin);
static dague_remote_deps_t*
remote_dep_release_incoming(dague_execution_unit_t* eu_context,
                            dague_remote_deps_t* origin,
                            remote_dep_datakey_t complete_mask);

static int remote_dep_nothread_send(dague_execution_unit_t* eu_context,
                                    dep_cmd_item_t **head_item);
static int remote_dep_nothread_memcpy(dague_execution_unit_t* eu_context,
                                      dep_cmd_item_t *item);

static int remote_dep_dequeue_send(int rank, dague_remote_deps_t* deps);
static int remote_dep_dequeue_new_object(dague_handle_t* obj);
#ifdef DAGUE_REMOTE_DEP_USE_THREADS
static int remote_dep_dequeue_init(dague_context_t* context);
static int remote_dep_dequeue_fini(dague_context_t* context);
static int remote_dep_dequeue_on(dague_context_t* context);
static int remote_dep_dequeue_off(dague_context_t* context);
/*static int remote_dep_dequeue_progress(dague_context_t* context);*/
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
static int remote_dep_dequeue_nothread_progress(dague_context_t* context);

#include "dequeue.h"

/**
 * Number of data movements to be extracted at each step. Bigger the number
 * larger the amount spent in ordering the tasks, but greater the potential
 * benefits of doing things in the right order.
 */
static int dague_param_nb_tasks_extracted = 20;
static int dague_param_enable_eager = DAGUE_DIST_EAGER_LIMIT;
static int dague_param_enable_aggregate = 1;

#define DEP_NB_CONCURENT 3
static int DEP_NB_REQ;

static int dague_comm_activations_max = 2*DEP_NB_CONCURENT;
static int dague_comm_data_get_max    = 2*DEP_NB_CONCURENT;
static int dague_comm_gets_max        = DEP_NB_CONCURENT * MAX_PARAM_COUNT;
static int dague_comm_gets            = 0;
static int dague_comm_puts_max        = DEP_NB_CONCURENT * MAX_PARAM_COUNT;
static int dague_comm_puts            = 0;
static int dague_comm_last_active_req = 0;

/**
 * The order is important as it will be used to compute the index in the
 * pending array of messages.
 */
typedef enum dep_cmd_action_t {
    DEP_ACTIVATE    = -1,
    DEP_NEW_OBJECT  =  0,
    DEP_MEMCPY,
    DEP_RELEASE,
/*    DEP_PROGRESS,
    DEP_PUT_DATA,
    DEP_GET_DATA,*/
    DEP_CTL,
    DEP_LAST  /* always the last element. it shoud not be used */
} dep_cmd_action_t;

union dep_cmd_u {
    struct {
        remote_dep_wire_get_t task;
        int                   peer;
    } activate;
    struct {
        dague_remote_deps_t  *deps;
    } release;
    struct {
        int enable;
    } ctl;
    struct {
        dague_handle_t       *obj;
    } new_object;
    struct {
        dague_handle_t       *dague_handle;
        dague_data_copy_t    *source;
        dague_data_copy_t    *destination;
        dague_datatype_t      datatype;
        int64_t               displ_s;
        int64_t               displ_r;
        int                   count;
    } memcpy;
};

struct dep_cmd_item_s {
    dague_list_item_t super;
    dague_list_item_t pos_list;
    dep_cmd_action_t  action;
    int               priority;
    dep_cmd_t         cmd;
};
#define dep_cmd_prio (offsetof(dep_cmd_item_t, priority))
#define dep_mpi_pos_list (offsetof(dep_cmd_item_t, priority) - offsetof(dep_cmd_item_t, pos_list))
#define rdep_prio (offsetof(dague_remote_deps_t, max_priority))

typedef struct dague_comm_callback_s dague_comm_callback_t;

static int
remote_dep_mpi_save_put_cb(dague_execution_unit_t* eu_context,
                           dague_comm_callback_t* cb, MPI_Status* status);
static void remote_dep_mpi_put_start(dague_execution_unit_t* eu_context, dep_cmd_item_t* item);
static int remote_dep_mpi_put_end_cb(dague_execution_unit_t* eu_context,
                                     dague_comm_callback_t* cb, MPI_Status* status);
#if 0 != RDEP_MSG_SHORT_LIMIT
static void remote_dep_mpi_put_short( dague_execution_unit_t* eu_context,
                                      dep_cmd_item_t* item);
#endif  /* 0 != RDEP_MSG_SHORT_LIMIT */
static int remote_dep_mpi_save_activate_cb(dague_execution_unit_t* eu_context,
                                           dague_comm_callback_t* cb, MPI_Status* status);
static void remote_dep_mpi_get_start(dague_execution_unit_t* eu_context, dague_remote_deps_t* deps);
static void remote_dep_mpi_get_end( dague_execution_unit_t* eu_context, int idx, dague_remote_deps_t* deps );
static int
remote_dep_mpi_get_end_cb(dague_execution_unit_t* eu_context,
                          dague_comm_callback_t* cb, MPI_Status* status);
static void remote_dep_mpi_new_object( dague_execution_unit_t* eu_context, dep_cmd_item_t *item );

#if DAGUE_DEBUG_VERBOSE != 0
static inline char*
remote_dep_cmd_to_string(remote_dep_wire_activate_t* origin,
                         char* str,
                         size_t len)
{
    dague_execution_context_t task;

    task.dague_handle = dague_handle_lookup( origin->handle_id );
    task.function     = task.dague_handle->functions_array[origin->function_id];
    memcpy(&task.locals, origin->locals, sizeof(assignment_t) * task.function->nb_locals);
    task.priority     = 0xFFFFFFFF;
    return dague_snprintf_execution_context(str, len, &task);
}
#endif

static pthread_t dep_thread_id;
dague_dequeue_t dep_cmd_queue;
dague_list_t    dep_cmd_fifo;             /* ordered non threaded fifo */
dague_list_t    dep_activates_fifo;       /* ordered non threaded fifo */
dague_list_t    dep_activates_noobj_fifo; /* non threaded fifo */
dague_list_t    dep_put_fifo;             /* ordered non threaded fifo */

/* help manage the messages in the same category, where a category is either messages
 * to the same destination, or with the same action key.
 */
static dep_cmd_item_t** dague_mpi_same_pos_items;
static int dague_mpi_same_pos_items_size;

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

    OBJ_CONSTRUCT(&dep_cmd_queue, dague_dequeue_t);
    OBJ_CONSTRUCT(&dep_cmd_fifo, dague_list_t);

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
        OBJ_CONSTRUCT(item, dague_list_item_t);
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

    assert(NULL == dague_dequeue_pop_front(&dep_cmd_queue));
    OBJ_DESTRUCT(&dep_cmd_queue);
    assert(NULL == dague_dequeue_pop_front(&dep_cmd_fifo));
    OBJ_DESTRUCT(&dep_cmd_fifo);

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
    OBJ_CONSTRUCT(item, dague_list_item_t);
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
        whatsup = remote_dep_dequeue_nothread_progress(context);
    } while(-1 != whatsup);
    /* Release all resources */
    remote_dep_mpi_fini(context);
    return (void*)context;
}

static int remote_dep_dequeue_new_object(dague_handle_t* obj)
{
    if(!mpi_initialized) return 0;
    dep_cmd_item_t* item = (dep_cmd_item_t*)calloc(1, sizeof(dep_cmd_item_t));
    OBJ_CONSTRUCT(item, dague_list_item_t);
    item->action = DEP_NEW_OBJECT;
    item->priority = 0;
    item->cmd.new_object.obj = obj;
    dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*)item);
    return 1;
}

static int remote_dep_dequeue_send(int rank,
                                   dague_remote_deps_t* deps)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    OBJ_CONSTRUCT(item, dague_list_item_t);
    item->action   = DEP_ACTIVATE;
    item->priority = deps->max_priority;
    item->cmd.activate.peer             = rank;
    item->cmd.activate.task.deps        = (remote_dep_datakey_t)deps;
    item->cmd.activate.task.output_mask = 0;
    item->cmd.activate.task.tag         = 0;
    dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*)item);
    return 1;
}

void dague_remote_dep_memcpy(dague_execution_unit_t* eu_context,
                             dague_handle_t* dague_handle,
                             dague_data_copy_t *dst,
                             dague_data_copy_t *src,
                             dague_dep_data_description_t* data)
{
    assert( dst );
    dep_cmd_item_t* item = (dep_cmd_item_t*)calloc(1, sizeof(dep_cmd_item_t));
    OBJ_CONSTRUCT(item, dague_list_item_t);
    item->action = DEP_MEMCPY;
    item->priority = 0;
    item->cmd.memcpy.dague_handle = dague_handle;
    item->cmd.memcpy.source       = src;
    item->cmd.memcpy.destination  = dst;
    item->cmd.memcpy.datatype     = data->layout;
    item->cmd.memcpy.displ_s      = data->displ;
    item->cmd.memcpy.displ_r      = 0;
    item->cmd.memcpy.count        = data->count;

    OBJ_RETAIN(src);
    remote_dep_inc_flying_messages(dague_handle, eu_context->virtual_process->dague_context);

    dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);
}

#define is_inplace(ctx,dep) NULL
#define is_read_only(ctx,dep) NULL

/**
 * This function is called from the task successors iterator. It exists for a
 * single purpose: to retrieve the datatype involved with the operation. Thus,
 * once a datatype has been succesfully retrieved it must cancel the iterator
 * progress in order to return ASAP the datatype to the communication engine.
 */
dague_ontask_iterate_t
remote_dep_mpi_retrieve_datatype(dague_execution_unit_t *eu,
                                 const dague_execution_context_t *newcontext,
                                 const dague_execution_context_t *oldcontext,
                                 const dep_t* dep,
                                 dague_dep_data_description_t* data,
                                 int src_rank, int dst_rank, int dst_vpid,
                                 void *param)
{
    (void)eu; (void)oldcontext; (void)dst_vpid; (void)newcontext;
    if( dst_rank != eu->virtual_process->dague_context->my_rank )
        return DAGUE_ITERATE_CONTINUE;

    dague_remote_deps_t *deps                = (dague_remote_deps_t*)param;
    struct remote_dep_output_param_s* output = &deps->output[dep->dep_datatype_index];

    dague_data_t* data_arena = is_read_only(oldcontext, dep);
    if(NULL == data_arena) {
        output->deps_mask &= ~(1U << dep->dep_index); /* unmark all data that are RO we already hold from previous tasks */
    } else {
        output->deps_mask |= (1U << dep->dep_index); /* mark all data that are not RO */
        data_arena = is_inplace(oldcontext, dep);  /* Can we do it inplace */
    }
    output->data     = *data;
    data_arena = dague_arena_get(data->arena, data->count);
    /* if still NULL allocate it */
    output->data.data = dague_data_get_copy(data_arena, 0);

    deps->priority   = oldcontext->priority;
    deps->incoming_mask |= (1U << dep->dep_datatype_index);
    deps->root       = src_rank;
    return DAGUE_ITERATE_STOP;
}

/**
 * Retrieve the datatypes involved in this communication. In addition the flag
 * DAGUE_ACTION_RECV_INIT_REMOTE_DEPS set the priority to the maximum priority
 * of all the children.
 */
static int
remote_dep_get_datatypes(dague_execution_unit_t* eu_context,
                         dague_remote_deps_t* origin)
{
    dague_execution_context_t task;
    uint32_t i, j, k, local_mask = 0;

    assert(NULL == origin->dague_handle);
    task.dague_handle = dague_handle_lookup(origin->msg.handle_id);
    if( NULL == task.dague_handle )
        return -1; /* the dague object doesn't exist yet */
    task.dague_handle = origin->dague_handle;
    task.function     = task.dague_handle->functions_array[origin->msg.function_id];
    task.priority     = 0;  /* unknown yet */
    for(i = 0; i < task.function->nb_locals; i++)
        task.locals[i] = origin->msg.locals[i];

    /* We need to convert from a dep_datatype_index mask into a dep_index mask. However,
     * in order to be able to use the above iterator we need to be able to identify the
     * dep_index for each particular datatype index, and call the iterate_successors on
     * each of the dep_index sets.
     */
    for(k = 0; origin->msg.output_mask>>k; k++) {
        if(!(origin->msg.output_mask & (1U<<k))) continue;
        for(local_mask = i = 0; NULL != task.function->out[i]; i++ ) {
            if(!(task.function->out[i]->flow_datatype_mask & (1U<<k))) continue;
            for(j = 0; NULL != task.function->out[i]->dep_out[j]; j++ )
                if(k == task.function->out[i]->dep_out[j]->dep_datatype_index)
                    local_mask |= (1U << task.function->out[i]->dep_out[j]->dep_index);
            if( 0 != local_mask ) break;  /* we have our local mask, go get the datatype */
        }
        DEBUG3(("MPI:\tRetrieve datatype with mask 0x%x (remote_dep_get_datatypes)\n", local_mask));
        task.function->iterate_successors(eu_context, &task,
                                          local_mask,
                                          remote_dep_mpi_retrieve_datatype,
                                          origin);
    }
    /**
     * At this point the msg->output_mask contains the root mask, and should be
     * keep as is and be propagated down the communication pattern. On the
     * origin->incoming_mask we have the mask of all local data to be retrieved from
     * the predecessor.
     */
    return 0;
}

/**
 * Trigger the local reception of a remote task data. Put the data in the
 * correct location then call the release_deps.
 */
static dague_remote_deps_t*
remote_dep_release_incoming(dague_execution_unit_t* eu_context,
                            dague_remote_deps_t* origin,
                            remote_dep_datakey_t complete_mask)
{
    dague_execution_context_t task;
    const dague_flow_t* target;
    int i, pidx = 0;
    uint32_t action_mask = 0;

    /* Update the mask of remaining dependencies to avoid releasing the same outputs twice */
    assert((origin->incoming_mask & complete_mask) == complete_mask);
    origin->incoming_mask ^= complete_mask;

    task.dague_handle = origin->dague_handle;
    task.function = task.dague_handle->functions_array[origin->msg.function_id];
    task.priority = origin->priority;
    for(i = 0; i < task.function->nb_locals;
        task.locals[i] = origin->msg.locals[i], i++);
    for(i = 0; i < task.function->nb_flows;
        task.data[i].data_in = task.data[i].data_out = NULL, task.data[i].data_repo = NULL, i++);

    target = task.function->out[pidx];
    for(i = 0; complete_mask>>i; i++) {
        assert(i < MAX_PARAM_COUNT);
        if( !((1U<<i) & complete_mask) ) continue;
        while( !((1U<<i) & target->flow_datatype_mask) ) {
            target = task.function->out[++pidx];
            assert(NULL != target);
        }
        DEBUG3(("MPI:\tDATA %p(%s) released from %p[%d] flow idx %d\n",
                DAGUE_DATA_COPY_GET_PTR(origin->output[i].data.data), target->name, origin, i, target->flow_index));
        task.data[target->flow_index].data_repo = NULL;
        task.data[target->flow_index].data_in   = origin->output[i].data.data;
    }

#ifdef DAGUE_DIST_COLLECTIVES
    /**
     * There is a catch here. If we release the last dep below we can run in a
     * case where the last task is executed, then completed and the object is
     * released before we have the opportunity to propagate the collective.
     * Thus, in order to avoid this case we have to propagate the activation
     * before releasing the last set of local tasks.
     */
    if(0 == origin->incoming_mask) {
        dague_remote_dep_propagate(eu_context, &task, origin);
        /* don't change the internals of the origin from now on */
    }
#endif  /* DAGUE_DIST_COLLECTIVES */

    /* We need to convert from a dep_datatype_index mask into a dep_index mask */
    for(int i = 0; NULL != task.function->out[i]; i++ ) {
        target = task.function->out[i];
        if( !(complete_mask & target->flow_datatype_mask) ) continue;
        for(int j = 0; NULL != target->dep_out[j]; j++ )
            if(complete_mask & (1U << target->dep_out[j]->dep_datatype_index))
                action_mask |= (1U << target->dep_out[j]->dep_index);
    }
    DEBUG3(("MPI:\tTranslate mask from 0x%lx to 0x%x (remote_dep_release_incoming)\n", complete_mask, action_mask));
    (void)task.function->release_deps(eu_context, &task,
                                      action_mask | DAGUE_ACTION_RELEASE_LOCAL_DEPS,
                                      NULL);
    assert(0 == (origin->incoming_mask & complete_mask));

    if(0 == origin->incoming_mask) {  /* if necessary release the deps */
#if !defined(DAGUE_DIST_COLLECTIVES)
        /**
         * Release the dependency owned by the communication engine for all data
         * that has been internally allocated by the engine.
         */
        for(i = 0; origin->outgoing_mask>>i; i++) {
            assert(i < MAX_PARAM_COUNT);
            if( !((1U<<i) & complete_mask) ) continue;
            if( NULL != origin->output[i].data.ptr )  /* don't release the CONTROLs */
                DAGUE_DATA_COPY_RELEASE(origin->output[i].data.data);
        }
        remote_deps_free(origin);
#endif  /* !DAGUE_DIST_COLLECTIVES */
        origin = NULL;
    }
    return origin;
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
    dague_dequeue_destruct(&dep_cmd_queue);
    return 0;
}
#endif

static int remote_dep_dequeue_nothread_progress(dague_context_t* context)
{
    dague_list_item_t *items;
    dep_cmd_item_t *item, *same_pos;
    dague_list_t temp_list;
    int ret = 0, how_many, position;
    dague_execution_unit_t* eu_context = context->virtual_processes[0]->execution_units[0];

    OBJ_CONSTRUCT(&temp_list, dague_list_t);
 check_pending_queues:
    /* Move a number of tranfers from the shared dequeue into our ordered lifo. */
    how_many = 0;
    while( NULL != (item = (dep_cmd_item_t*) dague_dequeue_try_pop_front(&dep_cmd_queue)) ) {
        if( DEP_CTL == item->action ) {
            /* A DEP_CTL is a barrier that must not be crossed, flush the
             * ordered fifo and don't add anything until it is consumed */
            if( dague_ulist_is_empty(&dep_cmd_fifo) && dague_ulist_is_empty(&temp_list) )
                goto handle_now;
            dague_dequeue_push_front(&dep_cmd_queue, (dague_list_item_t*)item);
            break;
        }
        how_many++;
        same_pos = NULL;
        /* Find the position in the array of the first possible item in the same category */
        position = (DEP_ACTIVATE == item->action) ? item->cmd.activate.peer : (context->nb_nodes + item->action);

        dague_list_item_singleton(&item->pos_list);
        same_pos = dague_mpi_same_pos_items[position];
        if((NULL != same_pos) && (same_pos->priority >= item->priority)) {
            /* insert the item in the peer list */
            dague_list_item_ring_push_sorted(&same_pos->pos_list, &item->pos_list, dep_mpi_pos_list);
        } else {
            if(NULL != same_pos) {
                /* this is the new head of the list. */
                dague_list_item_ring_push(&same_pos->pos_list, &item->pos_list);
                /* Remove previous elem from the priority list. The element
                 might be either in the dep_cmd_fifo if it is old enough to be
                 pushed there, or in the temp_list waiting to be moved
                 upstrea. Pay attention from which queue it is removed. */
#if defined(DAGUE_DEBUG)
                dague_list_nolock_remove((struct dague_list_t*)same_pos->super.belong_to, (dague_list_item_t*)same_pos);
#else
                dague_list_nolock_remove(NULL, (dague_list_item_t*)same_pos);
#endif
                dague_list_item_singleton((dague_list_item_t*)same_pos);
            }
            dague_mpi_same_pos_items[position] = item;
            /* And add ourselves in the temp list */
            dague_list_nolock_push_front(&temp_list, (dague_list_item_t*)item);
        }
        if(how_many > dague_param_nb_tasks_extracted)
            break;
    }
    if( !dague_ulist_is_empty(&temp_list) ) {
        /* Sort the temporary list */
        dague_list_nolock_sort(&temp_list, dep_cmd_prio);
        /* Remove the ordered items from the list, and clean the list */
        items = dague_list_nolock_unchain(&temp_list);
        /* Insert them into the locally ordered cmd_fifo */
        dague_list_nolock_chain_sorted(&dep_cmd_fifo, items, dep_cmd_prio);
    }
    /* Extract the head of the list and point the array to the correct value */
    if(NULL == (item = (dep_cmd_item_t*)dague_list_nolock_pop_front(&dep_cmd_fifo)) ) {
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
    position = (DEP_ACTIVATE == item->action) ? item->cmd.activate.peer : (context->nb_nodes + item->action);
    assert(DEP_CTL != item->action);
  handle_now:
    switch(item->action) {
    case DEP_CTL:
        ret = item->cmd.ctl.enable;
        OBJ_DESTRUCT(&temp_list);
        free(item);
        return ret;  /* FINI or OFF */
    case DEP_NEW_OBJECT:
        remote_dep_mpi_new_object(eu_context, item);
        break;
    case DEP_ACTIVATE:
        remote_dep_nothread_send(eu_context, &item);
        same_pos = item;
        goto have_same_pos;
    case DEP_MEMCPY:
        remote_dep_nothread_memcpy(eu_context, item);
        break;
    default:
        assert(0 && item->action); /* Not a valid action */
        break;
    }

    /* Correct the other structures */
    same_pos = (dep_cmd_item_t*)dague_list_item_ring_chop(&item->pos_list);
    if( NULL != same_pos)
        same_pos = container_of(same_pos, dep_cmd_item_t, pos_list);
    free(item);
  have_same_pos:
    if( NULL != same_pos) {
        dague_list_nolock_push_front(&temp_list, (dague_list_item_t*)same_pos);
    }
    dague_mpi_same_pos_items[position] = same_pos;

    goto check_pending_queues;
}

static int remote_dep_nothread_memcpy(dague_execution_unit_t* eu_context,
                                      dep_cmd_item_t *item)
{
    dep_cmd_t* cmd = &item->cmd;
    int rc = MPI_Sendrecv((char*)DAGUE_DATA_COPY_GET_PTR(cmd->memcpy.source     ) + cmd->memcpy.displ_s,
                          cmd->memcpy.count, cmd->memcpy.datatype, 0, 0,
                          (char*)DAGUE_DATA_COPY_GET_PTR(cmd->memcpy.destination) + cmd->memcpy.displ_r,
                          cmd->memcpy.count, cmd->memcpy.datatype, 0, 0,
                          MPI_COMM_SELF, MPI_STATUS_IGNORE);
    DAGUE_DATA_COPY_RELEASE(cmd->memcpy.source);
    remote_dep_dec_flying_messages(item->cmd.memcpy.dague_handle,
                                   eu_context->virtual_process->dague_context);
    return (MPI_SUCCESS == rc ? 0 : -1);
}

/******************************************************************************
 * ALL MPI SPECIFIC CODE GOES HERE
 ******************************************************************************/
enum {
    REMOTE_DEP_ACTIVATE_TAG = 0,
    REMOTE_DEP_GET_DATA_TAG,
    REMOTE_DEP_MAX_CTRL_TAG
} dague_remote_dep_tag_t;

#ifdef DAGUE_PROF_TRACE
static dague_thread_profiling_t* MPIctl_prof;
static dague_thread_profiling_t* MPIsnd_prof;
static dague_thread_profiling_t* MPIrcv_prof;
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
    MPIsnd_prof = dague_profiling_thread_init( 2*1024*1024, "MPI isend");
    MPIrcv_prof = dague_profiling_thread_init( 2*1024*1024, "MPI irecv");
}

static void remote_dep_mpi_profiling_fini(void)
{
    MPIsnd_prof = NULL;
    MPIrcv_prof = NULL;
    MPIctl_prof = NULL;
}

#define TAKE_TIME_WITH_INFO(PROF, KEY, I, src, dst, rdw) do {           \
        dague_profile_remote_dep_mpi_info_t __info;                     \
        dague_execution_context_t __exec_context;                       \
        dague_handle_t *__object = dague_handle_lookup( (rdw).handle_id ); \
        __exec_context.function = __object->functions_array[(rdw).function_id ]; \
        __exec_context.dague_handle = __object;                         \
        memcpy(&__exec_context.locals, (rdw).locals, MAX_LOCAL_COUNT * sizeof(assignment_t)); \
        dague_snprintf_execution_context(__info.func, 16, &__exec_context); \
        __info.rank_src = (src);                                        \
        __info.rank_dst = (dst);                                        \
        DAGUE_PROFILING_TRACE((PROF), (KEY), (I), PROFILE_OBJECT_ID_NULL, &__info); \
    } while(0)

#define TAKE_TIME(PROF, KEY, I) DAGUE_PROFILING_TRACE((PROF), (KEY), (I), PROFILE_OBJECT_ID_NULL, NULL);
#else
#define TAKE_TIME_WITH_INFO(PROF, KEY, I, src, dst, rdw) do {} while(0)
#define TAKE_TIME(PROF, KEY, I) do {} while(0)
#define remote_dep_mpi_profiling_init() do {} while(0)
#define remote_dep_mpi_profiling_fini() do {} while(0)
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

typedef int (*dague_comm_callback_f)(dague_execution_unit_t*,
                                     dague_comm_callback_t*,  /**< the associated callback structure */
                                     MPI_Status* status);     /**< the corresponding status */
struct dague_comm_callback_s {
    dague_comm_callback_f fct;
    long                  storage1;
    long                  storage2;
};

static MPI_Comm dep_comm;
static dague_comm_callback_t *array_of_callbacks;
static MPI_Request           *array_of_requests;
static int                   *array_of_indices;
static MPI_Status            *array_of_statuses;

/* TODO: fix heterogeneous restriction by using proper mpi datatypes */
#define dep_dtt MPI_BYTE
#define dep_count sizeof(remote_dep_wire_activate_t)
#define dep_extent dep_count
#define DEP_EAGER_BUFFER_SIZE (dep_extent+RDEP_MSG_EAGER_LIMIT)
#define datakey_dtt MPI_LONG
#define datakey_count 3
static char **dep_activate_buff;
static remote_dep_wire_get_t* dep_get_buff;

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
static int __VAL_NEXT_TAG = MIN_MPI_TAG;
static inline int next_tag(int k) {
    int __tag = __VAL_NEXT_TAG;
    if( __tag > (MAX_MPI_TAG-k) ) {
        printf("rank %d rollover: min %d < %d (+%d) < max %d\n", dague_debug_rank,
               MIN_MPI_TAG, __tag, k, MAX_MPI_TAG);
        __VAL_NEXT_TAG = __tag = MIN_MPI_TAG;
    } else
        __VAL_NEXT_TAG += k;
    return __tag;
}

static int remote_dep_mpi_init(dague_context_t* context)
{
    int i, mpi_tag_ub_exists, *ub;
    dague_comm_callback_t* cb;

    OBJ_CONSTRUCT(&dep_activates_fifo, dague_list_t);
    OBJ_CONSTRUCT(&dep_activates_noobj_fifo, dague_list_t);
    OBJ_CONSTRUCT(&dep_put_fifo, dague_list_t);

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
#if DAGUE_DEBUG_VERBOSE != 0
        if( MAX_MPI_TAG < INT_MAX ) {
            WARNING(("MPI:\tYour MPI implementation defines the maximal TAG value to %d (0x%08x), which might be too small should you have more than %d simultaneous remote dependencies\n",
                    MAX_MPI_TAG, (unsigned int)MAX_MPI_TAG, MAX_MPI_TAG / MAX_PARAM_COUNT));
        }
#endif
    }

    MPI_Comm_size(dep_comm, &(context->nb_nodes));
    MPI_Comm_rank(dep_comm, &(context->my_rank));

    dague_mpi_same_pos_items_size = context->nb_nodes + (int)DEP_LAST;
    dague_mpi_same_pos_items = (dep_cmd_item_t**)calloc(dague_mpi_same_pos_items_size,
                                                        sizeof(dep_cmd_item_t*));
    /* Extend the number of pending activations if we have a large number of peers */
    if( context->nb_nodes > (10*dague_comm_activations_max) )
        dague_comm_activations_max = context->nb_nodes / 10;
    if( context->nb_nodes > (10*dague_comm_data_get_max) )
        dague_comm_data_get_max = context->nb_nodes / 10;
    DEP_NB_REQ = (dague_comm_activations_max + dague_comm_data_get_max +
                  dague_comm_gets_max + dague_comm_puts_max);

    array_of_callbacks = (dague_comm_callback_t*)calloc(DEP_NB_REQ, sizeof(dague_comm_callback_t));
    array_of_requests  = (MPI_Request*)calloc(DEP_NB_REQ, sizeof(MPI_Request));
    array_of_indices   = (int*)calloc(DEP_NB_REQ, sizeof(int));
    array_of_statuses  = (MPI_Status*)calloc(DEP_NB_REQ, sizeof(MPI_Status));
    for(i = 0; i < DEP_NB_REQ; i++)
        array_of_requests[i] = MPI_REQUEST_NULL;

    /* Create all the persistent receives (activation and GET orders) and start them */
    dep_activate_buff = (char**)calloc(dague_comm_activations_max, sizeof(char*));
    dep_activate_buff[0] = (char*)calloc(dague_comm_activations_max, DEP_EAGER_BUFFER_SIZE*sizeof(char));
    for(i = 0; i < dague_comm_activations_max; i++) {
        dep_activate_buff[i] = dep_activate_buff[0] + i * DEP_EAGER_BUFFER_SIZE*sizeof(char);
        MPI_Recv_init(dep_activate_buff[i], DEP_EAGER_BUFFER_SIZE, MPI_PACKED,
                      MPI_ANY_SOURCE, REMOTE_DEP_ACTIVATE_TAG, dep_comm,
                      &array_of_requests[dague_comm_last_active_req]);
        cb = &array_of_callbacks[dague_comm_last_active_req];
        cb->fct      = remote_dep_mpi_save_activate_cb;
        cb->storage1 = dague_comm_last_active_req;
        cb->storage2 = i;
        MPI_Start(&array_of_requests[dague_comm_last_active_req]);
        dague_comm_last_active_req++;
    }

    dep_get_buff = (remote_dep_wire_get_t*)calloc(dague_comm_data_get_max, sizeof(remote_dep_wire_get_t));
    for(i = 0; i < dague_comm_data_get_max; i++) {
        MPI_Recv_init(&dep_get_buff[i], datakey_count, datakey_dtt,
                      MPI_ANY_SOURCE, REMOTE_DEP_GET_DATA_TAG, dep_comm,
                      &array_of_requests[dague_comm_last_active_req]);
        cb = &array_of_callbacks[dague_comm_last_active_req];
        cb->fct      = remote_dep_mpi_save_put_cb;
        cb->storage1 = dague_comm_last_active_req;
        cb->storage2 = i;
        MPI_Start(&array_of_requests[dague_comm_last_active_req]);
        dague_comm_last_active_req++;
    }

    remote_dep_mpi_profiling_init();
    return 0;
}

static int remote_dep_mpi_fini(dague_context_t* context)
{
    int i, flag;
    MPI_Status status;

    remote_dep_mpi_profiling_fini();

    /* Cancel and release all persistent requests */
    for(i = 0; i < dague_comm_activations_max + dague_comm_data_get_max; i++) {
        MPI_Cancel(&array_of_requests[i]);
        MPI_Test(&array_of_requests[i], &flag, &status);
        MPI_Request_free(&array_of_requests[i]);
    }
    dague_comm_last_active_req -= (dague_comm_activations_max + dague_comm_data_get_max);
    assert(0 == dague_comm_last_active_req);

    free(array_of_callbacks); array_of_callbacks = NULL;
    free(array_of_requests);  array_of_requests  = NULL;
    free(array_of_indices);   array_of_indices   = NULL;
    free(array_of_statuses);  array_of_statuses  = NULL;

    free(dague_mpi_same_pos_items); dague_mpi_same_pos_items = NULL;
    dague_mpi_same_pos_items_size = 0;

    free(dep_get_buff); dep_get_buff = NULL;
    free(dep_activate_buff[0]);
    free(dep_activate_buff); dep_activate_buff = NULL;

    OBJ_DESTRUCT(&dep_activates_fifo);
    OBJ_DESTRUCT(&dep_activates_noobj_fifo);
    OBJ_DESTRUCT(&dep_put_fifo);
    MPI_Comm_free(&dep_comm);
    (void)context;
    return 0;
}

static int remote_dep_mpi_on(dague_context_t* context)
{
#ifdef DAGUE_PROF_TRACE
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
 * into the provided buffer. If possible (eager allowed and enough room
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
    dague_remote_deps_t *deps = (dague_remote_deps_t*)item->cmd.activate.task.deps;
    remote_dep_wire_activate_t* msg = &deps->msg;
    int k, dsize, saved_position = *position;
    uint32_t peer_bank, peer_mask, expected = 0;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, 128);
#endif

    peer_bank = peer / (sizeof(uint32_t) * 8);
    peer_mask = 1U << (peer % (sizeof(uint32_t) * 8));

    MPI_Pack_size(dep_count, dep_dtt, dep_comm, &dsize);
    if( (length - (*position)) < dsize ) {  /* no room. bail out */
        DEBUG3(("Can't pack at %d/%d. Bail out!\n", *position, length));
        return 1;
    }
    /* Don't pack yet, we need to update the length field before packing */
    *position  += dsize;
    assert((0 != msg->output_mask) &&   /* this should be preset */
           (msg->output_mask & deps->outgoing_mask) == deps->outgoing_mask);
    msg->length = 0;
    item->cmd.activate.task.output_mask = 0;  /* clean start */
    /* Treat for special cases: CTL, Eager, etc... */
    for(k = 0; deps->outgoing_mask >> k; k++) {
        if( !((1U << k) & deps->outgoing_mask )) continue;
        if( !(deps->output[k].rank_bits[peer_bank] & peer_mask) ) continue;

        /* Remove CTL from the message we expect to send */
#if defined(DAGUE_PROF_DRY_DEP)
        deps->output[k].data.arena = NULL; /* make all data a control */
#endif
        if(NULL == deps->output[k].data.arena) {
            DEBUG2((" CTL\t%s\tparam %d\tdemoted to be a control\n", tmp, k));
            continue;
        }
        assert(deps->output[k].data.count > 0);
        if(dague_param_enable_eager) {
            /* Embed data (up to eager size) with the activate msg */
            MPI_Pack_size(deps->output[k].data.count, deps->output[k].data.layout,
                          dep_comm, &dsize);
            if((length - (*position)) >= dsize) {
                MPI_Pack((char*)DAGUE_DATA_COPY_GET_PTR(deps->output[k].data.data) + deps->output[k].data.displ,
                         deps->output[k].data.count, deps->output[k].data.layout,
                         packed_buffer, length, position, dep_comm);
                DEBUG2((" EGR\t%s\tparam %d\teager piggyback in the activate msg (%d/%d)\n",
                        tmp, k, *position, length));
                msg->length += dsize;
                continue;  /* go to the next */
            } else if( 0 != saved_position ) {
                DEBUG3(("DATA\t%s\tparam %d\texceed buffer length. Start again from here next iteration\n",
                        tmp, k));
                *position = saved_position;
                return 1;
            }
            /* the data doesn't fit in the buffer. */
        }
        expected++;
        item->cmd.activate.task.output_mask |= (1U<<k);
        DEBUG2(("DATA\t%s\tparam %d\tdeps %p send on demand (increase deps counter by %d [%d])\n",
                tmp, k, deps, expected, deps->pending_ack));
    }
    if(expected)
        dague_atomic_add_32b((int32_t*)&deps->pending_ack, expected);  /* Keep track of the inflight data */
    /* We can only have up to k data sends related to this remote_dep (include the order itself) */
    item->cmd.activate.task.tag = next_tag(k);
    msg->tag = item->cmd.activate.task.tag;

    DEBUG(("MPI:\tTO\t%d\tActivate\t% -8s\n"
           "    \t\t\twith datakey %lx\tmask %lx\t(tag=%d) eager mask %lu length %d\n",
           peer, tmp, msg->deps, msg->output_mask, msg->tag,
           msg->output_mask ^ item->cmd.activate.task.output_mask, msg->length));
    /* And now pack the updated message (msg->length and msg->output_mask) itself. */
    MPI_Pack(msg, dep_count, dep_dtt, packed_buffer, length, &saved_position, dep_comm);
    return 0;
}

/**
 * Starting with a particular item pack as many remote_dep_wire_activate
 * messages with the same destination (from the item ring associated with
 * pos_list) into a buffer. Upon completion the entire buffer is send to the
 * remote peer, the completed messages are released and the header is updated to
 * the next unsent message.
 */
static int remote_dep_nothread_send(dague_execution_unit_t* eu_context,
                                    dep_cmd_item_t **head_item)
{
    dague_remote_deps_t *deps;
    dep_cmd_item_t *item = *head_item;
    dague_list_item_t* ring = NULL;
    char packed_buffer[DEP_EAGER_BUFFER_SIZE];
    int peer, position = 0;

    peer = item->cmd.activate.peer;  /* this doesn't change */
    deps = (dague_remote_deps_t*)item->cmd.activate.task.deps;
    TAKE_TIME_WITH_INFO(MPIctl_prof, MPI_Activate_sk, act,
                        eu_context->virtual_process->dague_context->my_rank, peer, deps->msg);
  pack_more:
    assert(peer == item->cmd.activate.peer);

    deps = (dague_remote_deps_t*)item->cmd.activate.task.deps;

    dague_list_item_singleton((dague_list_item_t*)item);
    if( 0 == remote_dep_mpi_pack_dep(peer, item, packed_buffer,
                                     DEP_EAGER_BUFFER_SIZE, &position) ) {
        /* space left on the buffer. Move to the next item with the same destination */
        dep_cmd_item_t* next = (dep_cmd_item_t*)dague_list_item_ring_chop(&item->pos_list);
        if( NULL == ring ) ring = (dague_list_item_t*)item;
        else dague_list_item_ring_push(ring, (dague_list_item_t*)item);
        if( NULL != next ) {
            item = container_of(next, dep_cmd_item_t, pos_list);
            assert(DEP_ACTIVATE == item->action);
            if( dague_param_enable_aggregate ) {
                goto pack_more;
            }
        } else item = NULL;
    }
    *head_item = item;

    DAGUE_STATACC_ACCUMULATE_MSG(counter_control_messages_sent, packed, MPI_PACKED);
    MPI_Send((void*)packed_buffer, position, MPI_PACKED, peer, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, act++);
    DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(peer, (void*)&deps->msg, deps->msg);

    do {
        item = (dep_cmd_item_t*)ring;
        ring = dague_list_item_ring_chop(ring);
        deps = (dague_remote_deps_t*)item->cmd.activate.task.deps;

#if RDEP_MSG_SHORT_LIMIT != 0
        if( 0 != item->cmd.activate.task.output_mask ) {
            remote_dep_mpi_put_short(eu_context, item);
        } else
#endif   /* RDEP_MSG_SHORT_LIMIT != 0 */
            free(item);  /* only large messages are left */

        remote_dep_complete_and_cleanup(&deps, 1, eu_context->virtual_process->dague_context);
    } while( NULL != ring );
    return 0;
}

static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context)
{
    MPI_Status *status;
    int ret = 0, idx, outcount, pos;
    dague_comm_callback_t* cb;

    if( !DAGUE_THREAD_IS_MASTER(eu_context) ) return 0;

    do {
        MPI_Testsome(dague_comm_last_active_req, array_of_requests,
                     &outcount, array_of_indices, array_of_statuses);
        if(0 == outcount) goto feed_more_work;  /* can we push some more work? */

        /* Trigger the callbacks */
        for( idx = 0; idx < outcount; idx++ ) {

            cb = &array_of_callbacks[array_of_indices[idx]];
            status = &(array_of_statuses[idx]);

            cb->fct(eu_context, cb, status);
            ret++;
        }

        /* Compact the pending requests in order to minimize the testsome waiting time.
         * Parsing the array_of_indeices in the reverse order insure a smooth and fast
         * compacting.
         */
        for( idx = outcount-1; idx >= 0; idx-- ) {
            pos = array_of_indices[idx];
            if(MPI_REQUEST_NULL != array_of_requests[pos])
                continue;  /* The callback replaced the completed request, keep going */
            /* Get the last active callback to replace the empty one */
            dague_comm_last_active_req--;
            if( dague_comm_last_active_req > pos ) {
                array_of_requests[pos]  = array_of_requests[dague_comm_last_active_req];
                array_of_callbacks[pos] = array_of_callbacks[dague_comm_last_active_req];
            }
            array_of_requests[dague_comm_last_active_req] = MPI_REQUEST_NULL;
        }

      feed_more_work:
        if((dague_comm_gets < dague_comm_gets_max) && !dague_ulist_is_empty(&dep_activates_fifo)) {
            dague_remote_deps_t* deps = (dague_remote_deps_t*)dague_ulist_fifo_pop(&dep_activates_fifo);
            remote_dep_mpi_get_start(eu_context, deps);
        }
        if((dague_comm_puts < dague_comm_puts_max) && !dague_ulist_is_empty(&dep_put_fifo)) {
            dep_cmd_item_t* item = (dep_cmd_item_t*)dague_ulist_fifo_pop(&dep_put_fifo);
            remote_dep_mpi_put_start(eu_context, item);
        }
        if(0 == outcount) return ret;
    } while(1);
}

#if RDEP_MSG_SHORT_LIMIT != 0
/**
 * Compute the mask of all dependencies associated with a defined deps that can
 * be embedded in the outgoing message. This takes in account the control data
 * (with zero length), the eager data up to the allowed max amount of the
 * message as well as the short protocol (data that will follow shorthly without
 * a need for rendez-vous).
 */
static remote_dep_datakey_t
remote_dep_mpi_short_which(const dague_remote_deps_t* deps,
                           remote_dep_datakey_t output_mask)
{
    for(int k = 0; output_mask>>k; k++) {
        if( !(output_mask & (1U<<k)) ) continue;            /* No dependency */
        if( NULL == deps->output[k].data.arena ) continue;  /* CONTROL dependency */
        size_t extent = deps->output[k].data.arena->elem_size * deps->output[k].data.count;

        if( (extent <= (RDEP_MSG_SHORT_LIMIT)) | (extent <= (RDEP_MSG_EAGER_LIMIT)) ) {
            DEBUG3(("MPI:\tPEER\tNA\t%5s MODE  k=%d\tsize=%d <= %d\t(tag=base+%d)\n",
                    (extent <= (RDEP_MSG_EAGER_LIMIT) ? "Eager" : "Short"),
                    k, extent, RDEP_MSG_SHORT_LIMIT, k));
            continue;
        }
        output_mask ^= (1U<<k);
    }
    return output_mask;
}

static void remote_dep_mpi_put_short(dague_execution_unit_t* eu_context,
                                     dep_cmd_item_t* item)
{
    remote_dep_wire_get_t* task = &item->cmd.activate.task;
    dague_remote_deps_t* deps = (dague_remote_deps_t*)task->deps;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN);
#endif

    item->cmd.activate.task.output_mask = remote_dep_mpi_short_which(deps, task->output_mask);
    if( 0 == item->cmd.activate.task.output_mask ) {
        free(item);  /* nothing to do, no reason to keep it */
        return;
    }

    /* Check if we can process it right now */
    if( dague_comm_puts < dague_comm_puts_max ) {
        remote_dep_mpi_put_start(eu_context, item);
        return;
    }
    DEBUG3(("MPI: Put Short DELAYED for %s from %d tag %u which 0x%x (deps %p)\n",
            tmp, item->cmd.activate.peer, task->tag, task->output_mask, deps));

    dague_ulist_push_sorted(&dep_put_fifo, (dague_list_item_t*)item, dep_cmd_prio);
}
#endif  /* RDEP_MSG_SHORT_LIMIT != 0 */

static int
remote_dep_mpi_save_put_cb(dague_execution_unit_t* eu_context,
                           dague_comm_callback_t* cb,
                           MPI_Status* status)
{
    remote_dep_wire_get_t* task;
    dague_remote_deps_t *deps;
    dep_cmd_item_t* item;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
#endif

    item = (dep_cmd_item_t*) malloc(sizeof(dep_cmd_item_t));
    OBJ_CONSTRUCT(&item->super, dague_list_item_t);
    item->action = 0 /* DEP_GET_DATA */;
    item->cmd.activate.peer = status->MPI_SOURCE;

    task = &(item->cmd.activate.task);
    memcpy(task, &dep_get_buff[cb->storage2], sizeof(remote_dep_wire_get_t));
    deps = (dague_remote_deps_t*) (uintptr_t) task->deps;
    assert(0 != deps->pending_ack);
    assert(0 != deps->outgoing_mask);
    item->priority = deps->max_priority;

    /* Get the highest priority PUT operation */
    dague_ulist_push_sorted(&dep_put_fifo, (dague_list_item_t*)item, dep_cmd_prio);
    if( dague_comm_puts < dague_comm_puts_max ) {
        item = (dep_cmd_item_t*)dague_ulist_fifo_pop(&dep_put_fifo);
        remote_dep_mpi_put_start(eu_context, item);
    } else
        DEBUG3(("MPI: Put DELAYED for %s from %d tag %u which 0x%x (deps %p)\n",
                remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), item->cmd.activate.peer,
                task->tag, task->output_mask, (void*)deps));
    /* Let's re-enable the pending request in the same position */
    MPI_Start(&array_of_requests[cb->storage1]);
    return 0;
}

static void
remote_dep_mpi_put_start(dague_execution_unit_t* eu_context,
                         dep_cmd_item_t* item)
{
    remote_dep_wire_get_t* task = &(item->cmd.activate.task);
    dague_remote_deps_t* deps = (dague_remote_deps_t*) (uintptr_t) task->deps;
    int k, nbdtt, tag = task->tag;
    dague_comm_callback_t* cb;
    void* dataptr;
    MPI_Datatype dtt;
#if DAGUE_DEBUG_VERBOSE >= 2
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif

    (void)eu_context;
    DEBUG_MARK_CTL_MSG_GET_RECV(item->cmd.activate.peer, (void*)task, task);

#if !defined(DAGUE_PROF_DRY_DEP)
    assert(task->output_mask);
    DEBUG3(("MPI:\tPUT mask=%lx deps 0x%lx\n", task->output_mask, task->deps));

    for(k = 0; task->output_mask>>k; k++) {
        assert(k < MAX_PARAM_COUNT);
        if(!((1U<<k) & task->output_mask)) continue;

        if(dague_comm_puts == dague_comm_puts_max) {
            DEBUG3(("MPI:\treach PUT limit for deps 0x%lx. Reschedule.\n", deps));
            dague_ulist_push_front(&dep_put_fifo, (dague_list_item_t*)item);
            return;
        }
        DEBUG3(("MPI:\t[idx %d mask(0x%x / 0x%x)] %p, %p\n", k, (1U<<k), task->output_mask,
                deps->output[k].data.data, DAGUE_DATA_COPY_GET_PTR(deps->output[k].data.data)));
        dataptr = DAGUE_DATA_COPY_GET_PTR(deps->output[k].data.data);
        dtt     = deps->output[k].data.layout;
        nbdtt   = deps->output[k].data.count;
#if DAGUE_DEBUG_VERBOSE >= 2
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG2(("MPI:\tTO\t%d\tPut START\tunknown \tk=%d\twith deps 0x%lx at %p type %s\t(tag=%d displ = %ld)\n",
               item->cmd.activate.peer, k, task->deps, dataptr, type_name, tag+k, deps->output[k].data.displ));
#endif

        TAKE_TIME_WITH_INFO(MPIsnd_prof, MPI_Data_plds_sk, k,
                            eu_context->virtual_process->dague_context->my_rank,
                            item->cmd.activate.peer, deps->msg);
        task->output_mask ^= (1U<<k);
        MPI_Isend((char*)dataptr + deps->output[k].data.displ, nbdtt, dtt,
                  item->cmd.activate.peer, tag + k, dep_comm,
                  &array_of_requests[dague_comm_last_active_req]);
        cb = &array_of_callbacks[dague_comm_last_active_req];
        cb->fct      = remote_dep_mpi_put_end_cb;
        cb->storage1 = (long)deps;
        cb->storage2 = k;
        dague_comm_last_active_req++;
        dague_comm_puts++;
        assert(dague_comm_last_active_req <= DEP_NB_REQ);
        DEBUG_MARK_DTA_MSG_START_SEND(item->cmd.activate.peer, dataptr, tag+k);
    }
#endif  /* !defined(DAGUE_PROF_DRY_DEP) */
    if(0 == task->output_mask)
        free(item);
}

static int
remote_dep_mpi_put_end_cb(dague_execution_unit_t* eu_context,
                          dague_comm_callback_t* cb,
                          MPI_Status* status)
{
    dague_remote_deps_t* deps = (dague_remote_deps_t*)cb->storage1;

    DEBUG2(("MPI:\tTO\tna\tPut END  \tunknown \tk=%d\twith deps %p\tparams %lx\t(tag=%d) data ptr %p\n",
            cb->storage2, deps, cb->storage2, status->MPI_TAG,
            deps->output[cb->storage2].data.data)); (void)status;
    DEBUG_MARK_DTA_MSG_END_SEND(status->MPI_TAG);
    TAKE_TIME(MPIsnd_prof, MPI_Data_plds_ek, cb->storage2);
    remote_dep_complete_and_cleanup(&deps, 1, eu_context->virtual_process->dague_context);
    dague_comm_puts--;
    return 0;
}

/**
 * An activation message has been received, and the remote_dep_wire_activate_t
 * part has already been extracted into the deps->msg. This function handle the
 * rest of the receiver logic, extract the possible eager and control data from
 * the buffer, post all the short protocol receives and all other local
 * cleanups.
 */
static void remote_dep_mpi_recv_activate(dague_execution_unit_t* eu_context,
                                         dague_remote_deps_t* deps,
                                         char* packed_buffer,
                                         int length,
                                         int* position)
{
    remote_dep_datakey_t complete_mask = 0;
    int k, dsize, tag = (int)deps->msg.tag;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN);
#endif
#if RDEP_MSG_SHORT_LIMIT != 0
    remote_dep_datakey_t short_which = remote_dep_mpi_short_which(deps, deps->incoming_mask);
#if !defined(DAGUE_PROF_DRY_DEP)
    MPI_Request reqs[MAX_PARAM_COUNT];
    int nb_reqs = 0, flag;
#endif  /* !defined(DAGUE_PROF_DRY_DEP) */
#endif  /* RDEP_MSG_SHORT_LIMIT != 0 */

    DEBUG(("MPI:\tFROM\t%d\tActivate\t% -8s\n"
           "\twith datakey %lx\tparams %lx length %d (pack buf %d/%d)\n",
           deps->from, tmp, deps->msg.deps, deps->incoming_mask,
           deps->msg.length, *position, length));
    for(k = 0; deps->incoming_mask>>k; k++) {
        if(!(deps->incoming_mask & (1U<<k))) continue;
        /* Check for all CTL messages, that do not carry payload */
        if(NULL == deps->output[k].data.arena) {
            DEBUG2(("MPI:\tHERE\t%d\tGet NONE\t% -8s\tk=%d\twith datakey %lx at <NA> type CONTROL\n",
                    deps->from, tmp, k, deps->msg.deps));
            deps->output[k].data.data = (void*)2; /* the first non zero even value */
            complete_mask |= (1U<<k);
            continue;
        }

        if( dague_param_enable_eager && (length > *position) ) {
            /* Check if the data is EAGERly embedded in the activate */
            MPI_Pack_size(deps->output[k].data.count, deps->output[k].data.layout,
                          dep_comm, &dsize);
            if((length - (*position)) >= dsize) {
                assert(NULL == deps->output[k].data.data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
                if(NULL == deps->output[k].data.data) {
                    dague_data_t* data_arena = dague_arena_get(deps->output[k].data.arena, deps->output[k].data.count);
                    deps->output[k].data.data = dague_data_get_copy(data_arena, 0);
                    DEBUG3(("MPI:\tMalloc new remote tile %p size %" PRIu64 " count = %" PRIu64 " displ = %" PRIi64 "\n",
                            deps->output[k].data.data, deps->output[k].data.arena->elem_size,
                            deps->output[k].data.count, deps->output[k].data.displ));
                    assert(deps->output[k].data.data != NULL);
                }
#ifndef DAGUE_PROF_DRY_DEP
                DEBUG2((" EGR\t%s\tparam %d\teager from the activate msg (%d/%d)\n",
                        tmp, k, dsize, length - *position));
                MPI_Unpack(packed_buffer, length, position,
                           (char*)DAGUE_DATA_COPY_GET_PTR(deps->output[k].data.data) + deps->output[k].data.displ,
                           deps->output[k].data.count, deps->output[k].data.layout, dep_comm);
#endif
                complete_mask |= (1U<<k);
                continue;
            }
        }
#if RDEP_MSG_SHORT_LIMIT != 0
       /* Check if we have SHORT deps to satisfy quickly */
        if( short_which & (1U<<k) ) {

            assert(NULL == deps->output[k].data.data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
            if(NULL == deps->output[k].data.data) {
                deps->output[k].data.data = dague_arena_get(deps->output[k].data.arena,
                                                            deps->output[k].data.count);
                DEBUG3(("MPI:\tMalloc new remote tile %p size %" PRIu64 " count = %" PRIu64 " displ = %" PRIi64 "(short)\n",
                        deps->output[k].data.data, deps->output[k].data.arena->elem_size,
                        deps->output[k].data.count, deps->output[k].data.displ));
                assert(deps->output[k].data.data != NULL);
            }
            DEBUG2(("MPI:\tFROM\t%d\tGet SHORT\t% -8s\tk=%d\twith datakey %lx at %p\t(tag=%d)\n",
                    deps->from, tmp, k, deps->msg.deps, DAGUE_DATA_COPY_GET_PTR(deps->output[k].data.data), tag+k));
#ifndef DAGUE_PROF_DRY_DEP
            MPI_Irecv((char*)DAGUE_DATA_COPY_GET_PTR(deps->output[k].data.data) + deps->output[k].data.displ,
                      deps->output[k].data.count, deps->output[k].data.layout,
                      deps->from, tag+k, dep_comm, &reqs[nb_reqs]);
            nb_reqs++;
            MPI_Testall(nb_reqs, reqs, &flag, MPI_STATUSES_IGNORE);  /* a little progress */
#endif
            complete_mask |= (1U<<k);
            continue;
        }
#endif
        DEBUG2(("MPI:\tFROM\t%d\tGet DATA\t% -8s\tk=%d\twith datakey %lx tag=%d (to be posted)\n",
                deps->from, tmp, k, deps->msg.deps, tag+k));
    }
#if (RDEP_MSG_SHORT_LIMIT != 0) && !defined(DAGUE_PROF_DRY_DEP)
    while(nb_reqs) {  /* 'till flag become true */
        MPI_Testall(nb_reqs, reqs, &flag, MPI_STATUSES_IGNORE);
        if(flag) break;
        remote_dep_mpi_progress(eu_context);
    }
#endif  /* (RDEP_MSG_SHORT_LIMIT != 0) && !defined(DAGUE_PROF_DRY_DEP) */
    assert(length == *position);

    /* Release all the already satisfied deps without posting the RDV */
    if(complete_mask) {
#if DAGUE_DEBUG_VERBOSE >= 2
        for(int k = 0; complete_mask>>k; k++)
            if((1U<<k) & complete_mask)
                DEBUG2(("MPI:\tHERE\t%d\tGet PREEND\t% -8s\tk=%d\twith datakey %lx at %p ALREADY SATISFIED\t(tag=%d)\n",
                        deps->from, tmp, k, deps->msg.deps, DAGUE_DATA_COPY_GET_PTR(deps->output[k].data.data), tag+k ));
#endif
        /* If this is the only call then force the remote deps propagation */
        deps = remote_dep_release_incoming(eu_context, deps, complete_mask);
    }

    /* Store the request in the rdv queue if any unsatisfied dep exist at this point */
    if(NULL != deps) {
        assert(0 != deps->incoming_mask);
        assert(0 != deps->msg.output_mask);
        dague_ulist_push_sorted(&dep_activates_fifo, (dague_list_item_t*)deps, rdep_prio);
    }

    /* Check if we have some pending get orders */
    if((dague_comm_gets < dague_comm_gets_max) && !dague_ulist_is_empty(&dep_activates_fifo)) {
        deps = (dague_remote_deps_t*)dague_ulist_fifo_pop(&dep_activates_fifo);
        remote_dep_mpi_get_start(eu_context, deps);
    }
}

static int
remote_dep_mpi_save_activate_cb(dague_execution_unit_t* eu_context,
                                dague_comm_callback_t* cb,
                                MPI_Status* status)
{
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
#endif
    int position = 0, length;
    dague_remote_deps_t* deps = NULL;

    MPI_Get_count(status, MPI_PACKED, &length);
    while(position < length) {
        deps = remote_deps_allocate(&dague_remote_dep_context.freelist);
        MPI_Unpack(dep_activate_buff[cb->storage2], length, &position,
                   &deps->msg, dep_count, dep_dtt, dep_comm);
        deps->from = status->MPI_SOURCE;

        /* Retrieve the data arenas and update the msg.incoming_mask to reflect
         * the data we should be receiving from the predecessor.
         */
        if( -1 == remote_dep_get_datatypes(eu_context, deps) ) {
            /* the corresponding dague_handle doesn't exist, yet. Put it in unexpected */
            char* packed_buffer;
            DEBUG2(("MPI:\tFROM\t%d\tActivate NOOBJ\t% -8s\tk=%d\twith datakey %lx\tparams %lx\n",
                    deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                    cb->storage2, deps->msg.deps, deps->msg.output_mask));
            /* Copy the eager data to some temp storage */
            packed_buffer = malloc(deps->msg.length);
            memcpy(packed_buffer, dep_activate_buff[cb->storage2] + position, deps->msg.length);
            position += deps->msg.length;  /* move to the next order */
            deps->dague_handle = (dague_handle_t*)packed_buffer;  /* temporary storage */
            dague_ulist_fifo_push(&dep_activates_noobj_fifo, (dague_list_item_t*)deps);
            continue;
        }
        DEBUG(("MPI:\tFROM\t%d\tActivate\t% -8s\tk=%d\twith datakey %lx\tparams %lx\n",
               status->MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
               cb->storage2, deps->msg.deps, deps->msg.output_mask));
        /* Import the activation message and prepare for the reception */
        remote_dep_mpi_recv_activate(eu_context, deps, dep_activate_buff[cb->storage2],
                                     position + deps->msg.length, &position);
        assert( dague_param_enable_aggregate || (position == length));
    }
    assert(position == length);
    /* Let's re-enable the pending request in the same position */
    MPI_Start(&array_of_requests[cb->storage1]);
    return 0;
}

static void remote_dep_mpi_new_object( dague_execution_unit_t* eu_context,
                                       dep_cmd_item_t *item )
{
    dague_handle_t* obj = item->cmd.new_object.obj;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
#endif
    DAGUE_ULIST_ITERATOR(&dep_activates_noobj_fifo, item,
    ({
        dague_remote_deps_t* deps = (dague_remote_deps_t*)item;
        if( deps->msg.handle_id == obj->handle_id ) {
            char* buffer = (char*)deps->dague_handle;
            int rc, position = 0;
            rc = remote_dep_get_datatypes(eu_context, deps); assert( -1 != rc );
            DEBUG2(("MPI:\tFROM\t%d\tActivate NEWOBJ\t% -8s\twith datakey %lx\tparams %lx\n",
                    deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                    deps->msg.deps, deps->msg.output_mask));
            remote_dep_mpi_recv_activate(eu_context, deps, buffer, deps->msg.length, &position);
            (void)dague_ulist_remove(&dep_activates_noobj_fifo, item);
            free(buffer);
            (void)rc;
        }
    }));
}

static void remote_dep_mpi_get_start(dague_execution_unit_t* eu_context,
                                     dague_remote_deps_t* deps)
{
    remote_dep_wire_activate_t* task = &(deps->msg);
    int from = deps->from, k, count, nbdtt;
    remote_dep_wire_get_t msg;
    MPI_Datatype dtt;
    void* data;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN], type_name[MPI_MAX_OBJECT_NAME];
    int len;
    remote_dep_cmd_to_string(task, tmp, MAX_TASK_STRLEN);
#endif

    for(k = count = 0; deps->incoming_mask >> k; k++)
        if( ((1U<<k) & deps->incoming_mask) ) count++;
    if( (dague_comm_gets + count) > dague_comm_gets_max ) {
        assert(deps->msg.output_mask != 0);
        assert(deps->incoming_mask != 0);
        dague_ulist_push_front(&dep_activates_fifo, (dague_list_item_t*)deps);
        return;
    }
    (void)eu_context;
    DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, (void*)task, task);

    msg.output_mask = deps->incoming_mask;  /* Only get what I need */
    msg.deps        = task->deps;
    msg.tag         = task->tag;

    for(k = 0; deps->incoming_mask >> k; k++) {
        if( !((1U<<k) & deps->incoming_mask) ) continue;

#ifdef DAGUE_PROF_DRY_DEP
        (void)dtt; (void)nbdtt;
        /* Removing the corresponding bit prevent the sending of the GET_DATA request */
        remote_dep_mpi_get_end(eu_context, k, deps);
        deps->incoming_mask ^= (1U<<k);
#else
        dtt   = deps->output[k].data.layout;
        nbdtt = deps->output[k].data.count;
        dague_data_copy_t* data_copy = deps->output[k].data.data;
        assert(NULL == data_copy); /* we do not support in-place tiles now, make sure it doesn't happen yet */
        if(NULL == data_copy) {
            data = dague_arena_get(deps->output[k].data.arena, deps->output[k].data.count);
            DEBUG3(("MPI:\tMalloc new remote tile %p size %zu in %p[%d]\n", data,
                    deps->output[k].data.arena->elem_size * deps->output[k].data.count,
                    deps, k));
            assert(data != NULL);
            data_copy = dague_data_get_copy(data, 0);
            data_copy->coherency_state = DATA_COHERENCY_EXCLUSIVE;
            deps->output[k].data.data = data_copy;
        }
#  if DAGUE_DEBUG_VERBOSE != 0
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG2(("MPI:\tTO\t%d\tGet START\t% -8s\tk=%d\twith datakey %lx at %p type %s count %d displ %ld extent %d\t(tag=%d)\n",
                from, tmp, k, task->deps, DAGUE_DATA_COPY_GET_PTR(data), type_name, nbdtt,
                deps->output[k].data.displ, deps->output[k].data.arena->elem_size * nbdtt, msg.tag+k));
#  endif
        TAKE_TIME_WITH_INFO(MPIrcv_prof, MPI_Data_pldr_sk, k, from,
                            eu_context->virtual_process->dague_context->my_rank, deps->msg);
        DEBUG_MARK_DTA_MSG_START_RECV(from, data_copy, msg.tag+k);
        MPI_Irecv(DAGUE_DATA_COPY_GET_PTR(deps->output[k].data.data) + deps->output[k].data.displ, nbdtt,
                  dtt, from, msg.tag+k, dep_comm,
                  &array_of_requests[dague_comm_last_active_req]);
        dague_comm_callback_t* cb = &array_of_callbacks[dague_comm_last_active_req];
        cb->fct      = remote_dep_mpi_get_end_cb;
        cb->storage1 = (long)deps;
        cb->storage2 = k;
        dague_comm_last_active_req++;
        dague_comm_gets++;
        assert(dague_comm_last_active_req <= DEP_NB_REQ);
#endif
    }
    if(msg.output_mask) {
        TAKE_TIME_WITH_INFO(MPIctl_prof, MPI_Data_ctl_sk, get,
                            from, eu_context->virtual_process->dague_context->my_rank, (*task));
        DAGUE_STATACC_ACCUMULATE_MSG(counter_control_messages_sent, datakey_count, datakey_dtt);
        MPI_Send(&msg, datakey_count, datakey_dtt, from,
                 REMOTE_DEP_GET_DATA_TAG, dep_comm);
        TAKE_TIME(MPIctl_prof, MPI_Data_ctl_ek, get++);
        DEBUG_MARK_CTL_MSG_GET_SENT(from, (void*)&msg, &msg);
    }
}

static void remote_dep_mpi_get_end(dague_execution_unit_t* eu_context,
                                   int idx,
                                   dague_remote_deps_t* deps)
{
    /* The ref on the data will be released below */
    remote_dep_release_incoming(eu_context, deps, (1U<<idx));
}

static int
remote_dep_mpi_get_end_cb(dague_execution_unit_t* eu_context,
                          dague_comm_callback_t* cb,
                          MPI_Status* status)
{
    dague_remote_deps_t* deps = (dague_remote_deps_t*)cb->storage1;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
#endif

    DEBUG2(("MPI:\tFROM\t%d\tGet END  \t% -8s\tk=%d\twith datakey na        \tparams %lx\t(tag=%d)\n",
            status->MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
            (int)cb->storage2, deps->incoming_mask, status->MPI_TAG)); (void)status;
    DEBUG_MARK_DTA_MSG_END_RECV(status->MPI_TAG);
    TAKE_TIME(MPIrcv_prof, MPI_Data_pldr_ek, (int)cb->storage2);
    remote_dep_mpi_get_end(eu_context, (int)cb->storage2, deps);
    dague_comm_gets--;
    return 0;
}
