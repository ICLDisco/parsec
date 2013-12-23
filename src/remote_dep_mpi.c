/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* /!\  THIS FILE IS NOT INTENDED TO BE COMPILED ON ITS OWN
 *      It should be included from remote_dep.c if HAVE_MPI is defined
 */
#include "dague_config.h"

#include <mpi.h>
#include "profiling.h"
#include "list.h"

#define DAGUE_REMOTE_DEP_USE_THREADS

typedef struct dep_cmd_item_s dep_cmd_item_t;
typedef union dep_cmd_u dep_cmd_t;

static int remote_dep_mpi_init(dague_context_t* context);
static int remote_dep_mpi_fini(dague_context_t* context);
static int remote_dep_mpi_on(dague_context_t* context);
static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context);
static int remote_dep_get_datatypes(dague_remote_deps_t* origin);
static int remote_dep_release(dague_execution_unit_t* eu_context,
                              dague_remote_deps_t* origin,
                              remote_dep_datakey_t complete_mask);

static int remote_dep_nothread_send(dague_execution_unit_t* eu_context,
                                    dep_cmd_item_t **head_item);
static int remote_dep_nothread_memcpy(dague_execution_unit_t* eu_context,
                                      dep_cmd_item_t *item);

static int remote_dep_dequeue_send(int rank, dague_remote_deps_t* deps);
static int remote_dep_dequeue_new_object(dague_object_t* obj);
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

#define DEP_NB_CONCURENT 3
static int dague_mpi_activations = 1 * DEP_NB_CONCURENT;
static int dague_mpi_transfers  = 2 * DEP_NB_CONCURENT;
/**
 * Number of data movements to be extracted at each step. Bigger the number
 * larger the amount spent in ordering the tasks, but greater the potential
 * benefits of doing things in the right order.
 */
static int dague_param_nb_tasks_extracted = 20;
static int dague_param_enable_eager = DAGUE_DIST_EAGER_LIMIT;
static int dague_param_enable_aggregate = 1;

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
        dague_object_t       *obj;
    } new_object;
    struct {
        dague_object_t       *dague_object;
        dague_arena_chunk_t  *source;
        void                 *destination;
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

static void remote_dep_mpi_save_put( dague_execution_unit_t* eu_context, int i, MPI_Status* status );
static void remote_dep_mpi_put_start( dague_execution_unit_t* eu_context, dep_cmd_item_t* item, int i );
static void remote_dep_mpi_put_end( dague_execution_unit_t* eu_context, int i, int k, MPI_Status* status );
#if RDEP_MSG_SHORT_LIMIT != 0
static void remote_dep_mpi_put_short( dague_execution_unit_t* eu_context, remote_dep_wire_activate_t* msg );
#endif  /* RDEP_MSG_SHORT_LIMIT != 0 */
static void remote_dep_mpi_save_activate( dague_execution_unit_t* eu_context, int i, MPI_Status* status );
static void remote_dep_mpi_get_start( dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i );
static void remote_dep_mpi_get_end( dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i, int k );
static void remote_dep_mpi_new_object( dague_execution_unit_t* eu_context, dep_cmd_item_t *item );

#if DAGUE_DEBUG_VERBOSE != 0
static inline char*
remote_dep_cmd_to_string(remote_dep_wire_activate_t* origin,
                         char* str,
                         size_t len)
{
    dague_execution_context_t task;

    task.dague_object = dague_object_lookup(origin->object_id);
    task.function     = task.dague_object->functions_array[origin->function_id];
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
static dague_remote_deps_t** dep_pending_recv_array;
static dep_cmd_item_t** dep_pending_put_array;
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

    assert(NULL == dague_dequeue_pop_front(&dep_cmd_queue));
    dague_dequeue_destruct(&dep_cmd_queue);
    assert(NULL == dague_dequeue_pop_front(&dep_cmd_fifo));
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
        whatsup = remote_dep_dequeue_nothread_progress(context);
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

static int remote_dep_dequeue_send(int rank,
                                   dague_remote_deps_t* deps)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    DAGUE_LIST_ITEM_CONSTRUCT(item);
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
    (void)eu; (void)oldcontext; (void)dst_vpid;
    if( dst_rank == src_rank ) return DAGUE_ITERATE_CONTINUE;

    dague_release_dep_fct_arg_t *arg       = (dague_release_dep_fct_arg_t *)param;
    assert( arg->action_mask & DAGUE_ACTION_RECV_INIT_REMOTE_DEPS );
    struct remote_dep_output_param* output = &arg->deps->output[dep->dep_datatype_index];
    void* dataptr = is_read_only(oldcontext, dep);
    if(NULL == dataptr) {
        output->deps_mask &= ~(1 << dep->dep_index); /* unmark all data that are RO we already hold from previous tasks */
    } else {
        output->deps_mask |= (1 << dep->dep_index); /* mark all data that are not RO */
        dataptr = is_inplace(oldcontext, dep);  /* Can we do it inplace */
    }
    output->data     = *data;
    output->data.ptr = dataptr; /* if still NULL allocate it */
    if(newcontext->priority > output->priority) {
        output->priority = newcontext->priority;
        if(newcontext->priority > arg->deps->max_priority)
            arg->deps->max_priority = newcontext->priority;
    }
    arg->deps->activity_mask |= (1 << dep->dep_datatype_index);
    return DAGUE_ITERATE_STOP;
}

/**
 * Retrieve the datatypes involved in this communication. In addition the flag
 * DAGUE_ACTION_RECV_INIT_REMOTE_DEPS set the priority to the maximum priority
 * of all the children.
 */
static int remote_dep_get_datatypes(dague_remote_deps_t* origin)
{
    dague_execution_context_t task;
    uint32_t local_mask = 0;
    dague_release_dep_fct_arg_t arg;

    assert(NULL == origin->dague_object);
    origin->dague_object = dague_object_lookup( origin->msg.object_id );
    if( NULL == origin->dague_object )
        return -1; /* the dague object doesn't exist yet */
    origin->activity_mask = 0;
    task.dague_object = origin->dague_object;
    task.function     = task.dague_object->functions_array[origin->msg.function_id];
    task.priority     = 0;
    for(int i = 0; i < task.function->nb_locals; i++)
        task.locals[i] = origin->msg.locals[i];

    arg.output_usage = 0;
    arg.deps         = origin;
    arg.ready_lists  = NULL;  /* No new tasks here */

    /* We need to convert from a dep_datatype_index mask into a dep_index mask */
    for(int i = 0; NULL != task.function->out[i]; i++ ) {
        for(int j = 0; NULL != task.function->out[i]->dep_out[j]; j++ )
            if(origin->msg.output_mask & (1U << task.function->out[i]->dep_out[j]->dep_datatype_index))
                local_mask |= (1U << task.function->out[i]->dep_out[j]->dep_index);
        if( 0 == local_mask ) continue;  /* nothing to be done */
        arg.action_mask = DAGUE_ACTION_RECV_INIT_REMOTE_DEPS | local_mask;
        DEBUG3(("MPI:\tRetrieve datatype with mask 0x%x (remote_dep_get_datatypes)\n", local_mask));
        task.function->iterate_successors(NULL, &task,
                                          arg.action_mask,
                                          remote_dep_mpi_retrieve_datatype,
                                          &arg);
        local_mask = 0;
    }
    assert( origin->activity_mask == origin->msg.output_mask);
    return 0;
}

/**
 * Complete a remote task locally. Put the data in the correct location then
 * call the release_deps.
 */
static int remote_dep_release(dague_execution_unit_t* eu_context,
                              dague_remote_deps_t* origin,
                              remote_dep_datakey_t complete_mask)
{
    dague_execution_context_t task;
    const dague_flow_t* target;
    int ret, i, pidx = 0;
    uint32_t local_mask = 0;

    assert((origin->msg.output_mask & complete_mask) == complete_mask);
    task.dague_object = dague_object_lookup(origin->msg.object_id);
#if defined(DAGUE_DEBUG)
    task.priority = 0;
#endif
    assert(task.dague_object); /* Future: for composition, store this in a list
                                  to be considered upon creation of the object */
    task.function = task.dague_object->functions_array[origin->msg.function_id];
    for(i = 0; i < task.function->nb_locals;
        task.locals[i] = origin->msg.locals[i], i++);

    target = task.function->out[pidx];
    for(i = 0; complete_mask>>i; i++) {
        assert(i < MAX_PARAM_COUNT);
        if( !((1<<i) & complete_mask) ) continue;
        while( !((1<<i) & target->flow_mask) ) {
            target = task.function->out[++pidx];
            if(NULL == target)
                assert(0);
        }
        DEBUG3(("MPI:\tDATA %p(%s) released from %p[%d] flow idx %d\n",
                ADATA(origin->output[i].data.ptr), target->name, origin, i, target->flow_index));
        task.data[target->flow_index].data_repo = NULL;
        task.data[target->flow_index].data      = origin->output[i].data.ptr;
    }

    /* We need to convert from a dep_datatype_index mask into a dep_index mask */
    for(int i = 0; NULL != task.function->out[i]; i++ )
        for(int j = 0; NULL != task.function->out[i]->dep_out[j]; j++ )
            if(complete_mask & (1U << task.function->out[i]->dep_out[j]->dep_datatype_index))
                local_mask |= (1U << task.function->out[i]->dep_out[j]->dep_index);
    DEBUG3(("MPI:\tTranslate mask from 0x%lx to 0x%x (remote_dep_release)\n", complete_mask, local_mask));
    origin->activity_mask = 0;
    ret = task.function->release_deps(eu_context, &task,
#if defined(DAGUE_DIST_COLLECTIVES)
                                      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
#endif  /* defined(DAGUE_DIST_COLLECTIVES) */
                                      local_mask | DAGUE_ACTION_RELEASE_LOCAL_DEPS,
                                      origin);
    /**
     * Release the dependency owned by the communication engine for all data that has been
     * internally allocated by the engine.
     */
    for(i = 0; complete_mask>>i; i++) {
        assert(i < MAX_PARAM_COUNT);
        if( !((1<<i) & complete_mask) ) continue;
        if( NULL != origin->output[i].data.ptr )  /* don't release the CONTROLs */
            AUNREF(origin->output[i].data.ptr);
    }
    /* Update the mask of remaining dependencies to avoid releasing twice the same outputs */
    origin->msg.output_mask ^= complete_mask;
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

    dague_list_construct(&temp_list);
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
        dague_list_destruct(&temp_list);
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
    int rc = MPI_Sendrecv((char*)ADATA(cmd->memcpy.source) + cmd->memcpy.displ_s, cmd->memcpy.count, cmd->memcpy.datatype, 0, 0,
                          (char*)cmd->memcpy.destination + cmd->memcpy.displ_r, cmd->memcpy.count, cmd->memcpy.datatype, 0, 0,
                          MPI_COMM_SELF, MPI_STATUS_IGNORE);
    AUNREF(cmd->memcpy.source);
    remote_dep_dec_flying_messages(item->cmd.memcpy.dague_object,
                                   eu_context->virtual_process->dague_context);
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
        dague_object_t *__object = dague_object_lookup( (ctx).object_id ); \
        __exec_context.function = __object->functions_array[(ctx).function_id ]; \
        __exec_context.dague_object = __object;                         \
        memcpy(&__exec_context.locals, (ctx).locals, MAX_LOCAL_COUNT * sizeof(assignment_t)); \
        dague_snprintf_execution_context(__info.func, 16, &__exec_context); \
        __info.rank_src = (src);                                        \
        __info.rank_dst = (dst);                                        \
        DAGUE_PROFILING_TRACE((PROF), (KEY), (I), PROFILE_OBJECT_ID_NULL, &__info); \
    } while(0)

#define TAKE_TIME(PROF, KEY, I) DAGUE_PROFILING_TRACE((PROF), (KEY), (I), PROFILE_OBJECT_ID_NULL, NULL);
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
    dep_pending_put_array = (dep_cmd_item_t**)calloc(DEP_NB_CONCURENT, sizeof(dep_cmd_item_t*));
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
    free(dep_pending_put_array); dep_pending_put_array = NULL;
    free(dep_pending_recv_array); dep_pending_recv_array = NULL;

    free(dague_mpi_same_pos_items); dague_mpi_same_pos_items = NULL;
    dague_mpi_same_pos_items_size = 0;

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
static int remote_dep_mpi_pack_dep(int rank,
                                   dep_cmd_item_t* item,
                                   char* packed_buffer,
                                   int length,
                                   int* position)
{
    dague_remote_deps_t *deps = (dague_remote_deps_t*)item->cmd.activate.task.deps;
    remote_dep_wire_activate_t* msg = &deps->msg;
    int k, dsize, saved_position = *position, completed = 0;
    uint32_t rank_bank, rank_mask, embedded, expected = 0;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, 128);
#endif

    rank_bank = rank / (sizeof(uint32_t) * 8);
    rank_mask = 1 << (rank % (sizeof(uint32_t) * 8));

    MPI_Pack_size(dep_count, dep_dtt, dep_comm, &dsize);
    if( (length - (*position)) < dsize ) {  /* no room. bail out */
        DEBUG3(("Can't pack at %d/%d. Bail out!\n", *position, length));
        return 1;
    }
    /* Don't pack yet, we need to update the length field before packing */
    *position  += dsize;
    msg->output_mask = embedded = 0;  /* clean start */
    msg->length = 0;

    /* Treat for special cases: CTL, Eeager, etc... */
    for(k = 0; deps->activity_mask >> k; k++) {
        if( !((1U << k) & deps->activity_mask )) continue;
        if( !(deps->output[k].rank_bits[rank_bank] & rank_mask) ) continue;
        msg->output_mask |= (1<<k);

        /* Remove CTL from the message we expect to send */
#if defined(DAGUE_PROF_DRY_DEP)
        deps->output[k].data.arena = NULL; /* make all data a control */
#endif
        if(NULL == deps->output[k].data.arena) {
            DEBUG2((" CTL\t%s\tparam %d\tdemoted to be a control\n",
                    tmp, k));
            embedded |= (1<<k);
            completed++;
            continue;
        }
        assert(deps->output[k].data.count > 0);
        if(dague_param_enable_eager) {
            /* Embed data (up to eager size) with the activate msg */
            MPI_Pack_size(deps->output[k].data.count, deps->output[k].data.layout,
                          dep_comm, &dsize);
            if((length - (*position)) >= dsize) {
                DEBUG2((" EGR\t%s\tparam %d\teager piggyback in the activate message\n",
                        tmp, k));
                MPI_Pack((char*)ADATA(deps->output[k].data.ptr) + deps->output[k].data.displ,
                         deps->output[k].data.count, deps->output[k].data.layout,
                         packed_buffer, length, position, dep_comm);
                embedded |= (1<<k);
                completed++;
                msg->length += dsize;
                continue;  /* go to the next */
            }
            /* the data doesn't fit in the buffer. */
        }
        expected++;
        DEBUG2(("DATA\t%s\tparam %d\tdeps %p send on demand (increase deps counter by %d [%d])\n",
                tmp, k, deps, expected, deps->pending_ack));
    }
    if(expected)
        dague_atomic_add_32b(&deps->pending_ack, expected);  /* Keep track of the inflight data */
    DEBUG(("MPI:\tTO\t%d\tActivate\t% -8s\ti=na\n"
           "    \t\t\twith datakey %lx\tmask %lx\t(tag=%d) eager count %d length %d\n",
           rank, tmp, msg->deps, msg->output_mask, msg->tag, completed, msg->length));
    /* And now pack the updated message (msg->length and msg->output_mask)
     * itself. Pack the complete output_mask, and then update it to reflect what
     * is left to send. */
    MPI_Pack(msg, dep_count, dep_dtt, packed_buffer, length, &saved_position, dep_comm);
    msg->output_mask ^= embedded;  /* remove the packed ones */
    item->cmd.activate.task.output_mask = msg->output_mask;  /* save it for later */
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
    remote_dep_wire_activate_t* msg;
    dague_remote_deps_t *deps;
    dep_cmd_item_t *item = *head_item;
    dague_list_item_t* ring = NULL;
    char packed_buffer[DEP_EAGER_BUFFER_SIZE];
    int rank, position = 0;

    rank     = item->cmd.activate.peer;  /* this doesn't change */
  pack_more:
    item->cmd.activate.task.tag = next_tag(MAX_PARAM_COUNT); /* TODO: waste less tags to diminish
                                                              collision probability */
    deps     = (dague_remote_deps_t*)item->cmd.activate.task.deps;
    msg      = &deps->msg;
    msg->tag = item->cmd.activate.task.tag;

    dague_list_item_singleton((dague_list_item_t*)item);
    if( 0 == remote_dep_mpi_pack_dep(rank, item, packed_buffer,
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

    TAKE_TIME_WITH_INFO(MPIctl_prof, MPI_Activate_sk, act, eu_context->virtual_process->dague_context->my_rank, rank, *msg);
    DAGUE_STATACC_ACCUMULATE_MSG(counter_control_messages_sent, packed, MPI_PACKED);
    MPI_Send((void*)packed_buffer, position, MPI_PACKED, rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, act++);
    DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(rank, (void*)msg, *msg);

    do {
        item = (dep_cmd_item_t*)ring;
        ring = dague_list_item_ring_chop(ring);

#if RDEP_MSG_SHORT_LIMIT != 0
        if( 0 != item->cmd.activate.task.output_mask ) {
            deps = (dague_remote_deps_t*)item->cmd.activate.task.deps;
            msg  = &deps->msg;
            remote_dep_mpi_put_short(eu_context, item);
        }
#endif   /* RDEP_MSG_SHORT_LIMIT != 0 */

        remote_dep_complete_and_cleanup((dague_remote_deps_t**)&item->cmd.activate.task.deps, 1,
                                        eu_context->virtual_process->dague_context); /* we send the order */
        if( 0 == item->cmd.activate.task.output_mask ) {
            free(item);
        }
    } while( NULL != ring );
    return 0;
}

static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context)
{
#if DAGUE_DEBUG_VERBOSE != 0
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
                            status->MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), i, k,
                            deps->msg.output_mask, status->MPI_TAG));
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
        if( !(output_mask & (1<<k)) ) continue;

        if( NULL == deps->output[k].data.arena ) continue;
        size_t extent = deps->output[k].data.arena->elem_size * deps->output[k].data.count;

        if( (extent <= (RDEP_MSG_SHORT_LIMIT)) | (extent <= (RDEP_MSG_EAGER_LIMIT)) ) {
            DEBUG3(("MPI:\tPEER\tNA\t%5s MODE  k=%d\tsize=%d <= %d\t(tag=base+%d)\n",
                    (extent <= (RDEP_MSG_EAGER_LIMIT) ? "Eager" : "Short"),
                    k, extent, RDEP_MSG_SHORT_LIMIT, k));
            continue;
        }
        output_mask ^= (1<<k);
    }
    return output_mask;
}

#if RDEP_MSG_SHORT_LIMIT != 0
static void remote_dep_mpi_put_short(dague_execution_unit_t* eu_context,
                                     dep_cmd_item_t* item)
{
    dague_remote_deps_t* deps = (dague_remote_deps_t*)item->cmd.activate.task.deps;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN);
#endif

    item->cmd.activate.task.output_mask = remote_dep_mpi_short_which(deps, item->cmd.activate.task.output_mask);
    if( 0 == item->cmd.activate.task.output_mask ) continue;

    /* Check if we can process it right now */
    for(int i = 0; i < DEP_NB_CONCURENT; i++ ) {
        if( NULL == dep_pending_put_array[i] ) {
            remote_dep_mpi_put_start(eu_context, item, i);
            return;
        }
    }
    DEBUG3(("MPI: Put Short DELAYED for %s from %d tag %u which 0x%x (deps %p)\n",
            tmp, item->cmd.activate.peer, msg->tag, item->cmd.activate.task.output_mask, (void*)deps));

    dague_ulist_push_front(&dep_put_fifo, (dague_list_item_t*)item);
}
#endif  /* RDEP_MSG_SHORT_LIMIT != 0 */

static void remote_dep_mpi_save_put(dague_execution_unit_t* eu_context,
                                    int i,
                                    MPI_Status* status)
{
    dep_cmd_item_t* item;
    remote_dep_wire_get_t* task;
    dague_remote_deps_t *deps;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
#endif

    item = (dep_cmd_item_t*) malloc(sizeof(dep_cmd_item_t));
    DAGUE_LIST_ITEM_CONSTRUCT(item);
    item->action = 0 /* DEP_GET_DATA */;
    item->cmd.activate.peer = status->MPI_SOURCE;

    task = &(item->cmd.activate.task);
    memcpy(task, &dep_get_buff[i], sizeof(remote_dep_wire_get_t));
    deps = (dague_remote_deps_t*) (uintptr_t) task->deps;
    item->priority = deps->max_priority;

    dague_ulist_push_sorted(&dep_put_fifo, (dague_list_item_t*)item, dep_cmd_prio);
    /* Check if we can push any new puts */
    for( i = 0; i < DEP_NB_CONCURENT; i++ ) {
        if( NULL == dep_pending_put_array[i] ) {
            item = (dep_cmd_item_t*)dague_ulist_fifo_pop(&dep_put_fifo);
            remote_dep_mpi_put_start(eu_context, item, i );
            return;
        }
    }
    DEBUG3(("MPI: Put DELAYED for %s from %d tag %u which 0x%x (deps %p)\n",
            remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), item->cmd.activate.peer,
            task->tag, task->output_mask, (void*)deps));
}

static void
remote_dep_mpi_put_start(dague_execution_unit_t* eu_context,
                         dep_cmd_item_t* item, int i)
{
    remote_dep_wire_get_t* task = &(item->cmd.activate.task);
    dague_remote_deps_t* deps = (dague_remote_deps_t*) (uintptr_t) task->deps;
    int k, nbdtt, tag = task->tag;
    void* data;
    MPI_Datatype dtt;
#if DAGUE_DEBUG_VERBOSE >= 2
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif

    (void)eu_context;
    DEBUG_MARK_CTL_MSG_GET_RECV(item->peer, (void*)task, task);

    assert(task->output_mask);
    DEBUG3(("MPI:\tPUT mask=%lx deps 0x%lx\n", task->output_mask, task->deps));
    for(k = 0; task->output_mask>>k; k++) {
        assert(k < MAX_PARAM_COUNT);
        if(!((1<<k) & task->output_mask)) continue;

        DEBUG3(("MPI:\t[idx %d mask(0x%x / 0x%x)] %p, %p\n", k, (1<<k), task->output_mask,
                deps->output[k].data.ptr, ADATA(deps->output[k].data.ptr)));
        data = ADATA(deps->output[k].data.ptr);
        dtt = deps->output[k].data.layout;
        nbdtt = deps->output[k].data.count;
#if DAGUE_DEBUG_VERBOSE >= 2
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG2(("MPI:\tTO\t%d\tPut START\tunknown \tj=%d,k=%d\twith deps 0x%lx at %p type %s\t(tag=%d displ = %ld)\n",
               item->cmd.activate.peer, i, k, task->deps, data, type_name, tag+k, deps->output[k].data.displ));
#endif

        TAKE_TIME_WITH_INFO(MPIsnd_prof[i], MPI_Data_plds_sk, i,
                            eu_context->virtual_process->dague_context->my_rank, item->cmd.activate.peer, deps->msg);
#if !defined(DAGUE_PROF_DRY_DEP)
        MPI_Isend((char*)data + deps->output[k].data.displ, nbdtt, dtt,
                  item->cmd.activate.peer, tag + k, dep_comm, &dep_put_snd_req[i*MAX_PARAM_COUNT+k]);
#endif  /* defined() */
        DEBUG_MARK_DTA_MSG_START_SEND(item->peer, data, tag+k);
    }
    dep_pending_put_array[i] = item;
}

static void remote_dep_mpi_put_end(dague_execution_unit_t* eu_context,
                                   int i, int k,
                                   MPI_Status* status)
{
    dep_cmd_item_t* item = dep_pending_put_array[i];
    remote_dep_wire_get_t* task = &(item->cmd.activate.task);

    DEBUG2(("MPI:\tTO\tna\tPut END  \tunknown \tj=%d,k=%d\twith deps %p\tparams %lx\t(tag=%d) data ptr %p\n",
            i, k, (dague_remote_deps_t*)task->deps, task->output_mask, status->MPI_TAG,
            ((dague_remote_deps_t*)task->deps)->output[k].data.ptr)); (void)status;
    DEBUG_MARK_DTA_MSG_END_SEND(status->MPI_TAG);
    TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_ek, i);
    assert(task->output_mask & (1<<k));
    task->output_mask ^= (1<<k);
    remote_dep_complete_and_cleanup((dague_remote_deps_t**)&(task->deps),
                                    1, eu_context->virtual_process->dague_context);
    if( 0 == task->output_mask ) {
        dep_pending_put_array[i] = NULL;
        free(item);
        item = (dep_cmd_item_t*)dague_ulist_fifo_pop(&dep_put_fifo);
        if( NULL != item ) {
            remote_dep_mpi_put_start(eu_context, item, i );
        }
    }
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
    remote_dep_datakey_t short_which = remote_dep_mpi_short_which(deps, deps->msg.output_mask);
    remote_dep_datakey_t complete_mask = 0;
    int k, dsize, tag = (int)deps->msg.tag, orig_pos = *position;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
    remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN);
#endif

    DEBUG(("MPI:\tFROM\t%d\tActivate\t% -8s\n"
           "\twith datakey %lx\tparams %lx length %d (pack buf %d/%d)\n",
           deps->from, tmp, deps->msg.deps, deps->msg.output_mask,
           deps->msg.length, *position, length));
    for(k = 0; deps->msg.output_mask>>k; k++) {
        if(!(deps->msg.output_mask & (1<<k))) continue;
        /* Check for all CTL messages, that do not carry payload */
        if(NULL == deps->output[k].data.arena) {
            DEBUG2(("MPI:\tHERE\t%d\tGet NONE\t% -8s\ti=NA,k=%d\twith datakey %lx at <NA> type CONTROL extent 0\t(tag=%d)\n",
                    deps->from, tmp, k, deps->msg.deps, tag+k));
            deps->output[k].data.ptr = (void*)2; /* the first non zero even value */
            complete_mask |= 1<<k;
            continue;
        }

        if( dague_param_enable_eager && (length - (*position))) {
            /* Check if the data is EAGER embedded in the activate */
            MPI_Pack_size(deps->output[k].data.count, deps->output[k].data.layout,
                          dep_comm, &dsize);
            if((length - (*position)) >= dsize) {
                assert(NULL == deps->output[k].data.ptr); /* we do not support in-place tiles now, make sure it doesn't happen yet */
                if(NULL == deps->output[k].data.ptr) {
                    deps->output[k].data.ptr = dague_arena_get(deps->output[k].data.arena, deps->output[k].data.count);
                    DEBUG3(("MPI:\tMalloc new remote tile %p size %" PRIu64 " count = %d displ = %" PRIi64 "\n",
                            deps->output[k].data.ptr, deps->output[k].data.arena->elem_size,
                            deps->output[k].data.count, deps->output[k].data.displ));
                    assert(deps->output[k].data.ptr != NULL);
                }
#ifndef DAGUE_PROF_DRY_DEP
                DEBUG2((" EGR\t%s\tparam %d\teager from the activate message\n",
                        tmp, k));
                MPI_Unpack(packed_buffer, length, position,
                           (char*)ADATA(deps->output[k].data.ptr) + deps->output[k].data.displ,
                           deps->output[k].data.count, deps->output[k].data.layout, dep_comm);
#endif
                complete_mask |= 1<<k;
                continue;
            }
        }
        /* Check if we have SHORT deps to satisfy quickly */
        if( short_which & (1<<k) ) {

            assert(NULL == deps->output[k].data.ptr); /* we do not support in-place tiles now, make sure it doesn't happen yet */
            if(NULL == deps->output[k].data.ptr) {
                deps->output[k].data.ptr = dague_arena_get(deps->output[k].data.arena,
                                                           deps->output[k].data.count);
                DEBUG3(("MPI:\tMalloc new remote tile %p size %" PRIu64 " count = %d displ = %" PRIi64 "\n",
                        deps->output[k].data.ptr, deps->output[k].data.arena->elem_size,
                        deps->output[k].data.count, deps->output[k].data.displ));
                assert(deps->output[k].data.ptr != NULL);
            }
            DEBUG2(("MPI:\tFROM\t%d\tGet SHORT\t% -8s\ti=NA,k=%d\twith datakey %lx at %p\t(tag=%d)\n",
                    deps->from, tmp, k, deps->msg.deps, ADATA(deps->output[k].data.ptr), tag+k));
#ifndef DAGUE_PROF_DRY_DEP
            MPI_Request req; int flag = 0;
            MPI_Irecv((char*)ADATA(deps->output[k].data.ptr) + deps->output[k].data.displ,
                      deps->output[k].data.count, deps->output[k].data.layout,
                      deps->from, tag+k, dep_comm, &req);
            do {
                MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
                if(flag) break;
                remote_dep_mpi_progress(eu_context);
            } while(!flag);
#endif
            complete_mask |= 1<<k;
            continue;
        }
        DEBUG2(("MPI:\tFROM\t%d\tGet DATA\t% -8s\ti=NA,k=%d\twith datakey %lx tag=%d (to be posted)\n",
                deps->from, tmp, k, deps->msg.deps, tag+k));
    }
    assert(deps->msg.length == (uint32_t)((*position) - orig_pos));

    /* Release all the already satisfied deps without posting the RDV */
    if(complete_mask) {
#if DAGUE_DEBUG_VERBOSE >= 2
        for(int k = 0; complete_mask>>k; k++)
            if((1<<k) & complete_mask)
                DEBUG2(("MPI:\tHERE\t%d\tGet PREEND\t% -8s\ti=NA,k=%d\twith datakey %lx at %p ALREADY SATISFIED\t(tag=%d)\n",
                        deps->from, tmp, k, deps->msg.deps, ADATA(deps->output[k].data.ptr), tag+k ));
#endif
        remote_dep_release(eu_context, deps, complete_mask);
    }

    /* Store the request in the rdv queue if any unsatisfied dep exist at this
     * point */
    if(deps->msg.output_mask) {
        dague_ulist_push_sorted(&dep_activates_fifo, (dague_list_item_t*)deps, rdep_prio);
    } else {
        remote_deps_free(deps);
    }

    /* Check if we have some ordered rdv get to treat */
    for(int i = 0; i < DEP_NB_CONCURENT; i++ ) {
        if( NULL == dep_pending_recv_array[i] ) {
            deps = (dague_remote_deps_t*)dague_ulist_fifo_pop(&dep_activates_fifo);
            if(deps) remote_dep_mpi_get_start(eu_context, deps, i);
            break;
        }
    }
}

static void
remote_dep_mpi_save_activate(dague_execution_unit_t* eu_context,
                             int i,
                             MPI_Status* status )
{
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
#endif
    int position = 0, length;
    dague_remote_deps_t* deps = NULL;

    MPI_Get_count(status, MPI_PACKED, &length);
    while(position < length) {
        deps = remote_deps_allocate(&dague_remote_dep_context.freelist);
        MPI_Unpack(dep_activate_buff[i], length, &position,
                   &deps->msg, dep_count, dep_dtt, dep_comm);
        deps->from = status->MPI_SOURCE;
        DEBUG(("MPI:\tFROM\t%d\tActivate\t% -8s\ti=%d\twith datakey %lx\tparams %lx\n",
               status->MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
               i, deps->msg.deps, deps->msg.output_mask));

        /* Retrieve the data arenas and update the msg.output_mask to reflect
         * all the data we should be receiving from the predecessor.
         */
        if( -1 == remote_dep_get_datatypes(deps) ) {
            /* the corresponding dague_object doesn't exist, yet. Put it in unexpected */
            char* packed_buffer;
            DEBUG2(("MPI:\tFROM\t%d\tActivate NOOBJ\t% -8s\ti=%d\twith datakey %lx\tparams %lx\n",
                    deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                    i, deps->msg.deps, deps->msg.output_mask));
            /* Copy the eager data to some temp storage */
            packed_buffer = malloc(deps->msg.length);
            memcpy(packed_buffer, dep_activate_buff[i] + position, deps->msg.length);
            position += deps->msg.length;  /* move to the next order */
            deps->dague_object = (struct dague_object*)packed_buffer;  /* temporary storage */
            dague_ulist_fifo_push(&dep_activates_noobj_fifo, (dague_list_item_t*)deps);
            continue;
        }
        /* Import the activation message and prepare for the reception of all
         * included data.
         */
        remote_dep_mpi_recv_activate(eu_context, deps, dep_activate_buff[i],
                                     position + deps->msg.length, &position);
        assert( dague_param_enable_aggregate || (position == length));
    }
    assert(position == length);
}

static void remote_dep_mpi_new_object( dague_execution_unit_t* eu_context,
                                       dep_cmd_item_t *item )
{
    dague_object_t* obj = item->cmd.new_object.obj;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN];
#endif
    DAGUE_ULIST_ITERATOR(&dep_activates_noobj_fifo, item,
    ({
        dague_remote_deps_t* deps = (dague_remote_deps_t*)item;
        if( deps->msg.object_id == obj->object_id ) {
            char* buffer = (char*)deps->dague_object;
            int rc, position = 0;
            rc = remote_dep_get_datatypes(deps); assert( -1 != rc );
            DEBUG2(("MPI:\tFROM\t%d\tActivate NEWOBJ\t% -8s\ti=NA\twith datakey %lx\tparams %lx\n",
                    deps->from, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                    deps->msg.deps, deps->msg.output_mask));
            remote_dep_mpi_recv_activate(eu_context, deps, buffer, deps->msg.length, &position);
            (void)dague_ulist_remove(&dep_activates_noobj_fifo, item);
            free(buffer);
            remote_deps_free(deps);
            (void)rc;
        }
    }));
}

static void remote_dep_mpi_get_start(dague_execution_unit_t* eu_context,
                                     dague_remote_deps_t* deps,
                                     int i)
{
    MPI_Datatype dtt;
    int nbdtt;
    remote_dep_wire_get_t msg;
    remote_dep_wire_activate_t* task = &(deps->msg);
    int from = deps->from;
    void* data;
#if DAGUE_DEBUG_VERBOSE != 0
    char tmp[MAX_TASK_STRLEN], type_name[MPI_MAX_OBJECT_NAME];
    int len;
    remote_dep_cmd_to_string(task, tmp, MAX_TASK_STRLEN);
#endif

    (void)eu_context;
    DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, (void*)task, task);

    msg.output_mask = task->output_mask;
    msg.deps        = task->deps;
    msg.tag         = task->tag;

    for(int k = 0; msg.output_mask >> k; k++) {
        if( !((1<<k) & msg.output_mask) ) continue;
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
        (void)dtt; (void)nbdtt; (void)dep_put_rcv_req;
        /* Removing the corresponding bit prevent the sending of the GET_DATA request */
        msg.output_mask &= ~(1<<k);
        remote_dep_mpi_get_end(eu_context, deps, i, k);
#else
#  if DAGUE_DEBUG_VERBOSE != 0
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG2(("MPI:\tTO\t%d\tGet START\t% -8s\ti=%d,k=%d\twith datakey %lx at %p type %s count %d displ %ld extent %d\t(tag=%d)\n",
                from, tmp, i, k, task->deps, ADATA(data), type_name, nbdtt,
                deps->output[k].data.displ, deps->output[k].data.arena->elem_size * nbdtt, msg.tag+k));
#  endif
        TAKE_TIME_WITH_INFO(MPIrcv_prof[i], MPI_Data_pldr_sk, i+k, from,
                            eu_context->virtual_process->dague_context->my_rank, deps->msg);
        DEBUG_MARK_DTA_MSG_START_RECV(from, data, msg.tag+k);
        MPI_Irecv((char*)ADATA(data) + deps->output[k].data.displ, nbdtt,
                  dtt, from, msg.tag+k, dep_comm,
                  &dep_put_rcv_req[i*MAX_PARAM_COUNT+k]);
#endif
    }
    if(msg.output_mask) {
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
}

static void remote_dep_mpi_get_end(dague_execution_unit_t* eu_context,
                                   dague_remote_deps_t* deps,
                                   int i, int k)
{
    /* No need to release the ref on the data it will be done in the remote_dep_release */
    remote_dep_release(eu_context, deps, (1 << k));
    if(0 == deps->msg.output_mask) {
        remote_deps_free(deps);
        dep_pending_recv_array[i] = NULL;
        if( !dague_ulist_is_empty(&dep_activates_fifo) ) {
            deps = (dague_remote_deps_t*)dague_ulist_fifo_pop(&dep_activates_fifo);
            if( NULL != deps ) {
                remote_dep_mpi_get_start(eu_context, deps, i);
            }
        }
    }
}
