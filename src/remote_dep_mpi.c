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
#include "fifo.h"

#define DAGUE_REMOTE_DEP_USE_THREADS

static int remote_dep_mpi_init(dague_context_t* context);
static int remote_dep_mpi_fini(dague_context_t* context);
static int remote_dep_mpi_on(dague_context_t* context);
static int remote_dep_mpi_send_dep(dague_execution_unit_t* eu_context, int rank, remote_dep_wire_activate_t* msg);
static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context);
static int remote_dep_get_datatypes(dague_remote_deps_t* origin);
static int remote_dep_release(dague_execution_unit_t* eu_context, dague_remote_deps_t* origin);

static int remote_dep_nothread_send(dague_execution_unit_t* eu_context, int rank, dague_remote_deps_t* deps);
static int remote_dep_nothread_memcpy(void *dst, void *src, const dague_remote_dep_datatype_t datatype);

static int remote_dep_dequeue_send(int rank, dague_remote_deps_t* deps);
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
#   define remote_dep_send(rank, deps) remote_dep_dequeue_send(rank, deps)
#   define remote_dep_progress(ctx) ((void)ctx,0) 

#else
static int remote_dep_dequeue_nothread_init(dague_context_t* context);
static int remote_dep_dequeue_nothread_fini(dague_context_t* context);
#   define remote_dep_init(ctx) remote_dep_dequeue_nothread_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_nothread_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_mpi_on(ctx)
#   define remote_dep_off(ctx)  0
#   define remote_dep_send(rank, deps) remote_dep_dequeue_send(rank, deps)
#   define remote_dep_progress(ctx) remote_dep_dequeue_nothread_progress(ctx)
#endif 
static int remote_dep_dequeue_nothread_progress(dague_execution_unit_t* eu_context);

#include "dequeue.h"

#define DEP_NB_CONCURENT 3
static int dague_mpi_activations = 1 * DEP_NB_CONCURENT;
static int dague_mpi_transferts  = 2 * DEP_NB_CONCURENT;

typedef enum dep_cmd_action_t
{
    DEP_ACTIVATE,
    DEP_RELEASE,
/*    DEP_PROGRESS,
    DEP_PUT_DATA,
    DEP_GET_DATA,*/
    DEP_CTL,
    DEP_MEMCPY,
} dep_cmd_action_t;

typedef union dep_cmd_t
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
        dague_object_t*             dague_object;
        dague_arena_chunk_t*        source;
        void*                       destination;
        dague_remote_dep_datatype_t datatype;
    } memcpy;
} dep_cmd_t;

typedef struct dep_cmd_item_t
{
    dague_list_item_t super;
    dep_cmd_action_t  action;
    int               priority;
    dep_cmd_t         cmd;
} dep_cmd_item_t;

typedef struct dague_dep_wire_get_fifo_elem_t {
    dague_list_item_t           item;
    remote_dep_wire_get_t       task;
    int                         priority;
    int                         peer;
} dague_dep_wire_get_fifo_elem_t;

static void remote_dep_mpi_save_put( dague_execution_unit_t* eu_context, int i, MPI_Status* status );
static void remote_dep_mpi_put_start(dague_execution_unit_t* eu_context, dague_dep_wire_get_fifo_elem_t* item, int i);
static void remote_dep_mpi_put_end(dague_execution_unit_t* eu_context, int i, int k, MPI_Status* status);
static void remote_dep_mpi_put_eager(dague_execution_unit_t* eu_context, int rank, remote_dep_wire_activate_t* msg);
static void remote_dep_mpi_save_activation( dague_execution_unit_t* eu_context, int i, MPI_Status* status );
static void remote_dep_mpi_get_start(dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i );
static void remote_dep_mpi_get_end(dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i, int k);

#ifdef DAGUE_DEBUG
static char* remote_dep_cmd_to_string(remote_dep_wire_activate_t* origin, char* str, size_t len)
{
    unsigned int i, index = 0;
    dague_object_t* object;
    const dague_function_t* function;
    
    object = dague_object_lookup( origin->object_id );
    function = object->functions_array[origin->function_id];

    index += snprintf( str + index, len - index, "%s", function->name );
    if( index >= len ) return str;
    for( i = 0; i < function->nb_definitions; i++ ) {
        index += snprintf( str + index, len - index, "_%d",
                           origin->locals[i].value );
        if( index >= len ) return str;
    }
    return str;
}
#endif

static pthread_t dep_thread_id;
dague_dequeue_t dep_cmd_queue;
dague_list_t    dep_cmd_fifo;            /* ordered non threaded fifo */
dague_list_t    dague_activations_fifo;  /* ordered non threaded fifo */
dague_list_t    dague_put_fifo;          /* ordered non threaded fifo */
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

    assert(mpi_initialized == 0);

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
    dague_list_construct(&dague_activations_fifo);
    dague_list_construct(&dague_put_fifo);

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
        dague_context_t *ret;

        item->action = DEP_CTL;
        item->cmd.ctl.enable = -1;  /* turn off the MPI thread */
        item->priority = 0;
        DAGUE_LIST_ITEM_SINGLETON(item);
        dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);

        /* I am supposed to own the lock. Wake the MPI thread */
        pthread_cond_signal(&mpi_thread_condition);
        pthread_mutex_unlock(&mpi_thread_mutex);
        pthread_join(dep_thread_id, (void**) &ret);
        assert(ret == context);
    }

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
    item->action = DEP_CTL;
    item->cmd.ctl.enable = 0;  /* turn OFF the MPI thread */
    item->priority = 0;
    DAGUE_LIST_ITEM_SINGLETON(item);
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
    int ctl = -1, whatsup;

    ctl = dague_bindthread(context->nb_cores);
    if(ctl != context->nb_cores) do_nano = 1;
    else fprintf(stderr, "DAGuE\tMPI bound to physical core %d\n", ctl);

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
        whatsup = remote_dep_dequeue_nothread_progress(context->execution_units[0]);
    } while(-1 != whatsup);
    /* Release all resources */
    remote_dep_mpi_fini(context);
    pthread_exit((void*)context);
}


static int remote_dep_dequeue_send(int rank, dague_remote_deps_t* deps)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    item->action = DEP_ACTIVATE;
    item->cmd.activate.rank = rank;
    item->cmd.activate.deps = deps;
    item->priority = deps->max_priority;
    DAGUE_LIST_ITEM_SINGLETON(item);
    dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);
    return 1;
}

void dague_remote_dep_memcpy(dague_execution_unit_t* eu_context,
                             dague_object_t* dague_object,
                             void *dst,
                             dague_arena_chunk_t *src,
                             dague_remote_dep_datatype_t datatype)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    item->action = DEP_MEMCPY;
    item->cmd.memcpy.dague_object = dague_object;
    item->cmd.memcpy.source       = src;
    item->cmd.memcpy.destination  = dst;
    item->cmd.memcpy.datatype     = datatype;
    AREF(src);
    remote_dep_inc_flying_messages(dague_object, eu_context->master_context);
    item->priority = 0;
    DAGUE_LIST_ITEM_SINGLETON(item);
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
    assert(exec_context.dague_object); /* Future: for composition, store this in a list to be considered upon creation of the DO*/
    assert( NULL == origin->dague_object );
    origin->dague_object = exec_context.dague_object;
    exec_context.function = exec_context.dague_object->functions_array[origin->msg.function_id];

    for(int i = 0; i < exec_context.function->nb_definitions; i++)
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
    assert(exec_context.dague_object); /* Future: for composition, store this in a list to be considered upon creation of the DO*/
    exec_context.function = exec_context.dague_object->functions_array[origin->msg.function_id];
    for( i = 0; i < exec_context.function->nb_definitions; i++)
        exec_context.locals[i] = origin->msg.locals[i];

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != (target = exec_context.function->out[i])); i++) {
        whereto = target->flow_index;
        exec_context.data[whereto].data_repo = NULL;
        exec_context.data[whereto].data      = NULL;
        if(origin->msg.deps & (1 << i)) {
            //DEBUG(("MPI:\tDATA %p released from %p[%d]\n", GC_DATA(origin->output[i].data), origin, i));
            exec_context.data[whereto].data = origin->output[i].data;
#ifdef DAGUE_DEBUG
/*            {
                char tmp[128];
                void* _data = ADATA(exec_context.data[whereto].data);
                DEBUG((MPI:\t"%s: recv %p -> [0] %9.5f [1] %9.5f [2] %9.5f\n",
                       dague_service_to_string(&exec_context, tmp, 128),
                       _data, ((double*)_data)[0], ((double*)_data)[1], ((double*)_data)[2]));
            }*/
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

static inline dague_list_item_t* rdep_fifo_push_ordered_cmd( dague_list_t* fifo,
                                                             dague_list_item_t* elem )
{
    dague_list_item_t* position;
    dep_cmd_item_t* ec;
    dep_cmd_item_t* input = (dep_cmd_item_t*)elem;

    position = DAGUE_ULIST_ITERATOR(fifo, item, 
    {
        ec = (dep_cmd_item_t*)item;
        if( ec->priority < input->priority )
            break;
    });
    dague_ulist_add(fifo, position, elem);
    return elem;
}

#ifndef DAGUE_REMOTE_DEP_USE_THREADS
static int remote_dep_dequeue_nothread_init(dague_context_t* context)
{
    dague_dequeue_construct(&dep_cmd_queue);
    dague_list_construct(&dep_cmd_fifo);
    dague_list_construct(&dague_activations_fifo);
    dague_list_construct(&dague_put_fifo);

    return remote_dep_mpi_init(context);
}

static int remote_dep_dequeue_nothread_fini(dague_context_t* context)
{
    remote_dep_mpi_fini(context);
    dague_list_destruct(&dep_cmd_fifo);
    dague_list_destruct(&dague_activations_fifo);
    dague_list_destruct(&dague_put_fifo);
    
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
    while( NULL != (item = (dep_cmd_item_t*) dague_dequeue_pop_front(&dep_cmd_queue)) ) {
        if( DEP_CTL == item->action ) {
            /* A DEP_CTL is a barrier that must not be crossed, flush the
             * ordered fifo and don't add anything until it is consumed */
            if( !dague_ulist_is_empty(&dep_cmd_fifo) ) {
                DAGUE_LIST_ITEM_SINGLETON(item);
                dague_dequeue_push_front(&dep_cmd_queue, (dague_list_item_t*)item);
                break;
            } else goto handle_now;
        }
        rdep_fifo_push_ordered_cmd(&dep_cmd_fifo, (dague_list_item_t*)item);
    }
    item = (dep_cmd_item_t*)dague_ufifo_pop(&dep_cmd_fifo);

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
    case DEP_ACTIVATE:
        remote_dep_nothread_send(eu_context, item->cmd.activate.rank, item->cmd.activate.deps);
        break;
    case DEP_CTL:
        ret = item->cmd.ctl.enable;
        free(item);
        return ret;  /* FINI or OFF */
    case DEP_MEMCPY:
        remote_dep_nothread_memcpy(item->cmd.memcpy.destination, 
                                   item->cmd.memcpy.source,
                                   item->cmd.memcpy.datatype);
        remote_dep_dec_flying_messages(item->cmd.memcpy.dague_object, eu_context->master_context);
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
    msg.which = RDEP_MSG_EAGER(&deps->msg);
    for( k = 0; output_count; k++ ) {
        output_count -= deps->output[k].count;
        if(deps->output[k].rank_bits[rank_bank] & rank_mask) {
#if defined(DAGUE_PROF_DRY_DEP)
            deps->output[k].type = NULL; /* make all data a control */
#endif
            msg.which |= (1<<k);
        }
    }
    remote_dep_mpi_send_dep(eu_context, rank, &msg);
    return 0;
}

static int remote_dep_nothread_memcpy(void *dst, void *src, 
                                      const dague_remote_dep_datatype_t datatype)
{
    /* TODO: split the mpi part */
    int rc = MPI_Sendrecv(ADATA(src), 1, datatype, 0, 0,
                          dst, 1, datatype, 0, 0,
                          MPI_COMM_SELF, MPI_STATUS_IGNORE);
    AUNREF(src);
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
static dague_thread_profiling_t* MPIsnd_eager;
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

static int  dague_profile_remote_dep_mpi_info_to_string(void *info, char *text, size_t size)
{
    int res;
    dague_profile_remote_dep_mpi_info_t nfo = *(dague_profile_remote_dep_mpi_info_t*)info;
    res = snprintf(text, size, "%d -> %d: %s", nfo.rank_src, nfo.rank_dst, nfo.func);
    return res;
}

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
    MPIsnd_eager = dague_profiling_thread_init( 2*1024*1024, "MPI send()");
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
        memcpy(&__exec_context.locals, ctx.locals, MAX_LOCAL_COUNT * sizeof(assignment_t)); \
        dague_service_to_string( &__exec_context, __info.func, 16 );    \
        __info.rank_src = src;                                          \
        __info.rank_dst = dst;                                          \
        dague_profiling_trace((PROF), (KEY), (I), &__info);             \
    } while(0)

#define TAKE_TIME(PROF, KEY, I) dague_profiling_trace((PROF), (KEY), (I), NULL);
#else
#define TAKE_TIME_WITH_INFO(PROF, KEY, I, src, dst, ctx) do {} while(0)
#define TAKE_TIME(PROF, KEY, I) do {} while(0)
#define remote_dep_mpi_profiling_init() do {} while(0)
#endif  /* DAGUE_PROF_TRACE */

/* TODO: smart use of dague context instead of ugly globals */

static int active_eagers = 0;
static MPI_Comm dep_comm;
#define DEP_NB_REQ (2 * DEP_NB_CONCURENT + 3 * (DEP_NB_CONCURENT * MAX_PARAM_COUNT))
static MPI_Request  array_of_requests[DEP_NB_REQ];
static int          array_of_indices[DEP_NB_REQ];
static MPI_Status   array_of_statuses[DEP_NB_REQ];
static MPI_Request* dep_activate_req    = &array_of_requests[0 * DEP_NB_CONCURENT];
static MPI_Request* dep_get_req         = &array_of_requests[1 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_snd_req     = &array_of_requests[2 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_rcv_req     = &array_of_requests[2 * DEP_NB_CONCURENT + DEP_NB_CONCURENT * MAX_PARAM_COUNT];
static MPI_Request* dep_put_eager_req   = &array_of_requests[2 * DEP_NB_CONCURENT + 2 * DEP_NB_CONCURENT * MAX_PARAM_COUNT];

/* TODO: fix heterogeneous restriction by using proper mpi datatypes */
#define dep_dtt MPI_BYTE
#define dep_count sizeof(remote_dep_wire_activate_t)
static dague_remote_deps_t* dep_activate_buff[DEP_NB_CONCURENT];
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
static int MAX_MPI_TAG;
static int NEXT_TAG = REMOTE_DEP_MAX_CTRL_TAG+1;
#if 1
#define INC_NEXT_TAG(k) do { \
    assert(k < MAX_MPI_TAG); \
    if(NEXT_TAG < (MAX_MPI_TAG - k)) \
        NEXT_TAG += k; \
    else \
        NEXT_TAG = REMOTE_DEP_MAX_CTRL_TAG + k + 1; \
} while(0)
#else
#define INC_NEXT_TAG(k) do {} while(0)
#endif

static int remote_dep_mpi_init(dague_context_t* context)
{
    int i, mpi_tag_ub_exists, *ub;

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
        fprintf(stderr, "Your MPI implementation does not define MPI_TAG_UB and thus violates the standard (MPI-2.2, page 29, line 30).\n");
    } else {
        MAX_MPI_TAG = *ub;
#if defined( DAGUE_DEBUG )
        if( MAX_MPI_TAG < INT_MAX ) {
            DEBUG(("MPI:\tYour MPI implementation defines the maximal TAG value to %d (0x%08x), which might be too small should you have more than %d simultaneous remote dependencies\n",
                    MAX_MPI_TAG, (unsigned int)MAX_MPI_TAG, MAX_MPI_TAG / MAX_PARAM_COUNT));
        }
#endif
    }

    MPI_Comm_size(dep_comm, &(context->nb_nodes));
    MPI_Comm_rank(dep_comm, &(context->my_rank));
    for(i = 0; i < DEP_NB_REQ; i++) {        
        array_of_requests[i] = MPI_REQUEST_NULL;
    }

    /* Create the remote dependencies activation buffers */
    for(i = 0; i < DEP_NB_CONCURENT; i++) {
        dep_activate_buff[i] = remote_deps_allocation(&dague_remote_dep_context.freelist);
    }

    /* Create all the pending receives and start them */
    for(i = 0; i < DEP_NB_CONCURENT; i++) {
        MPI_Recv_init(&dep_activate_buff[i]->msg, dep_count, dep_dtt, MPI_ANY_SOURCE, REMOTE_DEP_ACTIVATE_TAG, dep_comm, &dep_activate_req[i]);
        MPI_Recv_init(&dep_get_buff[i], datakey_count, datakey_dtt, MPI_ANY_SOURCE, REMOTE_DEP_GET_DATA_TAG, dep_comm, &dep_get_req[i]);
        MPI_Start(&dep_activate_req[i]);
        MPI_Start(&dep_get_req[i]);
    }
    
    dep_pending_recv_array = (dague_remote_deps_t**)calloc(DEP_NB_CONCURENT,sizeof(dague_remote_deps_t*));
    dep_pending_put_array = (dague_dep_wire_get_fifo_elem_t**)calloc(DEP_NB_CONCURENT,sizeof(dague_dep_wire_get_fifo_elem_t*));

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
        DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)dep_activate_buff[i]);
        dague_atomic_lifo_push(&dague_remote_dep_context.freelist, (dague_list_item_t*)dep_activate_buff[i]);
    }
    for(i = 0; i < DEP_NB_REQ; i++) {
        assert(MPI_REQUEST_NULL == array_of_requests[i]);
    }
    free( dep_pending_put_array );
    free( dep_pending_recv_array );

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
#ifdef DAGUE_DEBUG
    char tmp[128];
#endif

#if !defined(DAGUE_PROF_TRACE)
    (void)eu_context;
#endif
    
    DEBUG(("MPI:\tTO\t%d\tActivate\t% -8s\ti=na\twith datakey %lx\tmask %lx\n", rank, remote_dep_cmd_to_string(msg, tmp, 128), msg->deps, msg->which));
    
    if(active_eagers == DEP_NB_CONCURENT)
    {
        RDEP_MSG_EAGER_CLR(msg);
    }

    TAKE_TIME_WITH_INFO(MPIctl_prof, MPI_Activate_sk, act, eu_context->master_context->my_rank, rank, (*msg));
    MPI_Send((void*) msg, dep_count, dep_dtt, rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, act++);

    DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(rank, (void*)msg, msg);

#if defined(DAGUE_STATS)
    {
        MPI_Aint _lb, _size;
        MPI_Type_get_extent(dep_dtt, &_lb, &_size);
        DAGUE_STATACC_ACCUMULATE(counter_control_messages_sent, 1);
        DAGUE_STATACC_ACCUMULATE(counter_bytes_sent, _size * dep_count);
    }
#endif
    
    /* Do not wait for completion of CTL */
    for(int k=0; msg->which>>k; k++) {
        if(0 == (msg->which & (1<<k))) continue;
        dague_remote_deps_t* deps = (dague_remote_deps_t*) msg->deps;
        if(NULL != deps->output[k].type) continue;

        DEBUG((" CTL\t%s\tparam%d\tdemoted to be a control\n",remote_dep_cmd_to_string(&deps->msg, tmp, 128), k));

        msg->which ^= (1<<k);
        if(0 == msg->which) {
            remote_dep_dec_flying_messages(((dague_remote_deps_t*)msg->deps)->dague_object, eu_context->master_context);
        }
        remote_dep_complete_one_and_cleanup(deps);
    }
    
    /* Proceed with Eager mode now */
    if(RDEP_MSG_EAGER(msg))
    {
        dague_object_t* obj = ((dague_remote_deps_t*)msg->deps)->dague_object;
        remote_dep_mpi_put_eager(eu_context, rank, msg); 
        /* In eager mode, everything is eager, so we are done, now */
        remote_dep_dec_flying_messages(obj, eu_context->master_context);
    }
    
    return 1;
}


static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context)
{
#ifdef DAGUE_DEBUG
    char tmp[128];
#endif
    MPI_Status *status;
    int ret = 0;
    int index, i, k, outcount;
    
    if(eu_context->eu_id != 0) return 0;
    
    do {
        MPI_Testsome(DEP_NB_REQ, array_of_requests, &outcount, array_of_indices, array_of_statuses);
        if(0 == outcount) break;  /* nothing ready right now */
        for( index = 0; index < outcount; index++ ) {
            i = array_of_indices[index];
            status = &(array_of_statuses[index]);

            if(i < dague_mpi_activations) {
                assert(REMOTE_DEP_ACTIVATE_TAG == status->MPI_TAG);
                DEBUG(("MPI:\tFROM\t%d\tActivate\t% -8s\ti=%d\twith datakey %lx\tparams %lx\n",
                       status->MPI_SOURCE, remote_dep_cmd_to_string(&dep_activate_buff[i]->msg, tmp, 128),
                       i, dep_activate_buff[i]->msg.deps, dep_activate_buff[i]->msg.which));
                remote_dep_mpi_save_activation( eu_context, i, status );
                MPI_Start(&dep_activate_req[i]);
            } else if(i < dague_mpi_transferts) {
                i -= dague_mpi_activations; /* shift i */
                assert(REMOTE_DEP_GET_DATA_TAG == status->MPI_TAG);
                remote_dep_mpi_save_put( eu_context, i, status );
                MPI_Start(&dep_get_req[i]);
            } else {
                i -= dague_mpi_transferts;  /* shift i */
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
                    if(i < (DEP_NB_CONCURENT * MAX_PARAM_COUNT))
                    {
                        k = i%MAX_PARAM_COUNT;
                        i = i/MAX_PARAM_COUNT;
                        deps = (dague_remote_deps_t*) dep_pending_recv_array[i];
                        DEBUG(("MPI:\tFROM\t%d\tGet END  \t% -8s\ti=%d,k=%d\twith datakey na        \tparams %lx\t(tag=%d)\n",
                            status->MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, 128), i, k, deps->msg.which, status->MPI_TAG));
                        DEBUG_MARK_DTA_MSG_END_RECV(status->MPI_TAG);
                        TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_ek, i+k);
                        remote_dep_mpi_get_end(eu_context, deps, i, k);
                        ret++;
                    } else {
                        i -= (DEP_NB_CONCURENT * MAX_PARAM_COUNT);
                        assert((i >= 0) && (i < DEP_NB_CONCURENT * MAX_PARAM_COUNT));
                        assert(REMOTE_DEP_PUT_DATA_TAG == status->MPI_TAG);
                        k = i%MAX_PARAM_COUNT;
                        i = i/MAX_PARAM_COUNT;
                        DEBUG(("MPI:\tTO\tna\tEag END   \t% -8s\ti=%d,k=%d\twith datakey na         \tparams %lx\t(tag=%d)\n", "na", i, k, REMOTE_DEP_PUT_DATA_TAG));
                    }
                }
            }
        }
    } while(1);
    return ret;
}

static inline dague_list_item_t* rdep_fifo_push_ordered_get( dague_list_t* fifo,
                                                             dague_list_item_t* elem )
{
    dague_list_item_t* position;
    dague_dep_wire_get_fifo_elem_t* ec;
    dague_dep_wire_get_fifo_elem_t* input = (dague_dep_wire_get_fifo_elem_t*)elem;

    position = DAGUE_ULIST_ITERATOR(fifo, item,
    {
        ec = (dague_dep_wire_get_fifo_elem_t*)item;
        if( ec->priority < input->priority )
            break;
    });
    dague_ulist_add(fifo, position, elem);
    return elem;
}

static void remote_dep_mpi_save_put( dague_execution_unit_t* eu_context, int i, MPI_Status* status )
{
#ifdef DAGUE_DEBUG
    char tmp[128];
#endif
    dague_dep_wire_get_fifo_elem_t* item;
    remote_dep_wire_get_t* task;
    dague_remote_deps_t *deps;
    
    item = (dague_dep_wire_get_fifo_elem_t*)malloc(sizeof(dague_dep_wire_get_fifo_elem_t));
    DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)item);
    task = &(item->task);
    memcpy( task, &dep_get_buff[i], sizeof(remote_dep_wire_get_t) );
    deps = (dague_remote_deps_t*) (uintptr_t) task->deps;
    item-> priority = deps->max_priority;
    item->peer = status->MPI_SOURCE;
    rdep_fifo_push_ordered_get( &dague_put_fifo, (dague_list_item_t*)item );
    DEBUG(("MPI: Put DELAYED for %s from %d tag %u which 0x%x (deps %p)\n",
           remote_dep_cmd_to_string(&deps->msg, tmp, 128), item->peer, task->tag, task->which, (void*)deps));
    /* Check if we can push any new puts */
    for( i = 0; i < DEP_NB_CONCURENT; i++ ) {
        if( NULL == dep_pending_put_array[i] ) {
            item = (dague_dep_wire_get_fifo_elem_t*)dague_ufifo_pop(&dague_put_fifo);
            remote_dep_mpi_put_start(eu_context, item, i );
            break;
        }
    }
}

static void remote_dep_mpi_put_eager(dague_execution_unit_t* eu_context, int rank, remote_dep_wire_activate_t* msg)
{
    dague_remote_deps_t* deps = (dague_remote_deps_t*) (uintptr_t) msg->deps;
    remote_dep_datakey_t which = RDEP_MSG_EAGER_CLR(msg);
    for(int k = 0; which>>k; k++) {
        assert(k < MAX_PARAM_COUNT);
        if(!((1<<k) & which)) continue;
        void* data = ADATA(deps->output[k].data);
        MPI_Datatype dtt = deps->output[k].type->opaque_dtt;
        int tag = REMOTE_DEP_PUT_DATA_TAG;

#ifdef DAGUE_DEBUG
        char type_name[MPI_MAX_OBJECT_NAME]; int len;
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG(("MPI:\tTO\t%d\tPut EAGER\tunknown \tj=X,k=%d\twith datakey %lx at %p type %s\t(tag=%d)\n",
               rank, k, msg->deps, data, type_name, tag));
#endif

#if defined(DAGUE_STATS)
        {
            MPI_Aint lb, size;
            MPI_Type_get_extent(dtt, &lb, &size);
            DAGUE_STATACC_ACCUMULATE(counter_data_messages_sent, 1);
            DAGUE_STATACC_ACCUMULATE(counter_bytes_sent, size);
        }
#endif

#if defined(DAGUE_PROF_TRACE)
        TAKE_TIME_WITH_INFO(MPIsnd_eager, MPI_Data_plds_sk, 0,
                            eu_context->master_context->my_rank, rank, (*msg));
#else
        (void) eu_context;
#endif /* DAGUE_PROF_TRACE */
#ifndef DAGUE_PROF_DRY_DEP
        MPI_Isend(data, 1, dtt, rank, tag, dep_comm, &dep_put_eager_req[active_eagers+k]); 
#endif
#if defined(DAGUE_PROF_TRACE)
        TAKE_TIME_WITH_INFO(MPIsnd_eager, MPI_Data_plds_ek, 0,
                            eu_context->master_context->my_rank, rank, (*msg));
#endif /* DAGUE_PROF_TRACE */
        DEBUG_MARK_DTA_MSG_START_SEND(rank, data, tag);

    }
    active_eagers++;
    remote_dep_complete_one_and_cleanup(deps); 
}

static void remote_dep_mpi_put_start(dague_execution_unit_t* eu_context, dague_dep_wire_get_fifo_elem_t* item, int i)
{
    remote_dep_wire_get_t* task = &(item->task);
    dague_remote_deps_t* deps = (dague_remote_deps_t*) (uintptr_t) task->deps;
    int tag = task->tag;
    void* data;
    MPI_Datatype dtt;
#ifdef DAGUE_DEBUG
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif

    DEBUG_MARK_CTL_MSG_GET_RECV(item->peer, (void*)task, task);

    assert(task->which);
    DEBUG(("MPI:\tPUT which=%lx\n", task->which));
    for(int k = 0; task->which>>k; k++) {
        assert(k < MAX_PARAM_COUNT);
        if(!((1<<k) & task->which)) continue;
        //DEBUG(("MPI:\t%p[%d] %p, %p\n", deps, k, deps->output[k].data, GC_DATA(deps->output[k].data)));
        data = ADATA(deps->output[k].data);
        dtt = deps->output[k].type->opaque_dtt;
#ifdef DAGUE_DEBUG
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG(("MPI:\tTO\t%d\tPut START\tunknown \tj=%d,k=%d\twith datakey %lx at %p type %s\t(tag=%d)\n",
               item->peer, i, k, task->deps, data, type_name, tag+k));
#endif

#if defined(DAGUE_STATS)
        {
            MPI_Aint lb, size;
            MPI_Type_get_extent(dtt, &lb, &size);
            DAGUE_STATACC_ACCUMULATE(counter_data_messages_sent, 1);
            DAGUE_STATACC_ACCUMULATE(counter_bytes_sent, size);
        }
#endif

#if defined(DAGUE_PROF_TRACE)
        TAKE_TIME_WITH_INFO(MPIsnd_prof[i], MPI_Data_plds_sk, i,
                            eu_context->master_context->my_rank, item->peer, deps->msg);
#else
        (void) eu_context;
#endif /* DAGUE_PROF_TRACE */
        MPI_Isend(data, 1, dtt, item->peer, tag + k, dep_comm, &dep_put_snd_req[i*MAX_PARAM_COUNT+k]);
        DEBUG_MARK_DTA_MSG_START_SEND(item->peer, data, tag+k);
    }
    dep_pending_put_array[i] = item;
}

static void remote_dep_mpi_put_end(dague_execution_unit_t* eu_context, int i, int k, MPI_Status* status)
{
    dague_dep_wire_get_fifo_elem_t* item = dep_pending_put_array[i];
    remote_dep_wire_get_t* task = &(item->task);
    dague_remote_deps_t* deps = (dague_remote_deps_t*)(uintptr_t)task->deps;

    DEBUG(("MPI:\tTO\tna\tPut END  \tunknown \tj=%d,k=%d\twith datakey %lx\tparams %lx\t(tag=%d)\n",
           i, k, deps, task->which, status->MPI_TAG)); (void)status;
    DEBUG_MARK_DTA_MSG_END_SEND(status->MPI_TAG);
    AUNREF(deps->output[k].data);
    TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_ek, i);
    task->which ^= (1<<k);
    /* Are we done yet ? */
    if(0 == task->which) {
        remote_dep_dec_flying_messages(deps->dague_object, eu_context->master_context);
    }
    remote_dep_complete_one_and_cleanup(deps);
    if( 0 == task->which ) {
        free(item);
        dep_pending_put_array[i] = NULL;
        if( !dague_ulist_is_empty(&dague_put_fifo) ) {
            item = (dague_dep_wire_get_fifo_elem_t*)dague_ufifo_pop(&dague_put_fifo);
            if( NULL != item ) {
                remote_dep_mpi_put_start(eu_context, item, i );
            }
        }
    }
}

static void remote_dep_mpi_save_activation( dague_execution_unit_t* eu_context, int i, MPI_Status* status )
{
#ifdef DAGUE_DEBUG
    char tmp[128];
#endif
    dague_remote_deps_t *saved_deps, *deps = dep_activate_buff[i];
    
    saved_deps = remote_deps_allocation(&dague_remote_dep_context.freelist);
    DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)saved_deps);
    /* Update the required fields */
    saved_deps->msg = deps->msg;
    RDEP_MSG_EAGER_CLR(&saved_deps->msg);
    saved_deps->msg.deps = 0; /* contains the mask of deps presatisfied */
    saved_deps->from = status->MPI_SOURCE;
    
    /* Retrieve the data arenas and update the msg.which to reflect all the data
     * we should be receiving from the father. If some of the dependencies have
     * been dropped, force their release.
     */
    remote_dep_get_datatypes(saved_deps);
  
    for(int k = 0; saved_deps->msg.which>>k; k++) {
        if(!(saved_deps->msg.which & (1<<k))) continue;
        /* Check for all CTL messages, that do not carry payload */
        if(NULL == saved_deps->output[k].type) {
            DEBUG(("MPI:\tHERE\t%d\tGet NONE\t% -8s\ti=%d,k=%d\twith datakey %lx at <NA> type CONTROL extent 0\t(tag=na)\n", saved_deps->from, remote_dep_cmd_to_string(&deps->msg,tmp,128), i, k, deps->msg.deps));
            saved_deps->output[k].data = (void*)2; /* the first non zero even value */
            saved_deps->msg.deps |= 1<<k;
            continue;
        }
        /* Check if we have eager deps to satisfy quickly */
        if(RDEP_MSG_EAGER(&deps->msg))
        {
            assert(NULL == saved_deps->output[k].data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
            if(NULL == saved_deps->output[k].data) {
                saved_deps->output[k].data = dague_arena_get(saved_deps->output[k].type);
                DEBUG(("MPI:\tMalloc new remote tile %p size %zu\n", saved_deps->output[k].data, saved_deps->output[k].type->elem_size));
                assert(saved_deps->output[k].data != NULL);
            }
            DEBUG(("MPI:\tFROM\t%d\tGet EAGER\t% -8s\ti=%d,k=%d\twith datakey %lx at %p\t(tag=%d)\n",
                   saved_deps->from, remote_dep_cmd_to_string(&saved_deps->msg, tmp, 128), i, k, deps->msg.deps, ADATA(saved_deps->output[k].data), REMOTE_DEP_PUT_DATA_TAG));
#ifndef DAGUE_PROF_DRY_DEP
            MPI_Recv(ADATA(saved_deps->output[k].data), 1, saved_deps->output[k].type->opaque_dtt, saved_deps->from, REMOTE_DEP_PUT_DATA_TAG, dep_comm, MPI_STATUS_IGNORE);
#endif
            saved_deps->msg.deps |= 1<<k;
            continue;
        }
    }

    /* Release all the already satisfied deps without posting the RDV */
    if(saved_deps->msg.deps) { 
#ifdef DAGUE_DEBUG
        for(int k = 0; saved_deps->msg.deps>>k; k++) 
            if((1<<k) & saved_deps->msg.deps)
                DEBUG(("MPI:\tHERE\t%d\tGet LOCAL\t% -8s\ti=%d,k=%d\twith datakey %lx at %p ALREADY SATISFIED\t(tag=na)\n",
                       saved_deps->from, remote_dep_cmd_to_string(&saved_deps->msg, tmp, 128), i, k, deps->msg.deps, ADATA(saved_deps->output[k].data) ));
#endif
        remote_dep_release(eu_context, saved_deps);
    }

    /* Store the request in the rdv queue if any unsatisfied dep exist at this
     * point */
    if(saved_deps->msg.which) {
        saved_deps->msg.deps = deps->msg.deps;
        rdep_fifo_push_ordered_cmd( &dague_activations_fifo, (dague_list_item_t*)saved_deps );
    }
    else
    {
        DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)saved_deps);
        dague_atomic_lifo_push(&dague_remote_dep_context.freelist, (dague_list_item_t*)saved_deps);
    }

    /* Check if we have some ordered rdv get to treat */
    for( i = 0; i < DEP_NB_CONCURENT; i++ ) {
        if( NULL == dep_pending_recv_array[i] ) {
            deps = (dague_remote_deps_t*)dague_ufifo_pop(&dague_activations_fifo);
            if(deps) remote_dep_mpi_get_start(eu_context, deps, i );
            break;
        }
    }
}

static void remote_dep_mpi_get_start(dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i)
{
#ifdef DAGUE_DEBUG
    char tmp[128];
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif
    MPI_Datatype dtt;
    remote_dep_wire_get_t msg;
    remote_dep_wire_activate_t* task = &(deps->msg);
    int from = deps->from;
    void* data;

    (void)eu_context;
    DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, (void*)task, task);

    msg.which = task->which;
    msg.deps  = task->deps;
    msg.tag   = NEXT_TAG;
    
    for(int k = 0; msg.which >> k; k++) {
        if( !((1<<k) & msg.which) ) continue;
        dtt = deps->output[k].type->opaque_dtt;
        data = deps->output[k].data;
        assert(NULL == data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
        if(NULL == data) {
            data = dague_arena_get(deps->output[k].type);
            DEBUG(("MPI:\tMalloc new remote tile %p size %zu\n", data, deps->output[k].type->elem_size));
            assert(data != NULL);
            deps->output[k].data = data;
        }
#ifdef DAGUE_PROF_DRY_DEP
        (void)dtt;
        (void)dep_put_rcv_req;
        msg.which &= ~(1<<k);
        remote_dep_mpi_get_end(eu_context, deps, i, k);
#else
#  ifdef DAGUE_DEBUG
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG(("MPI:\tTO\t%d\tGet START\t% -8s\ti=%d,k=%d\twith datakey %lx at %p type %s extent %d\t(tag=%d)\n", from, remote_dep_cmd_to_string(task, tmp, 128), i, k, task->deps, ADATA(data), type_name, deps->output[k].type->elem_size, NEXT_TAG+k));
        TAKE_TIME_WITH_INFO(MPIrcv_prof[i], MPI_Data_pldr_sk, i+k, from,
                            eu_context->master_context->my_rank, deps->msg);
#  endif /* defined(DAGUE_PROF_TRACE) */
        MPI_Irecv(ADATA(data), 1, 
                  dtt, from, NEXT_TAG + k, dep_comm, 
                  &dep_put_rcv_req[i*MAX_PARAM_COUNT+k]);
        DEBUG_MARK_DTA_MSG_START_RECV(from, data, NEXT_TAG + k);
#endif
    }
    if(msg.which)
    {
        TAKE_TIME_WITH_INFO(MPIctl_prof, MPI_Data_ctl_sk, get, 
                            from, eu_context->master_context->my_rank, (*task));
        MPI_Send(&msg, datakey_count, datakey_dtt, from, 
                 REMOTE_DEP_GET_DATA_TAG, dep_comm);
        assert(NULL == dep_pending_recv_array[i]);
        dep_pending_recv_array[i] = deps;
        TAKE_TIME(MPIctl_prof, MPI_Data_ctl_ek, get++);
        DEBUG_MARK_CTL_MSG_GET_SENT(from, (void*)&msg, &msg);

#if defined(DAGUE_STATS)
        {
            MPI_Aint _lb, _size;
            MPI_Type_get_extent(datakey_dtt, &_lb, &_size);
            DAGUE_STATACC_ACCUMULATE(counter_control_messages_sent, 1);
            DAGUE_STATACC_ACCUMULATE(counter_bytes_sent, _size * datakey_count);
        }
#endif
    }

    deps->msg.deps = 0; /* now this is the mask of finished deps */
    INC_NEXT_TAG(MAX_PARAM_COUNT);
}

static void remote_dep_mpi_get_end(dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i, int k)
{
    deps->msg.deps = 1<<k;
    remote_dep_release(eu_context, deps);
    AUNREF(deps->output[k].data);
    if(deps->msg.which == deps->msg.deps) {
        DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)deps);
        dague_atomic_lifo_push(&dague_remote_dep_context.freelist, (dague_list_item_t*)deps);
        dep_pending_recv_array[i] = NULL;
        if( !dague_ulist_is_empty(&dague_activations_fifo) ) {
            deps = (dague_remote_deps_t*)dague_ufifo_pop(&dague_activations_fifo);
            if( NULL != deps ) {
                remote_dep_mpi_get_start(eu_context, deps, i );
            }
        }
    }
}
