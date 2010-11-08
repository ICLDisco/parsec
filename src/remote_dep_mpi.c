/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* /!\  THIS FILE IS NOT INTENDED TO BE COMPILED ON ITS OWN
 *      It should be included from remote_dep.c if HAVE_MPI is defined
 */

#include <mpi.h>
#include "profiling.h"
#include "freelist.h"
#include "arena.h"
#include "fifo.h"

#define DAGUE_REMOTE_DEP_USE_THREADS

static int remote_dep_mpi_init(dague_context_t* context);
static int remote_dep_mpi_fini(dague_context_t* context);
static int remote_dep_mpi_on(dague_context_t* context);
static int remote_dep_mpi_off(dague_context_t* context);
static int remote_dep_mpi_send_dep(int rank, remote_dep_wire_activate_t* msg);
static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context);
static int remote_dep_get_datatypes(dague_remote_deps_t* origin);
static int remote_dep_release(dague_execution_unit_t* eu_context, dague_remote_deps_t* origin);

static int remote_dep_nothread_send(int rank, dague_remote_deps_t* deps);
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
static int remote_dep_dequeue_nothread_dini(dague_context_t* context);
static int remote_dep_dequeue_nothread_progress_one(dague_execution_unit_t* eu_context);
#   define remote_dep_init(ctx) remote_dep_dequeue_nothread_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_nothread_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_mpi_on(ctx)
#   define remote_dep_off(ctx)  remote_dep_mpi_off(ctx)
#   define remote_dep_send(rank, deps) remote_dep_dequeue_send(rank, deps)
#   define remote_dep_progress(ctx) remote_dep_dequeue_nothread_progress_one(ctx)
#endif 

static void remote_dep_mpi_put_start(remote_dep_wire_get_t* task, int to, int i);
static void remote_dep_mpi_save_activation( dague_execution_unit_t* eu_context, int i, MPI_Status* status );
static void remote_dep_mpi_get_start(dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i );
static void remote_dep_mpi_get_end(dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i, int k);

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
        dague_arena_chunk_t* source;
        void *destination;
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

#ifdef DAGUE_DEBUG
static char* remote_dep_cmd_to_string(remote_dep_wire_activate_t* origin, char* str, size_t len)
{
    unsigned int i, index = 0;
    dague_object_t* object;
    const dague_t* function;
    
    object = dague_object_lookup( origin->object_id );
    function = object->functions_array[origin->function_id];

    index += snprintf( str + index, len - index, "%s", function->name );
    if( index >= len ) return str;
    for( i = 0; i < function->nb_locals; i++ ) {
        index += snprintf( str + index, len - index, "_%d",
                           origin->locals[i].value );
        if( index >= len ) return str;
    }
    return str;
}
#endif

pthread_t dep_thread_id;
dague_dequeue_t dep_cmd_queue;
dague_fifo_t    dep_cmd_fifo;        /* ordered non threaded fifo */
dague_fifo_t    dague_activations_fifo;  /* ordered non threaded fifo */
dague_remote_deps_t** dep_pending_recv_array;
volatile int np;
static int dep_enabled;

static void *remote_dep_dequeue_main(dague_context_t* context);

static int remote_dep_dequeue_init(dague_context_t* context)
{
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);

    dague_dequeue_construct(&dep_cmd_queue);
    dague_fifo_construct(&dep_cmd_fifo);
    dague_fifo_construct(&dague_activations_fifo);
    dep_pending_recv_array = (dague_remote_deps_t**)calloc(DEP_NB_CONCURENT,sizeof(dague_remote_deps_t*));

    MPI_Comm_size(MPI_COMM_WORLD, (int*) &np);
    if(1 < np) {
        np = 0;
        pthread_create(&dep_thread_id,
                       &thread_attr,
                       (void* (*)(void*))remote_dep_dequeue_main,
                       (void*)context);
        
        while(0 == np); /* wait until the thread inits MPI */
    }
    return np;
}

#ifndef DAGUE_REMOTE_DEP_USE_THREADS
static int remote_dep_dequeue_nothread_init(dague_context_t* context)
{
    dague_dequeue_construct(&dep_cmd_queue);
    dague_fifo_construct(&dep_cmd_fifo);
    dague_fifo_construct(&dague_activations_fifo);
    dep_pending_recv_array = (dague_remote_deps_t**)calloc(DEP_NB_CONCURENT*sizeof(dague_remote_deps_t*));

    MPI_Comm_size(MPI_COMM_WORLD, (int*) &np);
    remote_dep_mpi_init(context);
    return np;
}

static int remote_dep_dequeue_nothread_fini(dague_context_t* context)
{
    remote_dep_mpi_fini(context);
    free(dep_pending_recv_array);
    dep_pending_recv_array = NULL;

    return 0;
}
#endif

static int remote_dep_dequeue_on(dague_context_t* context)
{
    if(1 < context->nb_nodes)
    {        
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        item->action = DEP_CTL;
        item->cmd.ctl.enable = 1;
        item->priority = 0;
        DAGUE_LIST_ITEM_SINGLETON(item);
        dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);
        return 1;
    }
    return 0;
}

static int remote_dep_dequeue_off(dague_context_t* context)
{
    if(1 < context->nb_nodes)
    {        
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        item->action = DEP_CTL;
        item->cmd.ctl.enable = 0;
        item->priority = 0;
        DAGUE_LIST_ITEM_SINGLETON(item);
        dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);
    }
    return 0;
}

static int remote_dep_dequeue_fini(dague_context_t* context)
{
    if(1 < context->nb_nodes)
    {        
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        dague_context_t *ret;
        item->action = DEP_CTL;
        item->cmd.ctl.enable = -1;
        item->priority = 0;
        DAGUE_LIST_ITEM_SINGLETON(item);
        dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);
        
        pthread_join(dep_thread_id, (void**) &ret);
        assert(ret == context);
    }
    return 0;
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

void dague_remote_dep_memcpy(void *dst, dague_arena_chunk_t *src, dague_remote_dep_datatype_t datatype)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    item->action = DEP_MEMCPY;
    item->cmd.memcpy.source = src;
    item->cmd.memcpy.destination = dst;
    item->cmd.memcpy.datatype = datatype;
    AREF(src);
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
    exec_context.function = exec_context.dague_object->functions_array[origin->msg.function_id];

    for(int i = 0; i < exec_context.function->nb_locals; i++)
        exec_context.locals[i] = origin->msg.locals[i];

    return exec_context.function->release_deps(NULL, &exec_context,
                                               DAGUE_ACTION_RECV_INIT_REMOTE_DEPS | origin->msg.which,
                                               origin, NULL);
#if 0
    /* Remove CONTROLS from the list Âf things to receive, make them locals */
    for(int k = 0; origin->msg.which>>k; k++)
    {
        if(NULL == origin->output[k].datatype)
        {
            origin->msg.which &= ~(1<<k);
        }
    }
#endif
}

static int remote_dep_release(dague_execution_unit_t* eu_context, dague_remote_deps_t* origin)
{
    int actions = DAGUE_ACTION_NO_PLACEHOLDER | DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_RELEASE_REMOTE_DEPS;
    dague_execution_context_t exec_context;
    dague_arena_chunk_t* data[MAX_PARAM_COUNT];
    int ret, i;
    
    exec_context.dague_object = dague_object_lookup( origin->msg.object_id );
    exec_context.function = exec_context.dague_object->functions_array[origin->msg.function_id];
    for( i = 0; i < exec_context.function->nb_locals; i++)
        exec_context.locals[i] = origin->msg.locals[i];

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != exec_context.function->out[i]); i++) {
        data[i] = NULL;
        if(origin->msg.deps & (1 << i)) {
            //DEBUG(("MPI:\tDATA %p released from %p[%d]\n", GC_DATA(origin->output[i].data), origin, i));
            data[i] = origin->output[i].data;
#ifdef DAGUE_DEBUG
/*            {
                char tmp[128];
                void* _data = ADATA(data[i]);
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
                                              origin, data);
    origin->msg.which ^= origin->msg.deps;
    origin->msg.deps = 0;
    return ret;
}

#define YIELD_TIME 5000
#include "bindthread.h"

static int do_nano = 0;
static int keep_probing = 1;

static inline dague_list_item_t* dague_fifo_push_ordered( dague_fifo_t* fifo,
                                                          dague_list_item_t* elem )
{
    dep_cmd_item_t* ec;
    dep_cmd_item_t* input = (dep_cmd_item_t*)elem;
    dague_list_item_t* current = (dague_list_item_t*)fifo->fifo_ghost.list_next;

    while( current != &(fifo->fifo_ghost) ) {
        ec = (dep_cmd_item_t*)current;
        if( ec->priority < input->priority )
            break;
        current = (dague_list_item_t *)current->list_next;
    }
    /* Add the input element before the current one */
    elem->list_prev = current->list_prev;
    elem->list_next = current;
    elem->list_prev->list_next = elem;
    elem->list_next->list_prev = elem;
    return elem;
}
#define DAGUE_FIFO_PUSH  dague_fifo_push_ordered

static int remote_dep_dequeue_nothread_progress_one(dague_execution_unit_t* eu_context)
{
    dep_cmd_item_t* item;
    int ctl;
    int ret = 0;

    /**
     * Move as many elements as possible from the dequeue into our ordered lifo.
     */
    while( NULL != (item = (dep_cmd_item_t*) dague_dequeue_pop_front(&dep_cmd_queue)) ) {
        DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)item);
        DAGUE_FIFO_PUSH(&dep_cmd_fifo, (dague_list_item_t*)item);
    }
    item = (dep_cmd_item_t*)dague_fifo_pop(&dep_cmd_fifo);

    if(NULL == item ) {
        if(dep_enabled) {
            ret = remote_dep_mpi_progress(eu_context);
        }
        if(do_nano && !ret) {
            struct timespec ts;
            ts.tv_sec = 0; ts.tv_nsec = YIELD_TIME;
            nanosleep(&ts, NULL);
        }
        return ret;
    }
    switch(item->action) {
    case DEP_ACTIVATE:
        remote_dep_nothread_send(item->cmd.activate.rank, item->cmd.activate.deps);
        break;
    case DEP_CTL:
        ctl = item->cmd.ctl.enable;
        assert((ctl * ctl) <= 1);
        if(-1 == ctl) {
            keep_probing = 0;
        }
        if(0 == ctl) {
            remote_dep_mpi_off(eu_context->master_context);
        }
        if(1 == ctl) {
            remote_dep_mpi_on(eu_context->master_context);
        }
        break;
    case DEP_MEMCPY:
        remote_dep_nothread_memcpy(item->cmd.memcpy.destination, 
                                   item->cmd.memcpy.source,
                                   item->cmd.memcpy.datatype);
        break;
    default:
        break;
    }
    free(item);
    return (ret + 1);
}

static void* remote_dep_dequeue_main(dague_context_t* context)
{
    int ctl = -1;

    ctl = dague_bindthread(context->nb_cores);
    if(ctl != context->nb_cores) do_nano = 1;
    else fprintf(stderr, "DAGuE\tMPI bound to physical core %d\n", ctl);
    np = remote_dep_mpi_init(context);

    do {
        remote_dep_dequeue_nothread_progress_one(context->execution_units[0]);
    } while(keep_probing);
    
    remote_dep_mpi_fini(context);
    return context;
}


static int remote_dep_nothread_send(int rank, dague_remote_deps_t* deps)
{
    int k;
    int rank_bank = rank / (sizeof(uint32_t) * 8);
    uint32_t rank_mask = 1 << (rank % (sizeof(uint32_t) * 8));
    int output_count = deps->output_count;
    remote_dep_wire_activate_t msg = deps->msg;


    msg.which = 0;
#if !defined(DAGUE_DEBUG_DRY_DEP)
    for(k = 0; output_count; k++)
    {
        output_count -= deps->output[k].count;
        if(deps->output[k].rank_bits[rank_bank] & rank_mask)
        {
            if(NULL == deps->output[k].type)
            {
#ifdef DAGUE_DEBUG
                char tmp[128];

                assert(0 == k); /* only the first parameter can be a control */
                DEBUG((" CTL\t%s\tparam%d\tdemoted to be a control\n",remote_dep_cmd_to_string(&deps->msg, tmp, 128), k));
#endif
                remote_dep_dec_flying_messages(deps->eu_context->master_context);
            }
            else
            {
                msg.which |= (1<<k);
            }
        }
    }
#endif
    remote_dep_mpi_send_dep(rank, &msg);
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
static dague_thread_profiling_t* MPIrcv_prof[DEP_NB_CONCURENT];
static unsigned long act = 0;
static int MPI_Activate_sk, MPI_Activate_ek;
static unsigned long get = 0;
static int MPI_Data_ctl_sk, MPI_Data_ctl_ek;
static int MPI_Data_plds_sk, MPI_Data_plds_ek;
static int MPI_Data_pldr_sk, MPI_Data_pldr_ek;

#define MPI_PROFILING_SIZE (64*1024)

static void remote_dep_mpi_profiling_init(void)
{
    int i;
    
    dague_profiling_add_dictionary_keyword( "MPI_ACTIVATE", "fill:#FF0000",
                                            0, NULL,
                                            &MPI_Activate_sk, &MPI_Activate_ek);
    dague_profiling_add_dictionary_keyword( "MPI_DATA_CTL", "fill:#000077",
                                            0, NULL,
                                            &MPI_Data_ctl_sk, &MPI_Data_ctl_ek);
    dague_profiling_add_dictionary_keyword( "MPI_DATA_PLD_SND", "fill:#B08080",
                                            0, NULL,
                                            &MPI_Data_plds_sk, &MPI_Data_plds_ek);
    dague_profiling_add_dictionary_keyword( "MPI_DATA_PLD_RCV", "fill:#80B080",
                                            0, NULL,
                                            &MPI_Data_pldr_sk, &MPI_Data_pldr_ek);
    
    MPIctl_prof = dague_profiling_thread_init( MPI_PROFILING_SIZE, "MPI ctl");
    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        MPIsnd_prof[i] = dague_profiling_thread_init( MPI_PROFILING_SIZE / DEP_NB_CONCURENT, "MPI isend(req=%d)", i);
        MPIrcv_prof[i] = dague_profiling_thread_init( MPI_PROFILING_SIZE / DEP_NB_CONCURENT, "MPI irecv(req=%d)", i);
    }    
}

#define TAKE_TIME(PROF, KEY, I)  dague_profiling_trace((PROF), (KEY), (I), NULL)
#else
#define TAKE_TIME(PROF, KEY, I)
#define remote_dep_mpi_profiling_init() do {} while(0)
#endif  /* DAGUE_PROF_TRACE */

/* TODO: smart use of dague context instead of ugly globals */
static MPI_Comm dep_comm;
#define DEP_NB_REQ (2 * DEP_NB_CONCURENT + 2 * (DEP_NB_CONCURENT * MAX_PARAM_COUNT))
static MPI_Request dep_req[DEP_NB_REQ];
static MPI_Request* dep_activate_req    = &dep_req[0 * DEP_NB_CONCURENT];
static MPI_Request* dep_get_req         = &dep_req[1 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_snd_req     = &dep_req[2 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_rcv_req     = &dep_req[2 * DEP_NB_CONCURENT + DEP_NB_CONCURENT * MAX_PARAM_COUNT];
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
#include <limits.h>
#if ULONG_MAX < UINTPTR_MAX
#error "unsigned long is not large enough to hold a pointer!"
#endif
static int MAX_MPI_TAG;
static int NEXT_TAG = REMOTE_DEP_MAX_CTRL_TAG+1;
#define INC_NEXT_TAG(k) do { \
    assert(k < MAX_MPI_TAG); \
    if(NEXT_TAG < (MAX_MPI_TAG - k)) \
        NEXT_TAG += k; \
    else \
        NEXT_TAG = REMOTE_DEP_MAX_CTRL_TAG + k + 1; \
} while(0)

static int remote_dep_mpi_init(dague_context_t* context)
{
    int i, np;
    int mpi_tag_ub_exists;
    int *ub;
    MPI_Comm_dup(MPI_COMM_WORLD, &dep_comm);

    MPI_Comm_get_attr(dep_comm, MPI_TAG_UB, &ub, &mpi_tag_ub_exists);    
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

    MPI_Comm_size(dep_comm, &np); context->nb_nodes = np;
    MPI_Comm_rank(dep_comm, &context->my_rank);
    for(i = 0; i < DEP_NB_REQ; i++)
    {        
        dep_req[i] = MPI_REQUEST_NULL;
    }
    dep_enabled = 0;
    remote_dep_mpi_profiling_init();

    return np;
}

static int remote_dep_mpi_on(dague_context_t* context)
{
    int i;

    (void)context;

    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        dep_activate_buff[i] = remote_deps_allocation(&remote_deps_freelist);
    }

#ifdef DAGUE_PROF_TRACE
    /* put a start marker on each line */
    TAKE_TIME(MPIctl_prof, MPI_Activate_sk, 0);
    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        TAKE_TIME(MPIsnd_prof[i], MPI_Activate_sk, 0);
        TAKE_TIME(MPIrcv_prof[i], MPI_Activate_sk, 0);
    }
    MPI_Barrier(dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, 0);
    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        TAKE_TIME(MPIsnd_prof[i], MPI_Activate_ek, 0);
        TAKE_TIME(MPIrcv_prof[i], MPI_Activate_ek, 0);
    }
#endif
    
    assert(dep_enabled == 0);
    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        MPI_Recv_init(&dep_activate_buff[i]->msg, dep_count, dep_dtt, MPI_ANY_SOURCE, REMOTE_DEP_ACTIVATE_TAG, dep_comm, &dep_activate_req[i]);
        MPI_Recv_init(&dep_get_buff[i], datakey_count, datakey_dtt, MPI_ANY_SOURCE, REMOTE_DEP_GET_DATA_TAG, dep_comm, &dep_get_req[i]);
        MPI_Start(&dep_activate_req[i]);
        MPI_Start(&dep_get_req[i]);
    }
    return dep_enabled = 1;
}

static int remote_dep_mpi_off(dague_context_t* context)
{
    MPI_Status status;
    int i, flag;

    (void)context;

    assert(dep_enabled == 1);

    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        MPI_Cancel(&dep_activate_req[i]); MPI_Test(&dep_activate_req[i], &flag, &status); MPI_Request_free(&dep_activate_req[i]);
        MPI_Cancel(&dep_get_req[i]); MPI_Test(&dep_get_req[i], &flag, &status);MPI_Request_free(&dep_get_req[i]);
    }
    for(i = 0; i < DEP_NB_REQ; i++)
    {
        assert(MPI_REQUEST_NULL == dep_req[i]);
    }
    return dep_enabled = 0;
}

static int remote_dep_mpi_fini(dague_context_t* context)
{
    if(dep_enabled) remote_dep_mpi_off(context);
    MPI_Comm_free(&dep_comm);
    return 0;
}

/* Send the activate tag */
static int remote_dep_mpi_send_dep(int rank, remote_dep_wire_activate_t* msg)
{
#ifdef DAGUE_DEBUG
    char tmp[128];
#endif
    
    assert(dep_enabled);
    TAKE_TIME(MPIctl_prof, MPI_Activate_sk, act);
    DEBUG(("MPI:\tTO\t%d\tActivate\t% -8s\ti=na\twith datakey %lx\tparams %lx\n", rank, remote_dep_cmd_to_string(msg, tmp, 128), msg->deps, msg->which));
    
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

    return 1;
}


static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context)
{
#ifdef DAGUE_DEBUG
    char tmp[128];
#endif
    MPI_Status status;
    int ret = 0;
    int i, flag;
    
    if(eu_context->eu_id != 0) return 0;
    
    assert(dep_enabled);
    do {
        MPI_Testany(DEP_NB_REQ, dep_req, &i, &flag, &status);
        if(!flag) continue;
        if(i < dague_mpi_activations) {
            assert(REMOTE_DEP_ACTIVATE_TAG == status.MPI_TAG);
            DEBUG(("MPI:\tFROM\t%d\tActivate\t% -8s\ti=%d\twith datakey %lx\tparams %lx\n",
                   status.MPI_SOURCE, remote_dep_cmd_to_string(&dep_activate_buff[i]->msg, tmp, 128),
                   i, dep_activate_buff[i]->msg.deps, dep_activate_buff[i]->msg.which));
            remote_dep_mpi_save_activation( eu_context, i, &status );
            MPI_Start(&dep_activate_req[i]);
            /*remote_dep_mpi_get_start(eu_context, i, &status );*/
        } else if(i < dague_mpi_transferts) {
            i -= dague_mpi_activations; /* shift i */
            assert(REMOTE_DEP_GET_DATA_TAG == status.MPI_TAG);
            remote_dep_mpi_put_start(&dep_get_buff[i], status.MPI_SOURCE, i);
        } else {
            i -= dague_mpi_transferts;  /* shift i */
            assert(i >= 0);
            if(i < (DEP_NB_CONCURENT * MAX_PARAM_COUNT)) {
                /* We finished sending the data, allow for more requests 
                 * to be processed */
                dague_remote_deps_t* deps; 
                int k;
                k = i % MAX_PARAM_COUNT;
                i = i / MAX_PARAM_COUNT;
                deps = (dague_remote_deps_t*) (uintptr_t) dep_get_buff[i].deps;
                DEBUG(("MPI:\tTO\tna\tPut END  \tunknown \tj=%d,k=%d\twith datakey %lx\tparams %lx\t(tag=%d)\n",
                       i, k, dep_get_buff[i].deps, dep_get_buff[i].which, status.MPI_TAG));
                DEBUG_MARK_DTA_MSG_END_SEND(status.MPI_TAG);
                AUNREF(deps->output[k].data);
                TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_ek, i);
                dep_get_buff[i].which ^= (1<<k);
                if(0 == dep_get_buff[i].which) {
                    MPI_Start(&dep_get_req[i]);
                    remote_dep_dec_flying_messages(eu_context->master_context);
                }

                /* remote_deps cleanup */
                deps->output_sent_count++;
                if(deps->output_count == deps->output_sent_count) {
                    unsigned int count;

                    k = 0;
                    count = 0;
                    while( count < deps->output_count ) {
                        for(unsigned int a = 0; a < (max_nodes_number + 31)/32; a++)
                            deps->output[k].rank_bits[a] = 0;
                        count += deps->output[k].count;
                        deps->output[k].count = 0;
#if defined(DAGUE_DEBUG)
                        deps->output[k].data = NULL;
                        deps->output[k].type = NULL;
#endif
                        k++;
                        assert(k < MAX_PARAM_COUNT);
                    }
                    deps->output_count = 0;
                    deps->output_sent_count = 0;
#if defined(DAGUE_DEBUG)
                    memset( &deps->msg, 0, sizeof(remote_dep_wire_activate_t) );
#endif
                    dague_atomic_lifo_push(deps->origin, 
                                           dague_list_item_singleton((dague_list_item_t*) deps));
                }
            } else {
                /* We received a data, call the matching release_dep */
                dague_remote_deps_t* deps;
                int k;
                i -= (DEP_NB_CONCURENT * MAX_PARAM_COUNT);
                assert((i >= 0) && (i < DEP_NB_CONCURENT * MAX_PARAM_COUNT));
                k = i%MAX_PARAM_COUNT;
                i = i/MAX_PARAM_COUNT;
                deps = (dague_remote_deps_t*) dep_pending_recv_array[i];
                DEBUG(("MPI:\tFROM\t%d\tGet END  \t% -8s\ti=%d,k=%d\twith datakey na        \tparams %lx\t(tag=%d)\n",
                       status.MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, 128), i, k, deps->msg.which, status.MPI_TAG));
                DEBUG_MARK_DTA_MSG_END_RECV(status.MPI_TAG);
                TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_ek, i+k);
                remote_dep_mpi_get_end(eu_context, deps, i, k);
                ret++;
            }
        }
    } while(flag);
    return ret;
}

static void remote_dep_mpi_put_start(remote_dep_wire_get_t* task, int to, int i)
{
    dague_remote_deps_t* deps = (dague_remote_deps_t*) (uintptr_t) task->deps;
    int tag = task->tag;
    void* data;
    MPI_Datatype dtt;
#ifdef DAGUE_DEBUG
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif

    DEBUG_MARK_CTL_MSG_GET_RECV(to, (void*)task, task);

    assert(dep_enabled);
    assert(task->which);
    DEBUG(("MPI:\tPUT which=%lx\n", task->which));
    for(int k = 0; task->which>>k; k++)
    {
        assert(k < MAX_PARAM_COUNT);
        if(!((1<<k) & task->which)) continue;
        //DEBUG(("MPI:\t%p[%d] %p, %p\n", deps, k, deps->output[k].data, GC_DATA(deps->output[k].data)));
        data = ADATA(deps->output[k].data);
        dtt = deps->output[k].type->opaque_dtt;
#ifdef DAGUE_DEBUG
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG(("MPI:\tTO\t%d\tPut START\tunknown \tj=%d,k=%d\twith datakey %lx at %p type %s\t(tag=%d)\n", to, i, k, task->deps, data, type_name, tag+k));
#endif

#if defined(DAGUE_STATS)
        {
            MPI_Aint lb, size;
            MPI_Type_get_extent(dtt, &lb, &size);
            DAGUE_STATACC_ACCUMULATE(counter_data_messages_sent, 1);
            DAGUE_STATACC_ACCUMULATE(counter_bytes_sent, size);
        }
#endif

        TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_sk, i);
        MPI_Isend(data, 1, dtt, to, tag + k, dep_comm, &dep_put_snd_req[i*MAX_PARAM_COUNT+k]);
        //DEBUG(("MPI:\tsend %p -> [0] %9.5f [1] %9.5f [2] %9.5f\n", data, ((double*)data)[0], ((double*)data)[1], ((double*)data)[2]));
        DEBUG_MARK_DTA_MSG_START_SEND(to, data, tag+k);
    }
}

static void remote_dep_mpi_save_activation( dague_execution_unit_t* eu_context, int i, MPI_Status* status )
{
#ifdef DAGUE_DEBUG
    char tmp[128];
#endif
    dague_remote_deps_t *saved_deps, *deps = dep_activate_buff[i];
    
    saved_deps = remote_deps_allocation(&remote_deps_freelist);
    DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)saved_deps);
    /* Update the required fields */
    saved_deps->msg = deps->msg;
    saved_deps->from = status->MPI_SOURCE;
    /**
     * Retrieve the data arenas and update the msg.which to reflect all the data
     * we should be receiving from the father. If some of the dependencies have
     * been dropped, force their release.
     */
    remote_dep_get_datatypes(saved_deps);

    assert( deps->msg.which == saved_deps->msg.which );  /* we do not support RO dep backtracking, make sure it doesn't happen yet */
    if(deps->msg.which != saved_deps->msg.which) {  /* some deps are considered as satisfied because they are RO */
        saved_deps->msg.which = deps->msg.which ^ saved_deps->msg.which;
        saved_deps->msg.deps = saved_deps->msg.which;
#ifdef DAGUE_DEBUG
        for(int k = 0; saved_deps->msg.which>>k; k++) 
            if((1<<k) & saved_deps->msg.which)
                DEBUG(("MPI:\tTO\t%d\tGet LOCAL\t% -8s\ti=%d,k=%d\twith data %lx at %p IS LOCAL\t(tag=%d)\n", saved_deps->from, remote_dep_cmd_to_string(&saved_deps->msg, tmp, 128), i, k, saved_deps, ADATA(saved_deps->output[k].data), NEXT_TAG+k));
#endif
        remote_dep_release(eu_context, saved_deps);
        if( saved_deps->msg.which == deps->msg.which ) {  /* all deps satisfied */
            DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)saved_deps);
            dague_atomic_lifo_push(&remote_deps_freelist, (dague_list_item_t*)saved_deps);
            goto submit_receives;
        }
        saved_deps->msg.deps = deps->msg.which ^ saved_deps->msg.which;
        saved_deps->msg.which = saved_deps->msg.deps;
    }
    dague_fifo_push_ordered( &dague_activations_fifo, (dague_list_item_t*)saved_deps );

 submit_receives:
    /* Check if we can push any new receives */
    for( i = 0; i < DEP_NB_CONCURENT; i++ ) {
        if( NULL == dep_pending_recv_array[i] ) {
            deps = (dague_remote_deps_t*)dague_fifo_pop(&dague_activations_fifo);
            remote_dep_mpi_get_start(eu_context, deps, i );
            break;
        }
    }
}

static void remote_dep_mpi_get_end(dague_execution_unit_t* eu_context, dague_remote_deps_t* deps, int i, int k)
{
                
                deps->msg.deps |= 1<<k;
                remote_dep_release(eu_context, deps);
                if(deps->msg.which == deps->msg.deps) {
                    DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)deps);
                    dague_atomic_lifo_push(&remote_deps_freelist, (dague_list_item_t*)deps);
                    dep_pending_recv_array[i] = NULL;
                    if( !dague_fifo_is_empty(&dague_activations_fifo) ) {
                        deps = (dague_remote_deps_t*)dague_fifo_pop(&dague_activations_fifo);
                        if( NULL != deps ) {
                            remote_dep_mpi_get_start(eu_context, deps, i );
                        }
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

    DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, (void*)task, task);

    msg.which = deps->msg.which;
    msg.deps = task->deps;
    msg.tag = NEXT_TAG;
    
    assert(dep_enabled);
    for(int k = 0; msg.which >> k; k++) {
        if( !((1<<k) & msg.which) ) continue;
        if( NULL == deps->output[k].type ) {
            DEBUG(("MPI:\tTO\t%d\tGet NONE\t% -8s\ti=%d,k=%d\twith datakey %lx at <NA> type CONTROL extent 0\t(tag=%d)\n", from, remote_dep_cmd_to_string(task,tmp,128), i, k, task->deps, NEXT_TAG+k));
            remote_dep_mpi_get_end(eu_context, deps, i, k);
            msg.which &= ~(1<<k);
            continue;
        }
        dtt = deps->output[k].type->opaque_dtt;
        data = deps->output[k].data;
        assert(NULL == data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
        if(NULL == data) {
            data = dague_arena_get(deps->output[k].type);
            DEBUG(("MPI:\tMalloc new remote tile %p size %zu\n", data, deps->output[k].type->elem_size));
            assert(data != NULL);
            deps->output[k].data = data;
        }
#ifdef DAGUE_DEBUG_DRY_DEP
        msg.which &= ~(1<<k);
#else
#ifdef DAGUE_DEBUG
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG(("MPI:\tTO\t%d\tGet START\t% -8s\ti=%d,k=%d\twith datakey %lx at %p type %s extent %d\t(tag=%d)\n", from, remote_dep_cmd_to_string(task, tmp, 128), i, k, task->deps, ADATA(data), type_name, deps->output[k].type->elem_size, NEXT_TAG+k));
#endif
        /*printf("%s:%d Allocate new TILE at %p\n", __FILE__, __LINE__, (void*)GC_DATA(deps->output[k].data));*/
        TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_sk, i+k);
        MPI_Irecv(ADATA(data), 1, 
                  dtt, from, NEXT_TAG + k, dep_comm, 
                  &dep_put_rcv_req[i*MAX_PARAM_COUNT+k]);
        DEBUG_MARK_DTA_MSG_START_RECV(from, data, NEXT_TAG + k);
#endif
    }
    if(msg.which)
    {
        TAKE_TIME(MPIctl_prof, MPI_Data_ctl_sk, get);
        MPI_Send(&msg, datakey_count, datakey_dtt, from, 
                 REMOTE_DEP_GET_DATA_TAG, dep_comm);
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

