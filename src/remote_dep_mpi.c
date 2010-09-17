/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* /!\  THIS FILE IS NOT INTENDED TO BE COMPILED ON ITS OWN
 *      It should be included from remote_dep.c if USE_MPI is defined
 */

#include <mpi.h>
#include "profiling.h"
#include "freelist.h"
#include "arena.h"

#define DAGUE_REMOTE_DEP_USE_THREADS
#define DEP_NB_CONCURENT 3
#undef FLOW_CONTROL

static int remote_dep_mpi_init(dague_context_t* context);
static int remote_dep_mpi_fini(dague_context_t* context);
static int remote_dep_mpi_on(dague_context_t* context);
static int remote_dep_mpi_off(dague_context_t* context);
static int remote_dep_mpi_send_dep(int rank, remote_dep_wire_activate_t* msg);
static int remote_dep_mpi_progress(dague_execution_unit_t* eu_context);


static int remote_dep_nothread_send(int rank, dague_remote_deps_t* deps);
static int remote_dep_nothread_get_datatypes(dague_remote_deps_t* origin);
static int remote_dep_nothread_release(dague_execution_unit_t* eu_context, dague_remote_deps_t* origin);
static int remote_dep_nothread_memcpy(void *dst, void *src, const dague_remote_dep_datatype_t datatype);

#ifdef DAGUE_REMOTE_DEP_USE_THREADS
static int remote_dep_dequeue_init(dague_context_t* context);
static int remote_dep_dequeue_fini(dague_context_t* context);
static int remote_dep_dequeue_on(dague_context_t* context);
static int remote_dep_dequeue_off(dague_context_t* context);
static int remote_dep_dequeue_send(int rank, dague_remote_deps_t* deps);
static int remote_dep_dequeue_progress(dague_execution_unit_t* eu_context);
#   define remote_dep_init(ctx) remote_dep_dequeue_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_dequeue_on(ctx)
#   define remote_dep_off(ctx)  remote_dep_dequeue_off(ctx)
#   define remote_dep_send(rank, deps) remote_dep_dequeue_send(rank, deps)
#   define remote_dep_progress(ctx) remote_dep_dequeue_progress(ctx)
#   define remote_dep_release(ctx, deps) remote_dep_nothread_release(ctx, deps)
#   define remote_dep_get_datatypes(deps) remote_dep_nothread_get_datatypes(deps)

#else
/* TODO */
#endif 


static void remote_dep_mpi_put_data(remote_dep_wire_get_t* task, int to, int i);
static void remote_dep_mpi_get_data(dague_execution_unit_t* eu_context, remote_dep_wire_activate_t* task, int from, int i);
#ifdef FLOW_CONTROL
static void remote_dep_mpi_short_get_data(dague_context_t* context, int from, int i);
#endif

/* Shared LIFO for the TILES */
dague_atomic_lifo_t* internal_alloc_lifo;
volatile int32_t internal_alloc_lifo_num_used = 0;
static int internal_alloc_lifo_init = 0;

#include "dequeue.h"

typedef enum dep_cmd_action_t
{
    DEP_ACTIVATE,
    DEP_RELEASE,
/*    DEP_PROGRESS,
    DEP_PUT_DATA,*/
    DEP_GET_DATA,
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
        int rank;
        int i;
    } get;
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
    dep_cmd_action_t action;
    dep_cmd_t cmd;
} dep_cmd_item_t;

/* condition variable code starts */
pthread_cond_t mpi_progress_condition_variable;
pthread_mutex_t mpi_progress_mutex;

static inline void init_condition_var(){
    pthread_mutex_init(&mpi_progress_mutex, NULL);
    pthread_cond_init(&mpi_progress_condition_variable, NULL);
}

static inline void sleep_on_condition(const struct timespec *ts){
    pthread_mutex_lock(&mpi_progress_mutex);
    pthread_cond_timedwait(&mpi_progress_condition_variable, &mpi_progress_mutex, ts);
    pthread_mutex_unlock(&mpi_progress_mutex);
}

static inline void signal_condition(void){
    pthread_mutex_lock(&mpi_progress_mutex);
    pthread_cond_signal(&mpi_progress_condition_variable);
    pthread_mutex_unlock(&mpi_progress_mutex);
}
/* condition variable code ends */

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
dague_dequeue_t dep_activate_queue;
volatile int np;
static int dep_enabled;

static void *remote_dep_dequeue_main(dague_context_t* context);

static int remote_dep_dequeue_init(dague_context_t* context)
{
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);

    dague_dequeue_construct(&dep_cmd_queue);
    dague_dequeue_construct(&dep_activate_queue);

    //init_condition_var();

    MPI_Comm_size(MPI_COMM_WORLD, (int*) &np);
    if(1 < np)
    {
        np = 0;
        pthread_create(&dep_thread_id,
                       &thread_attr,
                       (void* (*)(void*))remote_dep_dequeue_main,
                       (void*)context);
        
        while(0 == np); /* wait until the thread inits MPI */
    }
    return np;
}

static int remote_dep_dequeue_on(dague_context_t* context)
{
    if(1 < context->nb_nodes)
    {        
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        item->action = DEP_CTL;
        item->cmd.ctl.enable = 1;
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
    DAGUE_LIST_ITEM_SINGLETON(item);
    dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);
    //signal_condition();
    return 1;
}

static int remote_dep_dequeue_progress(dague_execution_unit_t* eu_context)
{
    dep_cmd_item_t* item;
    
    /* don't while, the thread is starving, let it go right away */
    if(NULL != (item = (dep_cmd_item_t*) dague_dequeue_pop_front(&dep_activate_queue)))
    {
        assert(DEP_RELEASE == item->action);
        remote_dep_nothread_release(eu_context, item->cmd.release.deps);
        free(item);
        return 1;
    }
    return 0;
}

void dague_remote_dep_memcpy(void *dst, dague_arena_chunk_t *src, dague_remote_dep_datatype_t datatype)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    item->action = DEP_MEMCPY;
    item->cmd.memcpy.source = src;
    item->cmd.memcpy.destination = dst;
    item->cmd.memcpy.datatype = datatype;
    AREF(src);
    DAGUE_LIST_ITEM_SINGLETON(item);
    dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);
    //signal_condition();
}

#define YIELD_TIME 5000
#include "bindthread.h"

static void* remote_dep_dequeue_main(dague_context_t* context)
{
    int keep_probing = 1;
    struct timespec ts;
    dep_cmd_item_t* item;
    int ctl;
    int do_nano = 0;
    
    ctl = dague_bindthread(context->nb_cores);
    printf("MPI bound to core %d\n", ctl);
    if(ctl != context->nb_cores) do_nano = 1; 
    np = remote_dep_mpi_init(context);
    
    ts.tv_sec = 0; ts.tv_nsec = YIELD_TIME;

    do {
        while(NULL == (item = (dep_cmd_item_t*) dague_dequeue_pop_front(&dep_cmd_queue)))
        {
            if(dep_enabled)
            {
                remote_dep_mpi_progress(context->execution_units[0]);
            }
/* condition variable code starts */
            //sleep_on_condition(&ts);
/* condition variable code ends */
            if(do_nano) nanosleep(&ts, NULL);
        }

        switch(item->action)
        {
            case DEP_ACTIVATE:
                remote_dep_nothread_send(item->cmd.activate.rank, item->cmd.activate.deps);
                break;
#ifdef FLOW_CONTROL
            case DEP_GET_DATA:
                remote_dep_mpi_short_get_data(context, item->cmd.get.rank, item->cmd.get.i);
                break;
#endif
            case DEP_CTL:
                ctl = item->cmd.ctl.enable;
                assert((ctl * ctl) <= 1);
                if(-1 == ctl)
                {
                    keep_probing = 0;
                }
                if(0 == ctl)
                {
                    remote_dep_mpi_off(context);
                }
                if(1 == ctl)
                {
                    remote_dep_mpi_on(context);
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
    for(k = 0; output_count; k++)
    {
        output_count -= deps->output[k].count;
        if(deps->output[k].rank_bits[rank_bank] & rank_mask)
        {
            msg.which |= (1<<k);
        }
    }
    remote_dep_mpi_send_dep(rank, &msg);
    return 0;
}

static int remote_dep_nothread_get_datatypes(dague_remote_deps_t* origin)
{
    dague_execution_context_t exec_context;

    exec_context.dague_object = dague_object_lookup( origin->msg.object_id );
    exec_context.function = exec_context.dague_object->functions_array[origin->msg.function_id];

    for(int i = 0; i < exec_context.function->nb_locals; i++)
        exec_context.locals[i] = origin->msg.locals[i];

    return exec_context.function->release_deps(NULL, &exec_context,
                                               DAGUE_ACTION_RECV_INIT_REMOTE_DEPS | origin->msg.which,
                                               origin, NULL);
}

static int remote_dep_nothread_release(dague_execution_unit_t* eu_context, dague_remote_deps_t* origin)
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
        if(origin->msg.deps & (1 << i)) {
            //DEBUG(("DATA %p released from %p[%d]\n", GC_DATA(origin->output[i].data), origin, i));
            data[i] = origin->output[i].data;
#ifdef DAGUE_DEBUG
/*            {
                char tmp[128];
                void* _data = ADATA(data[i]);
                DEBUG(("%s: recv %p -> [0] %9.5f [1] %9.5f [2] %9.5f\n",
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

#ifdef DAGUE_PROFILING
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
                                             &MPI_Activate_sk, &MPI_Activate_ek);
    dague_profiling_add_dictionary_keyword( "MPI_DATA_CTL", "fill:#000077",
                                             &MPI_Data_ctl_sk, &MPI_Data_ctl_ek);
    dague_profiling_add_dictionary_keyword( "MPI_DATA_PLD_SND", "fill:#B08080",
                                             &MPI_Data_plds_sk, &MPI_Data_plds_ek);
    dague_profiling_add_dictionary_keyword( "MPI_DATA_PLD_RCV", "fill:#80B080",
                                             &MPI_Data_pldr_sk, &MPI_Data_pldr_ek);
    
    MPIctl_prof = dague_profiling_thread_init( MPI_PROFILING_SIZE, "MPI ctl");
    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        MPIsnd_prof[i] = dague_profiling_thread_init( MPI_PROFILING_SIZE / DEP_NB_CONCURENT, "MPI isend(req=%d)", i);
        MPIrcv_prof[i] = dague_profiling_thread_init( MPI_PROFILING_SIZE / DEP_NB_CONCURENT, "MPI irecv(req=%d)", i);
    }    
}

#define TAKE_TIME(PROF, KEY, I)  dague_profiling_trace((PROF), (KEY), (I))
#else
#define TAKE_TIME(PROF, KEY, I)
#define remote_dep_mpi_profiling_init() do {} while(0)
#endif  /* DAGUE_PROFILING */

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
            DEBUG(("Your MPI implementation defines the maximal TAG value to %d (0x%08x), which might be too small should you have more than %d simultaneous remote dependencies\n",
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

    assert( 0 == internal_alloc_lifo_init );
    internal_alloc_lifo = (dague_atomic_lifo_t*)malloc(sizeof(dague_atomic_lifo_t));
    dague_atomic_lifo_construct( internal_alloc_lifo );
    internal_alloc_lifo_init = 1;
    internal_alloc_lifo_num_used = 0;

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

#ifdef DAGUE_PROFILING
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
    {
        dague_list_item_t* item;
        int nb_allocated_items = 0;
        while( NULL != (item = dague_atomic_lifo_pop(internal_alloc_lifo)) ) {
            free(item);
            nb_allocated_items++;
        }
        free(internal_alloc_lifo);
        internal_alloc_lifo = NULL;
        internal_alloc_lifo_init = 0;
        DEBUG(( "Total number of released TILES = %d\n", nb_allocated_items ));
        printf( "Total number of released TILES = %d\n", nb_allocated_items );
    }
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
    DEBUG(("TO\t%d\tActivate\t%s\ti=na\twith datakey %lx\tparams %lx\n", rank, remote_dep_cmd_to_string(msg, tmp, 128), msg->deps, msg->which));
    
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
        if(flag)
        {
            if(i < DEP_NB_CONCURENT)
            {
                assert(REMOTE_DEP_ACTIVATE_TAG == status.MPI_TAG);
                DEBUG(("FROM\t%d\tActivate\t%s\ti=%d\twith datakey %lx\tparams %lx\n",
                       status.MPI_SOURCE, remote_dep_cmd_to_string(&dep_activate_buff[i]->msg, tmp, 128),
                       i, dep_activate_buff[i]->msg.deps, dep_activate_buff[i]->msg.which));
                remote_dep_mpi_get_data(eu_context, &dep_activate_buff[i]->msg, status.MPI_SOURCE, i);
            } 
            else if(i < (2*DEP_NB_CONCURENT))
            {
                i -= DEP_NB_CONCURENT; /* shift i */
                assert(REMOTE_DEP_GET_DATA_TAG == status.MPI_TAG);
                remote_dep_mpi_put_data(&dep_get_buff[i], status.MPI_SOURCE, i);
            }
            else 
            {
                i -= DEP_NB_CONCURENT * 2;
                assert(i >= 0);
                if(i < (DEP_NB_CONCURENT * MAX_PARAM_COUNT))
                {
                    /* We finished sending the data, allow for more requests 
                     * to be processed */
                    dague_remote_deps_t* deps; 
                    int k;
                    k = i % MAX_PARAM_COUNT;
                    i = i / MAX_PARAM_COUNT;
                    deps = (dague_remote_deps_t*) (uintptr_t) dep_get_buff[i].deps;
                    DEBUG(("TO\tna\tPut END  \tunknown \tj=%d,k=%d\twith datakey %lx\tparams %lx\t(tag=%d)\n",
                           i, k, dep_get_buff[i].deps, dep_get_buff[i].which, status.MPI_TAG));
                    DEBUG_MARK_DTA_MSG_END_SEND(status.MPI_TAG);
                    AUNREF(deps->output[k].data);
                    TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_ek, i);
                    dep_get_buff[i].which ^= (1<<k);
                    if(0 == dep_get_buff[i].which)
                    {
                        MPI_Start(&dep_get_req[i]);
                        remote_dep_dec_flying_messages(eu_context->master_context);
                    }

                    /* remote_deps cleanup */
                    deps->output_sent_count++;
                    if(deps->output_count == deps->output_sent_count)
                    {
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
                }
                else
                {
                    /* We received a data, call the matching release_dep */
                    dague_remote_deps_t* deps;
                    int k;
                    i -= (DEP_NB_CONCURENT * MAX_PARAM_COUNT);
                    assert((i >= 0) && (i < DEP_NB_CONCURENT * MAX_PARAM_COUNT));
                    k = i%MAX_PARAM_COUNT;
                    i = i/MAX_PARAM_COUNT;
                    deps = (dague_remote_deps_t*) (uintptr_t) dep_activate_buff[i];
                    DEBUG(("FROM\t%d\tGet END  \t%s\ti=%d,k=%d\twith datakey %lx\tparams %lx\t(tag=%d)\n", status.MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, 128), i, k, deps->msg.deps, deps->msg.which, status.MPI_TAG));
                    DEBUG_MARK_DTA_MSG_END_RECV(status.MPI_TAG);
                    TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_ek, i+k);
                    deps->msg.deps |= 1<<k;
                    /* TODO: This function will use the previously defined mask (msg->deps) to
                     * activate dependencies. As we keep adding bits into this mask, in case where
                     * we have multiple dependencies some of them will be parsed several times.
                     */
                    remote_dep_release(eu_context, deps);
                    if(deps->msg.which == deps->msg.deps)
                    {
                        MPI_Start(&dep_activate_req[i]);
                    }
                    ret++;
                }
            }
        }
    } while(0/*flag*/);
    return ret;
}

static void remote_dep_mpi_put_data(remote_dep_wire_get_t* task, int to, int i)
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
    DEBUG(("PUT which=%lx\n", task->which));
    for(int k = 0; task->which>>k; k++)
    {
        assert(k < MAX_PARAM_COUNT);
        if(!((1<<k) & task->which)) continue;
        //DEBUG(("%p[%d] %p, %p\n", deps, k, deps->output[k].data, GC_DATA(deps->output[k].data)));
        data = ADATA(deps->output[k].data);
        dtt = deps->output[k].type->opaque_dtt;
#ifdef DAGUE_DEBUG
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG(("TO\t%d\tPut START\tunknown \tj=%d,k=%d\twith datakey %lx at %p type %s\t(tag=%d)\n", to, i, k, task->deps, data, type_name, tag+k));
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
        //DEBUG(("send %p -> [0] %9.5f [1] %9.5f [2] %9.5f\n", data, ((double*)data)[0], ((double*)data)[1], ((double*)data)[2]));
        DEBUG_MARK_DTA_MSG_START_SEND(to, data, tag+k);
    }
}

#ifdef FLOW_CONTROL
#define FLOW_CONTROL_MEM_CONSTRAINT 200
#define ATTEMPTS_STALLS_BEFORE_RESUME 3000000
static int stalls = 0;
static int old_context = -1;

static void remote_dep_mpi_short_get_data(dague_context_t* context, int from, int i)
{
    if(old_context != (int)context->taskstodo)
        {
            old_context = context->taskstodo;
            stalls = ATTEMPTS_STALLS_BEFORE_RESUME;
        } 
    if((internal_alloc_lifo_num_used > FLOW_CONTROL_MEM_CONSTRAINT) && (stalls < ATTEMPTS_STALLS_BEFORE_RESUME))
    {
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        item->action = DEP_GET_DATA;
        item->cmd.get.rank = from;
        item->cmd.get.i = i;
        DAGUE_LIST_ITEM_SINGLETON(item);
        dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);
        stalls++;
        remote_dep_mpi_progress(context->execution_units[0]);
    }
    else
    {
        printf("Stall finished after %d tries, %d of %d arena used\n", stalls, internal_alloc_lifo_num_used, FLOW_CONTROL_MEM_CONSTRAINT);
        remote_dep_mpi_get_data(&dep_activate_buff[i]->msg, from, i);
        old_context = context->taskstodo;
        stalls = 0;
    }
}
#endif

static void remote_dep_mpi_get_data(dague_execution_unit_t* eu_context, remote_dep_wire_activate_t* task, int from, int i)
{
#ifdef DAGUE_DEBUG
    char tmp[128];
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif
#ifdef FLOW_CONTROL
    dague_object_t* object = dague_object_lookup( task->object_id );
    const dague_t* function = object->functions_array[task->function_id];
    int doall = 0;
#endif
    MPI_Datatype dtt;
    remote_dep_wire_get_t msg;
    dague_remote_deps_t* deps = dep_activate_buff[i];
    void* data;
    remote_dep_datakey_t locals;
    DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, (void*)task, task);

    locals = deps->msg.which;
    remote_dep_get_datatypes(deps);
    locals ^= deps->msg.which;
    assert(0 == locals); /* we do not support RO dep backtracking, make sure it doesn't happen yet */
    msg.which = deps->msg.which;
    msg.deps = task->deps;
    msg.tag = NEXT_TAG;
    
    assert(dep_enabled);
    for(int k = 0; msg.which>>k; k++)
    {        
        if((1<<k) & msg.which)
        {
            dtt = deps->output[k].type->opaque_dtt;
            data = deps->output[k].data;
            assert(NULL == data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
            if(NULL == data)
            { 
                /* Why do we still have the internal_alloc_lifo? */
                data = (void*)dague_atomic_lifo_pop( internal_alloc_lifo );
                if( NULL == data ) {
#ifdef FLOW_CONTROL
                    /* basic attempt at flow control */
                    if(! (doall || (internal_alloc_lifo_num_used <= FLOW_CONTROL_MEM_CONSTRAINT) || (stalls >= ATTEMPTS_STALLS_BEFORE_RESUME)) )
                    {
                        /* do it later */
                        printf("TO\t%d\tGet LATER\t%s\tbecause %d>%d\n", from, function->name, internal_alloc_lifo_num_used, FLOW_CONTROL_MEM_CONSTRAINT);
                        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
                        item->action = DEP_GET_DATA;
                        item->cmd.get.rank = from;
                        item->cmd.get.i = i;
                        DAGUE_LIST_ITEM_SINGLETON(item);
                        dague_dequeue_push_back(&dep_cmd_queue, (dague_list_item_t*) item);
                        return;
                    }
                    printf("Malloc a new remote tile (%d used of %d)\n", internal_alloc_lifo_num_used, FLOW_CONTROL_MEM_CONSTRAINT);
#endif
                    data = dague_arena_get(deps->output[k].type);
                    DEBUG(("Malloc new remote tile %p size %zu\n", data, deps->output[k].type->elem_size));
                    assert(data != NULL);
                }
#ifdef FLOW_CONTROL
                doall = 1; /* if we do one, do all */
                dague_atomic_inc_32b(&internal_alloc_lifo_num_used);
#endif    
                deps->output[k].data = data;
            }
#ifdef DAGUE_DEBUG
            MPI_Type_get_name(dtt, type_name, &len);
            DEBUG(("TO\t%d\tGet START\t%s\ti=%d,k=%d\twith datakey %lx at %p type %s extent %d\t(tag=%d)\n", from, remote_dep_cmd_to_string(task, tmp, 128), i, k, task->deps, ADATA(data), type_name, deps->output[k].type->elem_size, NEXT_TAG+k));
#endif
            /*printf("%s:%d Allocate new TILE at %p\n", __FILE__, __LINE__, (void*)GC_DATA(deps->output[k].data));*/
            TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_sk, i+k);
            MPI_Irecv(ADATA(data), 1, 
                      dtt, from, NEXT_TAG + k, dep_comm, 
                      &dep_put_rcv_req[i*MAX_PARAM_COUNT+k]);
            DEBUG_MARK_DTA_MSG_START_RECV(from, data, NEXT_TAG + k);
        }
    }
    TAKE_TIME(MPIctl_prof, MPI_Data_ctl_sk, get);
    MPI_Send(&msg, datakey_count, datakey_dtt, from, 
             REMOTE_DEP_GET_DATA_TAG, dep_comm);
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

#ifdef DAGUE_DEBUG
    for(int k = 0; locals>>k; k++) 
       if((1<<k) & locals)
           DEBUG(("TO\t%d\tGet LOCAL\t%s\ti=%d,k=%d\twith data %lx at %p IS LOCAL\t(tag=%d)\n", from, remote_dep_cmd_to_string(task, tmp, 128), i, k, task->deps, data, NEXT_TAG+k));
#endif
    if(locals) 
    {
        deps->msg.which = locals;
        deps->msg.deps = locals;
        remote_dep_release(eu_context, deps);
        assert(0 == deps->msg.which);
        deps->msg.which = msg.which; /* restore context for real recv deps */
        if(0 == msg.which)
        {
            MPI_Start(&dep_activate_req[i]);
        }
    }
    deps->msg.deps = 0; /* now this is the mask of finished deps */
    INC_NEXT_TAG(MAX_PARAM_COUNT);
}

/* Exported default datatype */
static MPI_Datatype MPI_DAGUE_DEFAULT_DATA_TYPE;
dague_arena_t DAGUE_DEFAULT_DATA_TYPE;

void remote_dep_mpi_create_default_datatype(int tile_size, MPI_Datatype base)
{
    char type_name[MPI_MAX_OBJECT_NAME];
    MPI_Aint lb, ub;
    
    snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_DOUBLE*%d*%d", tile_size, tile_size);
    
    MPI_Type_contiguous(tile_size * tile_size, base, &MPI_DAGUE_DEFAULT_DATA_TYPE);
    MPI_Type_set_name(MPI_DAGUE_DEFAULT_DATA_TYPE, type_name);
    MPI_Type_commit(&MPI_DAGUE_DEFAULT_DATA_TYPE);
    MPI_Type_get_extent(MPI_DAGUE_DEFAULT_DATA_TYPE, &lb, &ub);
    dague_arena_construct(&DAGUE_DEFAULT_DATA_TYPE, ub, 
                          DAGUE_ARENA_ALIGNMENT_SSE, &MPI_DAGUE_DEFAULT_DATA_TYPE);
}
