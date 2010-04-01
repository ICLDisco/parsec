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

#define DPLASMA_REMOTE_DEP_USE_THREADS
#define DEP_NB_CONCURENT 3

static int remote_dep_mpi_init(dplasma_context_t* context);
static int remote_dep_mpi_fini(dplasma_context_t* context);
static int remote_dep_mpi_on(dplasma_context_t* context);
static int remote_dep_mpi_off(dplasma_context_t* context);
static int remote_dep_mpi_send_dep(int rank, remote_dep_wire_activate_t* msg);
static int remote_dep_mpi_progress(dplasma_execution_unit_t* eu_context);


static int remote_dep_nothread_send(int rank, dplasma_remote_deps_t* deps);
static int remote_dep_nothread_get_datatypes(dplasma_remote_deps_t* origin);
static int remote_dep_nothread_release(dplasma_execution_unit_t* eu_context, dplasma_remote_deps_t* origin);
static int remote_dep_nothread_memcpy(void *dst, gc_data_t *src, const dplasma_remote_dep_datatype_t datatype);

#ifdef DPLASMA_REMOTE_DEP_USE_THREADS
static int remote_dep_dequeue_init(dplasma_context_t* context);
static int remote_dep_dequeue_fini(dplasma_context_t* context);
static int remote_dep_dequeue_on(dplasma_context_t* context);
static int remote_dep_dequeue_off(dplasma_context_t* context);
static int remote_dep_dequeue_send(int rank, dplasma_remote_deps_t* deps);
static int remote_dep_dequeue_progress(dplasma_execution_unit_t* eu_context);
static int remote_dep_dequeue_release(dplasma_execution_unit_t* eu_context, dplasma_remote_deps_t* origin);
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


#include "dequeue.h"

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
        dplasma_remote_deps_t* deps;
    } activate;
    struct {
        dplasma_remote_deps_t* deps;
    } release;
    struct {
        int enable;        
    } ctl;
    struct {
        gc_data_t *source;
        void *destination;
        dplasma_remote_dep_datatype_t datatype;
    } memcpy;
} dep_cmd_t;

typedef struct dep_cmd_item_t
{
    dplasma_list_item_t super;
    dep_cmd_action_t action;
    dep_cmd_t cmd;
} dep_cmd_item_t;

static char* remote_dep_cmd_to_string(remote_dep_wire_activate_t* origin, char* str, size_t len)
{
    int i, index = 0;
    dplasma_t* function = (dplasma_t*) (uintptr_t) origin->function;
    
    index += snprintf( str + index, len - index, "%s", function->name );
    if( index >= len ) return str;
    for( i = 0; i < function->nb_locals; i++ ) {
        index += snprintf( str + index, len - index, "_%d",
                           origin->locals[i].value );
        if( index >= len ) return str;
    }
    return str;
}

pthread_t dep_thread_id;
dplasma_dequeue_t dep_cmd_queue;
dplasma_dequeue_t dep_activate_queue;
volatile int np;
static int dep_enabled;

static void *remote_dep_dequeue_main(dplasma_context_t* context);

static int remote_dep_dequeue_init(dplasma_context_t* context)
{
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
    
    
    dplasma_dequeue_construct(&dep_cmd_queue);
    dplasma_dequeue_construct(&dep_activate_queue);
    
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

static int remote_dep_dequeue_on(dplasma_context_t* context)
{
    if(1 < context->nb_nodes)
    {        
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        item->action = DEP_CTL;
        item->cmd.ctl.enable = 1;
        DPLASMA_LIST_ITEM_SINGLETON(item);
        dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) item);
        return 1;
    }
    return 0;
}

static int remote_dep_dequeue_off(dplasma_context_t* context)
{
    if(1 < context->nb_nodes)
    {        
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        dplasma_context_t *ret;
        item->action = DEP_CTL;
        item->cmd.ctl.enable = 0;
        DPLASMA_LIST_ITEM_SINGLETON(item);
        dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) item);
    }
    return 0;
}

static int remote_dep_dequeue_fini(dplasma_context_t* context)
{
    if(1 < context->nb_nodes)
    {        
        dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        dplasma_context_t *ret;
        item->action = DEP_CTL;
        item->cmd.ctl.enable = -1;
        DPLASMA_LIST_ITEM_SINGLETON(item);
        dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) item);
        
        pthread_join(dep_thread_id, (void**) &ret);
        assert(ret == context);
    }
    return 0;
}



static int remote_dep_dequeue_send(int rank, dplasma_remote_deps_t* deps)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    item->action = DEP_ACTIVATE;
    item->cmd.activate.rank = rank;
    item->cmd.activate.deps = deps;
    DPLASMA_LIST_ITEM_SINGLETON(item);
    dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) item);
    return 1;
}


static int remote_dep_dequeue_release(dplasma_execution_unit_t* eu_context, dplasma_remote_deps_t* origin)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    item->action = DEP_RELEASE;
    item->cmd.release.deps = origin;
    DPLASMA_LIST_ITEM_SINGLETON(item);
    dplasma_dequeue_push_back(&dep_activate_queue, (dplasma_list_item_t*) item);
    return 1;
}


static int remote_dep_dequeue_progress(dplasma_execution_unit_t* eu_context)
{
    dep_cmd_item_t* item;
    
    /* don't while, the thread is starving, let it go right away */
    if(NULL != (item = (dep_cmd_item_t*) dplasma_dequeue_pop_front(&dep_activate_queue)))
    {
        assert(DEP_RELEASE == item->action);
        remote_dep_nothread_release(eu_context, item->cmd.release.deps);
        free(item);
        return 1;
    }
    return 0;
}

void dplasma_remote_dep_memcpy(void *dst, gc_data_t *src, dplasma_remote_dep_datatype_t datatype)
{
    dep_cmd_item_t* item = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    item->action = DEP_MEMCPY;
    item->cmd.memcpy.source = src;
    item->cmd.memcpy.destination = dst;
    item->cmd.memcpy.datatype = datatype;
    gc_data_ref(src);
    DPLASMA_LIST_ITEM_SINGLETON(item);
    dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) item);
}

#define YIELD_TIME 5000

static void* remote_dep_dequeue_main(dplasma_context_t* context)
{
    int keep_probing = 1;
    struct timespec ts;
    dep_cmd_item_t* item;
    int ctl;
    
    np = remote_dep_mpi_init(context);
    
    ts.tv_sec = 0; ts.tv_nsec = YIELD_TIME;
    
    do {
        while(NULL == (item = (dep_cmd_item_t*) dplasma_dequeue_pop_front(&dep_cmd_queue)))
        {
            if(dep_enabled)
            {
                remote_dep_mpi_progress(context->execution_units[0]);
            }
            nanosleep(&ts, NULL);
        }

        switch(item->action)
        {
            case DEP_ACTIVATE:
                remote_dep_nothread_send(item->cmd.activate.rank, item->cmd.activate.deps);
                break;
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


static int remote_dep_nothread_send(int rank, dplasma_remote_deps_t* deps)
{
    int k;
    int rank_bank = rank / (sizeof(uint32_t) * 8);
    uint32_t rank_mask = 1 << (rank % (sizeof(uint32_t) * 8));
    int output_count = deps->output_count;

    deps->msg.which = 0;
    for(k = 0; output_count; k++)
    {
        output_count -= deps->output[k].count;
        if(deps->output[k].rank_bits[rank_bank] & rank_mask)
        {
            deps->msg.which |= (1<<k);
        }
    }
    remote_dep_mpi_send_dep(rank, &deps->msg);
}

static int remote_dep_nothread_get_datatypes(dplasma_remote_deps_t* origin)
{
    int ret;
    int actions;
    dplasma_execution_context_t exec_context;
    
    exec_context.function = (dplasma_t*) (uintptr_t) origin->msg.function;
    for(int i = 0; i < exec_context.function->nb_locals; i++)
        exec_context.locals[i] = origin->msg.locals[i];
    
    actions = (DPLASMA_ACTION_GETTYPE_REMOTE_DEPS | DPLASMA_ACTION_DEPS_MASK);
    
    ret = exec_context.function->release_deps(NULL, &exec_context, actions, origin, NULL);
    return ret;
}

static int remote_dep_nothread_release(dplasma_execution_unit_t* eu_context, dplasma_remote_deps_t* origin)
{
    int ret;
    dplasma_execution_context_t exec_context;
    gc_data_t* data[MAX_PARAM_COUNT];
    
    exec_context.function = (dplasma_t*) (uintptr_t) origin->msg.function;
    for(int i = 0; i < exec_context.function->nb_locals; i++)
        exec_context.locals[i] = origin->msg.locals[i];
    for(int i = 0; origin->msg.deps >> i; i++)
    {
        if(origin->msg.deps & (1 << i))
        {
            assert(origin->msg.which & (1 << i));
            data[i] = origin->output[i].data;
        }
    }
    ret = exec_context.function->release_deps(eu_context, &exec_context, 
                                              DPLASMA_ACTION_RELEASE_LOCAL_DEPS | origin->msg.deps, 
                                              NULL, data);
    origin->msg.which ^= origin->msg.deps;
    origin->msg.deps = 0;
    return ret;
}

static int remote_dep_nothread_memcpy(void *dst, gc_data_t *src, 
                                      const dplasma_remote_dep_datatype_t datatype)
{
    /* TODO: split the mpi part */
    MPI_Sendrecv(GC_DATA(src), 1, datatype, 0, 0,
                 dst, 1, datatype, 0, 0,
                 MPI_COMM_SELF, MPI_STATUS_IGNORE);
    gc_data_unref(src);
}





/******************************************************************************/
/* ALL MPI SPECIFIC CODE GOES HERE 
/******************************************************************************/
enum {
    REMOTE_DEP_ACTIVATE_TAG,
    REMOTE_DEP_GET_DATA_TAG,
    REMOTE_DEP_PUT_DATA_TAG,
    REMOTE_DEP_MAX_CTRL_TAG
} dplasma_remote_dep_tag_t;

/* Exported default datatype */
MPI_Datatype DPLASMA_DEFAULT_DATA_TYPE;

#ifdef DPLASMA_PROFILING
static dplasma_thread_profiling_t* MPIctl_prof;
static dplasma_thread_profiling_t* MPIsnd_prof[DEP_NB_CONCURENT];
static dplasma_thread_profiling_t* MPIrcv_prof[DEP_NB_CONCURENT];
static int MPI_Activate_sk, MPI_Activate_ek;
static int MPI_Data_ctl_sk, MPI_Data_ctl_ek;
static int MPI_Data_plds_sk, MPI_Data_plds_ek;
static int MPI_Data_pldr_sk, MPI_Data_pldr_ek;

static void remote_dep_mpi_profiling_init(void)
{
    int i;
    
    dplasma_profiling_add_dictionary_keyword( "MPI_ACTIVATE", "fill:#FF0000",
                                             &MPI_Activate_sk, &MPI_Activate_ek);
    dplasma_profiling_add_dictionary_keyword( "MPI_DATA_CTL", "fill:#000077",
                                             &MPI_Data_ctl_sk, &MPI_Data_ctl_ek);
    dplasma_profiling_add_dictionary_keyword( "MPI_DATA_PLD_SND", "fill:#B08080",
                                             &MPI_Data_plds_sk, &MPI_Data_plds_ek);
    dplasma_profiling_add_dictionary_keyword( "MPI_DATA_PLD_RCV", "fill:#80B080",
                                             &MPI_Data_pldr_sk, &MPI_Data_pldr_ek);
    
    MPIctl_prof = dplasma_profiling_thread_init( 4096, "MPI ctl");
    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        MPIsnd_prof[i] = dplasma_profiling_thread_init( 4096 / DEP_NB_CONCURENT, "MPI isend(req=%d)", i);
        MPIrcv_prof[i] = dplasma_profiling_thread_init( 4096 / DEP_NB_CONCURENT, "MPI irecv(req=%d)", i);
    }    
}

#define TAKE_TIME(PROF, KEY, I)  dplasma_profiling_trace((PROF), (KEY), (I))
#else
#define TAKE_TIME(PROF, KEY, I)
#define remote_dep_mpi_profiling_init() do {} while(0)
#endif  /* DPLASMA_PROFILING */

/* TODO: smart use of dplasma context instead of ugly globals */
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
static dplasma_remote_deps_t* dep_activate_buff[DEP_NB_CONCURENT];
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

/* This function hashes a tag from a pointer. There is no collision because the 
 * pointers used to feed it are coming from remote_deps arrays, either from 
 * static here, or from free_list (a consecutive array as well). Therefore 
 * we should see no collision until we reach in the order of 
 * MAX_MPI_TAG/MAX_PARAM_COUNT simultaneous flying messages
 */
/*#define PTR_TO_TAG(ptr) ((int) (((((intptr_t) ptr) / sizeof(dplasma_remote_deps_t)) * MAX_PARAM_COUNT) % MAX_MPI_TAG))*/

static int remote_dep_mpi_init(dplasma_context_t* context)
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
#if defined( DPLASMA_DEBUG )
        if( MAX_MPI_TAG < INT_MAX ) {
            DEBUG(("Your MPI implementation defines the maximal TAG value to %d (0x%08x), which might be too small should you have more than %d simultaneous remote dependencies\n",
                    MAX_MPI_TAG, (unsigned int)MAX_MPI_TAG, MAX_MPI_TAG / MAX_PARAM_COUNT));
        }
#endif
    }

    MPI_Comm_size(dep_comm, &np);
    
    for(i = 0; i < DEP_NB_REQ; i++)
    {        
        dep_req[i] = MPI_REQUEST_NULL;
    }
    dep_enabled = 0;
    remote_dep_mpi_profiling_init();
    return np;
}

static int remote_dep_mpi_on(dplasma_context_t* context)
{
    int i;

    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        dep_activate_buff[i] = remote_deps_allocation(&remote_deps_freelist);
    }

#ifdef DPLASMA_PROFILING
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

static int remote_dep_mpi_off(dplasma_context_t* context)
{
    int i;

    assert(dep_enabled == 1);

    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        MPI_Cancel(&dep_activate_req[i]); MPI_Request_free(&dep_activate_req[i]);
        MPI_Cancel(&dep_get_req[i]); MPI_Request_free(&dep_get_req[i]);
    }
    for(i = 0; i < DEP_NB_REQ; i++)
    {
        assert(MPI_REQUEST_NULL == dep_req[i]);
    }
    return dep_enabled = 0;
}

static int remote_dep_mpi_fini(dplasma_context_t* context)
{
    if(dep_enabled) remote_dep_mpi_off(context);
    MPI_Comm_free(&dep_comm);
    return 0;
}


static int act = 1;

/* Send the activate tag */
static int remote_dep_mpi_send_dep(int rank, remote_dep_wire_activate_t* msg)
{
#ifdef DPLASMA_DEBUG
    char tmp[128];
#endif
    
    assert(dep_enabled);
    TAKE_TIME(MPIctl_prof, MPI_Activate_sk, act);
    DEBUG(("TO\t%d\tActivate\t%s\ti=na\twith datakey %lx\n", rank, remote_dep_cmd_to_string(msg, tmp, 128), msg->deps));
    
    MPI_Send((void*) msg, dep_count, dep_dtt, rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);

    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, act++);
    DEBUG_MARK_CTL_MSG_ACTIVATE_SENT(rank, (void*)msg, msg);

#if defined(DPLASMA_STATS)
    {
        MPI_Aint lb, size;
        MPI_Type_get_extent(dep_dtt, &lb, &size);
        DPLASMA_STATACC_ACCUMULATE(counter_control_messages_sent, 1);
        DPLASMA_STATACC_ACCUMULATE(counter_bytes_sent, size * dep_count);
    }
#endif

    return 1;
}


static void remote_dep_mpi_put_data(remote_dep_wire_get_t* task, int to, int i);
static void remote_dep_mpi_get_data(remote_dep_wire_activate_t* task, int from, int i);

static int remote_dep_mpi_progress(dplasma_execution_unit_t* eu_context)
{
#ifdef DPLASMA_DEBUG
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
            if(REMOTE_DEP_ACTIVATE_TAG == status.MPI_TAG)
            {
                DEBUG(("FROM\t%d\tActivate\t%s\ti=%d\twith datakey %lx\n", status.MPI_SOURCE, remote_dep_cmd_to_string(&dep_activate_buff[i]->msg, tmp, 128), i, dep_activate_buff[i]->msg.deps));
                remote_dep_mpi_get_data(&dep_activate_buff[i]->msg, status.MPI_SOURCE, i);
            } 
            else if(REMOTE_DEP_GET_DATA_TAG == status.MPI_TAG)
            {
                i -= DEP_NB_CONCURENT; /* shift i */
                assert(i >= 0);
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
                    dplasma_remote_deps_t* deps; 
                    int k;
                    k = i % MAX_PARAM_COUNT;
                    i = i / MAX_PARAM_COUNT;
                    deps = (dplasma_remote_deps_t*) (uintptr_t) dep_get_buff[i].deps;
                    DEBUG(("TO\tna\tPut END  \tunknown \tj=%d,k=%d\twith datakey %lx\t(tag=%d)\n", i, k, dep_get_buff[i].deps, status.MPI_TAG));
                    DEBUG_MARK_DTA_MSG_END_SEND(status.MPI_TAG);
                    gc_data_unref(deps->output[k].data);
                    //TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_ek, i);
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
                        int count;

                        k = 0;
                        count = 0;
                        while( count < deps->output_count ) {
                            for(int a = 0; a < (max_nodes_number + 31)/32; a++)
                                deps->output[k].rank_bits[a] = 0;
                            count += deps->output[k].count;
                            deps->output[k].count = 0;
#if defined(DPLASMA_DEBUG)
                            deps->output[k].data = NULL;
                            deps->output[k].type = NULL;
#endif

                        }
                        deps->output_count = 0;
                        deps->output_sent_count = 0;
#if defined(DPLASMA_DEBUG)
                        memset( &deps->msg, 0, sizeof(remote_dep_wire_activate_t) );
#endif
                        dplasma_atomic_lifo_push(deps->origin, 
                                                 dplasma_list_item_singleton((dplasma_list_item_t*) deps));
                    }
                }
                else
                {
                    /* We received a data, call the matching release_dep */
                    dplasma_remote_deps_t* deps;
                    int k;
                    i -= (DEP_NB_CONCURENT * MAX_PARAM_COUNT);
                    assert((i >= 0) && (i < DEP_NB_CONCURENT * MAX_PARAM_COUNT));
                    k = i%MAX_PARAM_COUNT;
                    i = i/MAX_PARAM_COUNT;
                    deps = (dplasma_remote_deps_t*) (uintptr_t) dep_activate_buff[i];
                    DEBUG(("FROM\t%d\tGet END  \t%s\ti=%d,k=%d\twith datakey na\t(tag=%d)\n", status.MPI_SOURCE, remote_dep_cmd_to_string(&deps->msg, tmp, 128), i, k, status.MPI_TAG));
                    DEBUG_MARK_DTA_MSG_END_RECV(status.MPI_TAG);
                    TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_ek, i);
                    deps->msg.deps |= 1<<k;
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
    dplasma_remote_deps_t* deps = (dplasma_remote_deps_t*) (uintptr_t) task->deps;
    dplasma_t* function = (dplasma_t*) (uintptr_t) deps->msg.function;
    int tag = task->tag;
    void* data;
    MPI_Datatype dtt;
#ifdef DPLASMA_DEBUG
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif

    DEBUG_MARK_CTL_MSG_GET_RECV(to, (void*)task, task);

    assert(dep_enabled);
    assert(task->which);

    for(int k = 0; task->which>>k; k++)
    {
        assert(k < MAX_PARAM_COUNT);
        if(!((1<<k) & task->which)) continue;
        data = GC_DATA(deps->output[k].data);
        dtt = *deps->output[k].type;
#ifdef DPLASMA_DEBUG
        MPI_Type_get_name(dtt, type_name, &len);
        DEBUG(("TO\t%d\tPut START\tunknown \tj=%d,k=%d\twith data %d at %p type %s\t(tag=%d)\n", to, i, k, task->deps, data, type_name, tag+k));
#endif

#if defined(DPLASMA_STATS)
        {
            MPI_Aint lb, size;
            MPI_Type_get_extent(dtt, &lb, &size);
            DPLASMA_STATACC_ACCUMULATE(counter_data_messages_sent, 1);
            DPLASMA_STATACC_ACCUMULATE(counter_bytes_sent, size);
        }
#endif

        TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_sk, i);
        MPI_Isend(data, 1, dtt, to, tag + k, dep_comm, &dep_put_snd_req[i*MAX_PARAM_COUNT+k]);
        DEBUG_MARK_DTA_MSG_START_SEND(to, data, tag+k);
    }
}

static int get = 1;

static void remote_dep_mpi_get_data(remote_dep_wire_activate_t* task, int from, int i)
{
#ifdef DPLASMA_DEBUG
    char tmp[128];
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif
    MPI_Aint lb, size;
    MPI_Datatype dtt;
    remote_dep_wire_get_t msg;
    dplasma_t* function = (dplasma_t*) (uintptr_t) task->function;
    dplasma_remote_deps_t* deps = dep_activate_buff[i];

    DEBUG_MARK_CTL_MSG_ACTIVATE_RECV(from, (void*)task, task);

    msg.deps =  task->deps;
    msg.which = task->which;
    msg.tag = NEXT_TAG;
    
    remote_dep_get_datatypes(deps);
    
    assert(dep_enabled);
    for(int k = 0; task->which>>k; k++)
    {        
        if((1<<k) & msg.which)
        {
            dtt = *deps->output[k].type;
            MPI_Type_get_extent(dtt, &lb, &size);
#ifdef DPLASMA_DEBUG
            MPI_Type_get_name(dtt, type_name, &len);
            DEBUG(("TO\t%d\tGet START\t%s\ti=%d,k=%d\twith data %lx type %s extent %d\t(tag=%d)\n", from, remote_dep_cmd_to_string(task, tmp, 128), i, k, task->deps, type_name, size, NEXT_TAG+k));
#endif
#if defined(DPLASMA_STATS)
            /* The hack "size>0 ? size : 1" is for statistics, so that we can store 
             * the size of the pointed data into the cache_friendliness pointer.
             * In case of a 0-sized object, we pass 1 to force the creation of a
             * garbage collectable pointer
             */            
            assert( size < 0xffffffff );
            deps->output[k].data = gc_data_new(malloc(size), size > 0 ? (uint32_t)size : 1); 
#else
            deps->output[k].data = gc_data_new(malloc(size), 1); 
#endif
            TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_sk, i+k);
            MPI_Irecv(GC_DATA(deps->output[k].data), 1, 
                      dtt, from, NEXT_TAG + k, dep_comm, 
                      &dep_put_rcv_req[i*MAX_PARAM_COUNT+k]);
            DEBUG_MARK_DTA_MSG_START_RECV(from, GC_DATA(deps->output[k].data), NEXT_TAG + k);
        }
    }
    INC_NEXT_TAG(MAX_PARAM_COUNT);
    task->deps = 0; /* now this is the mask of finished deps */
    TAKE_TIME(MPIctl_prof, MPI_Data_ctl_sk, get);
    MPI_Send(&msg, datakey_count, datakey_dtt, from, 
             REMOTE_DEP_GET_DATA_TAG, dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Data_ctl_ek, get++);
    DEBUG_MARK_CTL_MSG_GET_SENT(from, (void*)&msg, &msg);

#if defined(DPLASMA_STATS)
    {
        MPI_Aint lb, size;
        MPI_Type_get_extent(datakey_dtt, &lb, &size);
        DPLASMA_STATACC_ACCUMULATE(counter_control_messages_sent, 1);
        DPLASMA_STATACC_ACCUMULATE(counter_bytes_sent, size * datakey_count);
    }
#endif
}

void remote_dep_mpi_create_default_datatype(int tile_size, MPI_Datatype base)
{
    char type_name[MPI_MAX_OBJECT_NAME];
    
    snprintf(type_name, MPI_MAX_OBJECT_NAME, "Default MPI_DOUBLE*%d*%d", tile_size, tile_size);
    
    MPI_Type_contiguous(tile_size * tile_size, base, &DPLASMA_DEFAULT_DATA_TYPE);
    MPI_Type_set_name(DPLASMA_DEFAULT_DATA_TYPE, type_name);
    MPI_Type_commit(&DPLASMA_DEFAULT_DATA_TYPE);
}
