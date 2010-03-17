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

#define USE_MPI_THREAD
#define DEP_NB_CONCURENT 3

static int remote_dep_mpi_init(dplasma_context_t* context);
static int remote_dep_mpi_fini(dplasma_context_t* context);
static int remote_dep_mpi_on(dplasma_context_t* context);
static int remote_dep_mpi_off(dplasma_context_t* context);
static int remote_dep_mpi_send(const dplasma_execution_context_t* task, int rank, gc_data_t* data);
static int remote_dep_mpi_progress(dplasma_execution_unit_t* eu_context);
static int remote_dep_mpi_release(dplasma_execution_unit_t* eu_context, dplasma_execution_context_t* exec_context, gc_data_t** data);

#if defined(USE_MPI_THREAD)
    static int remote_dep_dequeue_init(dplasma_context_t* context);
    static int remote_dep_dequeue_fini(dplasma_context_t* context);
    static int remote_dep_dequeue_on(dplasma_context_t* context);
    static int remote_dep_dequeue_off(dplasma_context_t* context);
    static int remote_dep_dequeue_send(const dplasma_execution_context_t* task, int rank, gc_data_t* data);
    static int remote_dep_dequeue_progress(dplasma_execution_unit_t* eu_context);
    static int remote_dep_dequeue_release(dplasma_execution_unit_t* eu_context, dplasma_execution_context_t* exec_context, gc_data_t** data);
#   define remote_dep_init(ctx) remote_dep_dequeue_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_dequeue_on(ctx)
#   define remote_dep_off(ctx)  remote_dep_dequeue_off(ctx)
#   define remote_dep_send(task, rank, data) remote_dep_dequeue_send(task, rank, data)
#   define remote_dep_progress(ctx) remote_dep_dequeue_progress(ctx)
#   define remote_dep_release(ctx, task, data) remote_dep_dequeue_release(ctx, task, data)

#else
#   define remote_dep_init(ctx) remote_dep_mpi_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_mpi_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_mpi_on(ctx)
#   define remote_dep_off(ctx)  remote_dep_mpi_off(ctx)
#   define remote_dep_send(task, rank, data) remote_dep_mpi_send(task, rank, data)
#   define remote_dep_progress(ctx) remote_dep_mpi_progress(ctx)
#   define remote_dep_release(ctx, task, data) remote_dep_mpi_release(ctx, task, data)
#endif 

#ifdef DPLASMA_PROFILING
static dplasma_thread_profiling_t* MPIctl_prof;
static dplasma_thread_profiling_t* MPIsnd_prof[DEP_NB_CONCURENT];
static dplasma_thread_profiling_t* MPIrcv_prof[DEP_NB_CONCURENT];
static int MPI_Activate_sk, MPI_Activate_ek;
static int MPI_Data_ctl_sk, MPI_Data_ctl_ek;
static int MPI_Data_plds_sk, MPI_Data_plds_ek;
static int MPI_Data_pldr_sk, MPI_Data_pldr_ek;

#define TAKE_TIME(PROF, KEY, I)  dplasma_profiling_trace((PROF), (KEY), (I))
#else
#define TAKE_TIME(PROF, KEY, I)
#endif  /* DPLASMA_PROFILING */

int remote_dep_transport_init(dplasma_context_t* context)
{
#ifdef DPLASMA_PROFILING
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
#endif /* DPLASMA_PROFILING */
    return remote_dep_init(context);
}

int remote_dep_transport_fini(dplasma_context_t* context)
{
    return remote_dep_fini(context);
}

int dplasma_remote_dep_on(dplasma_context_t* context)
{
    return remote_dep_on(context);
}

int dplasma_remote_dep_off(dplasma_context_t* context)
{
    return remote_dep_off(context);
}

int dplasma_remote_dep_progress(dplasma_execution_unit_t* eu_context)
{
    return remote_dep_progress(eu_context);
}



enum {
    REMOTE_DEP_ACTIVATE_TAG,
    REMOTE_DEP_GET_DATA_TAG,
    REMOTE_DEP_PUT_DATA_TAG,
} dplasma_remote_dep_tag_t;

/* TODO: smart use of dplasma context instead of ugly globals */
static int dep_enabled;
static MPI_Comm dep_comm;
#define DEP_NB_REQ (3 * DEP_NB_CONCURENT + DEP_NB_CONCURENT * MAX_PARAM_COUNT)
static MPI_Request dep_req[DEP_NB_REQ];
static MPI_Request* dep_activate_req    = &dep_req[0 * DEP_NB_CONCURENT];
static MPI_Request* dep_get_req         = &dep_req[1 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_snd_req     = &dep_req[2 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_rcv_req     = &dep_req[3 * DEP_NB_CONCURENT];
/* TODO: fix heterogeneous restriction by using proper mpi datatypes */
#define dep_dtt MPI_BYTE
#define dep_count sizeof(dplasma_execution_context_t)
static dplasma_execution_context_t dep_activate_buff[DEP_NB_CONCURENT];
#define datakey_dtt MPI_LONG_LONG
#define datakey_count 1
static unsigned long long dep_get_buff[DEP_NB_CONCURENT];

#include <limits.h>
#if ULLONG_MAX < UINTPTR_MAX
#error "unsigned long long is not large enough to hold a pointer!"
#endif

static int remote_dep_mpi_init(dplasma_context_t* context)
{
    int i, np;
    MPI_Comm_dup(MPI_COMM_WORLD, &dep_comm);
    MPI_Comm_size(dep_comm, &np);
    
    for(i = 0; i < DEP_NB_REQ; i++)
    {        
        dep_req[i] = MPI_REQUEST_NULL;
    }
    dep_enabled = 0;
    return np;
}

static int remote_dep_mpi_on(dplasma_context_t* context)
{
    int i;

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
        MPI_Recv_init(&dep_activate_buff[i], dep_count, dep_dtt, MPI_ANY_SOURCE, REMOTE_DEP_ACTIVATE_TAG, dep_comm, &dep_activate_req[i]);
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

#define TILE_SIZE (DPLASMA_TILE_SIZE * DPLASMA_TILE_SIZE)

#ifdef CRC_CHECK
#define CRC_PRINT(data, pos) do \
{ \
    double _crc = 0.0f; \
    int _i; \
    for(_i = 0; _i < TILE_SIZE; _i++) \
    { \
        _crc += ((double*) (data))[_i] * _i; \
    } \
    MPI_Comm_rank(MPI_COMM_WORLD, &_i); \
    printf("[%g:" pos "] on %d\n", _crc, _i); \
} while(0)
#else
#define CRC_PRINT(data, pos)
#endif 

static int remote_dep_mpi_release(dplasma_execution_unit_t* eu_context, dplasma_execution_context_t* exec_context, gc_data_t** data)
{
    return exec_context->function->release_deps(eu_context, exec_context, DPLASMA_ACTION_DEPS_MASK, NULL, data);
}


#include <limits.h>
#define PTR_TO_TAG(ptr) ((int) (((intptr_t) ptr) & INT_MAX))

static void remote_dep_mpi_put_data(gc_data_t* data, int to, int i);
static void remote_dep_mpi_get_data(dplasma_execution_context_t* task, int from, int i);

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
                dplasma_execution_context_t* e = &dep_activate_buff[i];
                DEBUG(("FROM\t%d\tActivate\t%s\ti=%d\n", status.MPI_SOURCE, dplasma_service_to_string(e, tmp, 128), i));
                remote_dep_mpi_get_data(e, status.MPI_SOURCE, i);
            } 
            else if(REMOTE_DEP_GET_DATA_TAG == status.MPI_TAG)
            {
                i -= DEP_NB_CONCURENT; /* shift i */
                assert(i >= 0);
                remote_dep_mpi_put_data((gc_data_t*) (intptr_t)dep_get_buff[i], status.MPI_SOURCE, i);
            }
            else 
            {
                i -= DEP_NB_CONCURENT * 2;
                assert(i >= 0);
                if(i < DEP_NB_CONCURENT)
                {
                    /* We finished sending the data, allow for more requests 
                     * to be processed */
                    TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_ek, i);
                    DEBUG(("TO\tna\tPut data\tunknown \tj=%d\tsend of %p (hash %d) complete\n", i, dep_get_buff[i], PTR_TO_TAG(dep_get_buff[i])));
                    gc_data_unref((gc_data_t*) (uintptr_t) dep_get_buff[i]);
                    MPI_Start(&dep_get_req[i]);
                    dplasma_remote_dep_dec_flying_messages(eu_context->master_context);
                }
                else
                {
                    i -= DEP_NB_CONCURENT;
                    assert((i >= 0) && (i < DEP_NB_CONCURENT));
                    /* We received a data, call the matching release_dep */
                    DEBUG(("FROM\t%d\tPut data\tunknown \ti=%d\trecv complete\n", status.MPI_SOURCE, i));
                    CRC_PRINT(((gc_data_t*) dep_activate_buff[i].list_item.cache_friendly_emptiness)->data, "R");
                    TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_ek, i);
                    remote_dep_release(eu_context, &dep_activate_buff[i], 
                                       (gc_data_t**) &dep_activate_buff[i].list_item.cache_friendly_emptiness);
                    MPI_Start(&dep_activate_req[i]);
                    ret++;
                }
            }
        }
    } while(0/*flag*/);
    return ret;
}

static void remote_dep_mpi_put_data(gc_data_t* data, int to, int i)
{
    assert(dep_enabled);
    DEBUG(("TO\t%d\tPut data\tunknown \tj=%d\twith data at %p (hash %d)\n", to, i, data, PTR_TO_TAG(data)));
    TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_sk, i);
    MPI_Isend(data->data, TILE_SIZE, MPI_DOUBLE, to, PTR_TO_TAG(data), dep_comm, &dep_put_snd_req[i]);
}

static int get = 1;

static void remote_dep_mpi_get_data(dplasma_execution_context_t* task, int from, int i)
{
#ifdef DPLASMA_DEBUG
    char tmp[128];
#endif
    unsigned long long datakey = (intptr_t) task->list_item.cache_friendly_emptiness;
    task->list_item.cache_friendly_emptiness = gc_data_new(malloc(sizeof(double) * TILE_SIZE), 1);
    assert(dep_enabled);
    
    DEBUG(("TO\t%d\tGet data\t%s\ti=%d\twith data at %p (hash %d)\n", from, dplasma_service_to_string(task, tmp, 128), i, (void*) (intptr_t) datakey, PTR_TO_TAG(datakey)));
    TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_sk, i);
    MPI_Irecv(((gc_data_t*) task->list_item.cache_friendly_emptiness)->data, TILE_SIZE, 
              MPI_DOUBLE, from, PTR_TO_TAG(datakey), dep_comm, &dep_put_rcv_req[i]);

    TAKE_TIME(MPIctl_prof, MPI_Data_ctl_sk, get);
    MPI_Send(&datakey, datakey_count, datakey_dtt, from, REMOTE_DEP_GET_DATA_TAG, dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Data_ctl_ek, get++);
}

static int act = 1;

/* Send the activate tag */
static int remote_dep_mpi_send(const dplasma_execution_context_t* task, int rank, gc_data_t *data)
{
#ifdef DPLASMA_DEBUG
    char tmp[128];
#endif

    assert(dep_enabled);
    TAKE_TIME(MPIctl_prof, MPI_Activate_sk, act);
    DEBUG(("TO\t%d\tActivate\t%s\ti=na\twith data at %p\n", rank, dplasma_service_to_string(task, tmp, 128), data));
    CRC_PRINT((double**)(data->data), "S");
    
    ((dplasma_execution_context_t*) task)->list_item.cache_friendly_emptiness = data;
    MPI_Send((void*) task, dep_count, dep_dtt, rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, act++);
    
    return 1;
}






#if defined(USE_MPI_THREAD)

#include "dequeue.h"

typedef enum dep_cmd_t
{
    DEP_ACTIVATE,
    DEP_PROGRESS,
    DEP_PUT_DATA,
    DEP_GET_DATA,
    DEP_CTL,
} dep_cmd_t;

typedef union dep_cmd_item_content_t
{
    struct {
        dplasma_execution_context_t origin;
        gc_data_t* data;
        int rank;
    } activate;
    int enable;
} dep_cmd_item_content_t;

typedef struct dep_cmd_item_t
{
    dplasma_list_item_t super;
    dep_cmd_t cmd;
    dep_cmd_item_content_t u;
} dep_cmd_item_t;


pthread_t dep_thread_id;
dplasma_dequeue_t dep_cmd_queue;
dplasma_dequeue_t dep_activate_queue;
volatile int np;

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
        dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        
        cmd->super.list_prev = (dplasma_list_item_t*) cmd;
        cmd->cmd = DEP_CTL;
        cmd->u.enable = 1;
        dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) cmd);
        return 1;
    }
    return 0;
}

static int remote_dep_dequeue_off(dplasma_context_t* context)
{
    if(1 < context->nb_nodes)
    {        
        dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        dplasma_context_t *ret;
        
        cmd->super.list_prev = (dplasma_list_item_t*) cmd;
        cmd->cmd = DEP_CTL;
        cmd->u.enable = 0;
        dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) cmd);
    }
    return 0;
}

static int remote_dep_dequeue_fini(dplasma_context_t* context)
{
    if(1 < context->nb_nodes)
    {        
        dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        dplasma_context_t *ret;
    
        cmd->super.list_prev = (dplasma_list_item_t*) cmd;
        cmd->cmd = DEP_CTL;
        cmd->u.enable = -1;
        dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) cmd);
    
        pthread_join(dep_thread_id, (void**) &ret);
        assert(ret == context);
    }
    return 0;
}



static int remote_dep_dequeue_send(const dplasma_execution_context_t* task, int rank, gc_data_t* data)
{
    dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));

    cmd->super.list_prev = (dplasma_list_item_t*) cmd;
    cmd->cmd = DEP_ACTIVATE;
    cmd->u.activate.origin = *task;
    gc_data_ref(data);
    cmd->u.activate.data = data;
    cmd->u.activate.rank = rank;
    
    dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) cmd);
    return 1;
}


static int remote_dep_dequeue_release(dplasma_execution_unit_t* eu_context, dplasma_execution_context_t* exec_context, gc_data_t** data)
{
    dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    
    cmd->super.list_prev = (dplasma_list_item_t*) cmd;
    cmd->cmd = DEP_ACTIVATE;
    cmd->u.activate.origin = *exec_context;
    cmd->u.activate.data = data[0];
    /* don't fill rank, it's useless */
    
    dplasma_dequeue_push_back(&dep_activate_queue, (dplasma_list_item_t*) cmd);
    return 1;
}


static int remote_dep_dequeue_progress(dplasma_execution_unit_t* eu_context)
{
    dep_cmd_item_t* cmd;
        
    /* don't while, the thread is starving, let it go right away */
    if(NULL != (cmd = (dep_cmd_item_t*) dplasma_dequeue_pop_front(&dep_activate_queue)))
    {
        gc_data_t *data[1];
        data[0] = cmd->u.activate.data;
        remote_dep_mpi_release(eu_context, &cmd->u.activate.origin, data);
        free(cmd);
        return 1;
    }
    return 0;
}

#define YIELD_TIME 5000

static void* remote_dep_dequeue_main(dplasma_context_t* context)
{
    int keep_probing = 1;
    struct timespec ts;
    dep_cmd_item_t* cmd;
    
    np = remote_dep_mpi_init(context);
    
    ts.tv_sec = 0; ts.tv_nsec = YIELD_TIME;
    
    do {
        while(NULL == (cmd = (dep_cmd_item_t*) dplasma_dequeue_pop_front(&dep_cmd_queue)))
        {
            if(dep_enabled)
            {
                remote_dep_mpi_progress(context->execution_units[0]);
            }
            nanosleep(&ts, NULL);
        }
        
        switch(cmd->cmd)
        {                
            case DEP_ACTIVATE:
                remote_dep_mpi_send(&cmd->u.activate.origin, cmd->u.activate.rank, cmd->u.activate.data);
                break;
            case DEP_CTL:
                if(cmd->u.enable == -1)
                {
                    keep_probing = 0;
                    break;
                }
                if(cmd->u.enable == 0)
                {
                    remote_dep_mpi_off(context);
                    break;
                }
                if(cmd->u.enable == 1)
                {
                    remote_dep_mpi_on(context);
                    break;
                }
                assert((cmd->u.enable * cmd->u.enable) <= 1);
            default:
                break;
        }
        free(cmd);
    } while(keep_probing);
    
    remote_dep_mpi_fini(context);
    return context;
}

#endif

#ifdef DEPRECATED
int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                const param_t* origin_param,
                                const dplasma_execution_context_t* exec_context,
                                const param_t* dest_param )
{
    int rank; 
#ifdef DPLASMA_DEBUG
    char tmp[128];
    char tmp2[128];
#endif
    
    rank = dplasma_remote_dep_compute_grid_rank(eu_context, origin, exec_context);
    assert(rank >= 0);
    assert(rank < eu_context->master_context->nb_nodes);
    if(dplasma_remote_dep_is_forwarded(eu_context, rank))
    {    
        return 0;
    }
    dplasma_remote_dep_mark_forwarded(eu_context, rank);
    DEBUG(("%s -> %s\ttrigger REMOTE process rank %d\n", dplasma_service_to_string(origin, tmp2, 128), dplasma_service_to_string(exec_context, tmp, 128), rank ));
    return remote_dep_send(origin, rank, NULL);
}
#endif
