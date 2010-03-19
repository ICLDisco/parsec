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

union dep_cmd_item_content_t;

static int remote_dep_mpi_init(dplasma_context_t* context);
static int remote_dep_mpi_fini(dplasma_context_t* context);
static int remote_dep_mpi_on(dplasma_context_t* context);
static int remote_dep_mpi_off(dplasma_context_t* context);
static int remote_dep_mpi_send(union dep_cmd_item_content_t* cmdu);
static int remote_dep_mpi_progress(dplasma_execution_unit_t* eu_context);
static int remote_dep_mpi_release(dplasma_execution_unit_t* eu_context, union dep_cmd_item_content_t* cmdu);

static int remote_dep_dequeue_init(dplasma_context_t* context);
static int remote_dep_dequeue_fini(dplasma_context_t* context);
static int remote_dep_dequeue_on(dplasma_context_t* context);
static int remote_dep_dequeue_off(dplasma_context_t* context);
static int remote_dep_dequeue_send(int rank, dplasma_remote_deps_t* deps);
static int remote_dep_dequeue_progress(dplasma_execution_unit_t* eu_context);
static int remote_dep_dequeue_release(dplasma_execution_unit_t* eu_context, union dep_cmd_item_content_t* cmdu);
#   define remote_dep_init(ctx) remote_dep_dequeue_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_fini(ctx)
#   define remote_dep_on(ctx)   remote_dep_dequeue_on(ctx)
#   define remote_dep_off(ctx)  remote_dep_dequeue_off(ctx)
#   define remote_dep_send(rank, deps) remote_dep_dequeue_send(rank, deps)
#   define remote_dep_progress(ctx) remote_dep_dequeue_progress(ctx)
#   define remote_dep_release(ctx, cmdu) remote_dep_dequeue_release(ctx, cmdu)

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



#include "dequeue.h"

typedef enum dep_cmd_t
{
    DEP_ACTIVATE,
    DEP_PROGRESS,
    DEP_PUT_DATA,
    DEP_GET_DATA,
    DEP_CTL,
    DEP_MEMCPY,
} dep_cmd_t;

typedef unsigned long remote_dep_datakey_t;

typedef union dep_cmd_item_content_t
{
    struct {
        int rank;
        dplasma_t* function;
        assignment_t locals[MAX_LOCAL_COUNT];
        gc_data_t* data[MAX_PARAM_COUNT];
    } activate;
    struct {
        gc_data_t *source;
        void *destination;
        MPI_Datatype datatype;
    } memcpy;
    int enable;
} dep_cmd_item_content_t;

typedef struct dep_cmd_item_t
{
    dplasma_list_item_t super;
    dep_cmd_t cmd;
    dep_cmd_item_content_t u;
} dep_cmd_item_t;

static char* remote_dep_cmd_to_string(dep_cmd_item_content_t* cmdu, char* str, size_t len)
{
    int i, index = 0;
    
    index += snprintf( str + index, len - index, "%s", cmdu->activate.function->name );
    if( index >= len ) return str;
    for( i = 0; i < cmdu->activate.function->nb_locals; i++ ) {
        index += snprintf( str + index, len - index, "_%d",
                          cmdu->activate.locals[i].value );
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
        dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        cmd->cmd = DEP_CTL;
        cmd->u.enable = 1;
        DPLASMA_LIST_ITEM_SINGLETON(cmd);
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
        cmd->cmd = DEP_CTL;
        cmd->u.enable = 0;
        DPLASMA_LIST_ITEM_SINGLETON(cmd);
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
        cmd->cmd = DEP_CTL;
        cmd->u.enable = -1;
        DPLASMA_LIST_ITEM_SINGLETON(cmd);
        dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) cmd);
        
        pthread_join(dep_thread_id, (void**) &ret);
        assert(ret == context);
    }
    return 0;
}



static int remote_dep_dequeue_send(int rank, dplasma_remote_deps_t* deps)
{
    dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    cmd->cmd = DEP_ACTIVATE;
    cmd->u.activate.rank = rank;
    cmd->u.activate.function = deps->exec_context->function; /* TODO use index */
    for(int i = 0; i < MAX_LOCAL_COUNT; i++)
        cmd->u.activate.locals[i] = deps->exec_context->locals[i];
    for(int i = 0; i < MAX_PARAM_COUNT; i++)
        cmd->u.activate.data[i] = deps->output[i].data;
    DPLASMA_LIST_ITEM_SINGLETON(cmd);
    dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) cmd);
    return 1;
}


static int remote_dep_dequeue_release(dplasma_execution_unit_t* eu_context, dep_cmd_item_content_t* cmdu)
{
    dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    cmd->cmd = DEP_ACTIVATE;
    /* don't fill rank, it's useless */
    cmd->u.activate.function = cmdu->activate.function;
    for(int i = 0; i < MAX_LOCAL_COUNT; i++)
        cmd->u.activate.locals[i] = cmdu->activate.locals[i];
    for(int i = 0; i < MAX_PARAM_COUNT; i++)
        cmd->u.activate.data[i] = cmdu->activate.data[i];
    DPLASMA_LIST_ITEM_SINGLETON(cmd);
    dplasma_dequeue_push_back(&dep_activate_queue, (dplasma_list_item_t*) cmd);
    return 1;
}


static int remote_dep_dequeue_progress(dplasma_execution_unit_t* eu_context)
{
    dep_cmd_item_t* cmd;
    
    /* don't while, the thread is starving, let it go right away */
    if(NULL != (cmd = (dep_cmd_item_t*) dplasma_dequeue_pop_front(&dep_activate_queue)))
    {
        assert(DEP_ACTIVATE == cmd->cmd);
        remote_dep_mpi_release(eu_context, &cmd->u);
        free(cmd);
        return 1;
    }
    return 0;
}

void dplasma_remote_dep_memcpy(void *dst, gc_data_t *src, MPI_Datatype datatype)
{
    dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
    
    cmd->super.list_prev = (dplasma_list_item_t*) cmd;
    cmd->cmd = DEP_MEMCPY;
    cmd->u.memcpy.source = src;
    cmd->u.memcpy.destination = dst;
    cmd->u.memcpy.datatype = datatype;
    /* don't fill rank, it's useless */
    
    gc_data_ref( cmd->u.memcpy.source );
    dplasma_dequeue_push_back(&dep_activate_queue, (dplasma_list_item_t*) cmd);
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
                remote_dep_mpi_send(&cmd->u);
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
                break;
            case DEP_MEMCPY:
                MPI_Sendrecv( GC_DATA(cmd->u.memcpy.source), 1, cmd->u.memcpy.datatype, 0, 0,
                              cmd->u.memcpy.destination, 1, cmd->u.memcpy.datatype, 0, 0,
                              MPI_COMM_SELF, MPI_STATUS_IGNORE );
                gc_data_unref( cmd->u.memcpy.source );
                break;
            default:
                break;
        }
        free(cmd);
    } while(keep_probing);
    
    remote_dep_mpi_fini(context);
    return context;
}



/******************************************************************************/
/* ALL MPI SPECIFIC CODE GOES HERE 
/******************************************************************************/
enum {
    REMOTE_DEP_ACTIVATE_TAG,
    REMOTE_DEP_GET_DATA_TAG,
    REMOTE_DEP_PUT_DATA_TAG
} dplasma_remote_dep_tag_t;

/* TODO: smart use of dplasma context instead of ugly globals */
static MPI_Comm dep_comm;
#define DEP_NB_REQ (3 * DEP_NB_CONCURENT + DEP_NB_CONCURENT * MAX_PARAM_COUNT)
static MPI_Request dep_req[DEP_NB_REQ];
static MPI_Request* dep_activate_req    = &dep_req[0 * DEP_NB_CONCURENT];
static MPI_Request* dep_get_req         = &dep_req[1 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_snd_req     = &dep_req[2 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_rcv_req     = &dep_req[3 * DEP_NB_CONCURENT];
/* TODO: fix heterogeneous restriction by using proper mpi datatypes */
#define dep_dtt MPI_BYTE
#define dep_count sizeof(dep_cmd_item_content_t)
static dep_cmd_item_content_t dep_activate_buff[DEP_NB_CONCURENT];
#define datakey_dtt MPI_LONG
#define datakey_count 1
static remote_dep_datakey_t dep_get_buff[DEP_NB_CONCURENT];

#include <limits.h>
#if ULONG_MAX < UINTPTR_MAX
#error "unsigned long is not large enough to hold a pointer!"
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
    remote_dep_mpi_profiling_init();
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

static int remote_dep_mpi_release(dplasma_execution_unit_t* eu_context, dep_cmd_item_content_t* cmdu)
{
    dplasma_execution_context_t exec_context;
    gc_data_t* data[MAX_PARAM_COUNT];
    
    exec_context.function = cmdu->activate.function;
    for(int i = 0; i < MAX_LOCAL_COUNT; i++)
        exec_context.locals[i] = cmdu->activate.locals[i];
    for(int i = 0; i < MAX_PARAM_COUNT; i++)
        data[i] = cmdu->activate.data[i];
    return exec_context.function->release_deps(eu_context, &exec_context, DPLASMA_ACTION_DEPS_MASK, NULL, data);
}


#include <limits.h>
#define PTR_TO_TAG(ptr) ((int) (((intptr_t) ptr) & INT_MAX))

static void remote_dep_mpi_put_data(gc_data_t* data, int to, int i);
static void remote_dep_mpi_get_data(dep_cmd_item_content_t* task, int from, int i);

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
                DEBUG(("FROM\t%d\tActivate\t%s\ti=%d\n", status.MPI_SOURCE, remote_dep_cmd_to_string(&dep_activate_buff[i], tmp, 128), i));
                remote_dep_mpi_get_data(&dep_activate_buff[i], status.MPI_SOURCE, i);
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
                    DEBUG(("TO\tna\tPut END  \tunknown \tj=%d\twith data %d at %p\n", i, PTR_TO_TAG(dep_get_buff[i]), (uintptr_t) dep_get_buff[i]));
                    gc_data_unref((gc_data_t*) (uintptr_t) dep_get_buff[i]);
                    MPI_Start(&dep_get_req[i]);
                    remote_dep_dec_flying_messages(eu_context->master_context);
                }
                else
                {
                    i -= DEP_NB_CONCURENT;
                    assert((i >= 0) && (i < DEP_NB_CONCURENT));
                    /* We received a data, call the matching release_dep */
                    DEBUG(("FROM\t%d\tGet END  \t%s\ti=%d\twith data %d to %p\n", status.MPI_SOURCE, remote_dep_cmd_to_string(&dep_activate_buff[i], tmp, 128), i, status.MPI_TAG, &dep_activate_buff[i].activate.data[0]));
                    CRC_PRINT(GC_DATA((gc_data_t*) dep_activate_buff[i].activate.data[0]), "R");
                    TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_ek, i);
                    remote_dep_release(eu_context, &dep_activate_buff[i]);
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
    DEBUG(("TO\t%d\tPut START\tunknown \tj=%d\twith data %d at %p\n", to, i, PTR_TO_TAG(data), data));
    TAKE_TIME(MPIsnd_prof[i], MPI_Data_plds_sk, i);
    MPI_Isend(GC_DATA(data), TILE_SIZE, MPI_DOUBLE, to, PTR_TO_TAG(data), dep_comm, &dep_put_snd_req[i]);
}

static int get = 1;

static void remote_dep_mpi_get_data(dep_cmd_item_content_t* task, int from, int i)
{
#ifdef DPLASMA_DEBUG
    char tmp[128];
#endif

    assert(dep_enabled);
    for(int k = 0; k < 1; k++)
    {
        remote_dep_datakey_t datakey = (intptr_t) task->activate.data[k];
        task->activate.data[k] = gc_data_new(malloc(sizeof(double) * TILE_SIZE), 1);
    
        DEBUG(("TO\t%d\tGet START\t%s\ti=%d\twith data %d at %p\n", from, remote_dep_cmd_to_string(task, tmp, 128), i, PTR_TO_TAG(datakey), (void*) (intptr_t) datakey));
        TAKE_TIME(MPIrcv_prof[i], MPI_Data_pldr_sk, i);
        MPI_Irecv(GC_DATA(task->activate.data[k]), TILE_SIZE, 
                  MPI_DOUBLE, from, PTR_TO_TAG(datakey), dep_comm, &dep_put_rcv_req[i]);

        TAKE_TIME(MPIctl_prof, MPI_Data_ctl_sk, get);
        MPI_Send(&datakey, datakey_count, datakey_dtt, from, REMOTE_DEP_GET_DATA_TAG, dep_comm);
        TAKE_TIME(MPIctl_prof, MPI_Data_ctl_ek, get++); 
    }
}

static int act = 1;

/* Send the activate tag */
static int remote_dep_mpi_send(dep_cmd_item_content_t* cmdu)
{
#ifdef DPLASMA_DEBUG
    char tmp[128];
#endif

    assert(dep_enabled);
    TAKE_TIME(MPIctl_prof, MPI_Activate_sk, act);
    DEBUG(("TO\t%d\tActivate\t%s\ti=na\twith data %d at %p\n", cmdu->activate.rank, remote_dep_cmd_to_string(cmdu, tmp, 128), PTR_TO_TAG(cmdu->activate.data[0]), cmdu->activate.data[0]));
    CRC_PRINT((double**)GC_DATA(cmdu->activate.data[0]), "S");
    
    MPI_Send((void*) cmdu, dep_count, dep_dtt, cmdu->activate.rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
    TAKE_TIME(MPIctl_prof, MPI_Activate_ek, act++);
    
    return 1;
}

void remote_dep_mpi_create_default_datatype(int tile_size, MPI_Datatype base)
{
    MPI_Type_contiguous(tile_size * tile_size, base, &DPLASMA_DEFAULT_DATA_TYPE);
    MPI_Type_commit(&DPLASMA_DEFAULT_DATA_TYPE);
}
