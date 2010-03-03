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

#define USE_MPI_THREAD_NONE

static int remote_dep_mpi_init(dplasma_context_t* context);
static int remote_dep_mpi_fini(dplasma_context_t* context);
static int remote_dep_mpi_send(const dplasma_execution_context_t* task, int rank, void** data);
static int remote_dep_mpi_progress(dplasma_execution_unit_t* eu_context);
static int remote_dep_mpi_release(dplasma_execution_unit_t* eu_context, dplasma_execution_context_t* exec_context, void** data);

#if defined(USE_MPI_THREAD)
    static int remote_dep_dequeue_init(dplasma_context_t* context);
    static int remote_dep_dequeue_fini(dplasma_context_t* context);
    static int remote_dep_dequeue_send(const dplasma_execution_context_t* task, int rank, void** data);
    static int remote_dep_dequeue_progress(dplasma_execution_unit_t* eu_context);
    static int remote_dep_dequeue_release(dplasma_execution_unit_t* eu_context, dplasma_execution_context_t* exec_context, void** data);
#   define remote_dep_init(ctx) remote_dep_dequeue_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_dequeue_fini(ctx)
#   define remote_dep_send(task, rank, data) remote_dep_dequeue_send(task, rank, data)
#   define remote_dep_progress(ctx) remote_dep_dequeue_progress(ctx)
#   define remote_dep_release(ctx, task, data) remote_dep_dequeue_release(ctx, task, data)

#elif defined(USE_MPI_THREAD_MUTEX) /* Deprecated, do not use */
    static int remote_dep_thread_init(dplasma_context_t* context);
    static int remote_dep_thread_fini(dplasma_context_t* fini);
    static int remote_dep_thread_send(const dplasma_execution_context_t* task, int rank, void** data);
    static int remote_dep_thread_progress(dplasma_execution_unit_t* eu_context);
#   define remote_dep_init(ctx) remote_dep_thread_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_thread_fini(ctx)
#   define remote_dep_send(task, rank, data) remote_dep_thread_send(task, rank, data)
#   define remote_dep_progress(ctx) remote_dep_thread_progress(ctx)
#   define remote_dep_release(ctx, task, data) remote_dep_mpi_release(ctx, task, data);

#else
#   define remote_dep_init(ctx) remote_dep_mpi_init(ctx)
#   define remote_dep_fini(ctx) remote_dep_mpi_fini(ctx)
#   define remote_dep_send(task, rank, data) remote_dep_mpi_send(task, rank, data)
#   define remote_dep_progress(ctx) remote_dep_mpi_progress(ctx)
#   define remote_dep_release(ctx, task, data) remote_dep_mpi_release(ctx, task, data)
#endif 

#ifdef DPLASMA_PROFILING
static dplasma_thread_profiling_t *MPI_prof;
static int MPI_Activate_sk, MPI_Activate_ek;
static int MPI_Data_ctl_sk, MPI_Data_ctl_ek;
static int MPI_Data_plds_sk, MPI_Data_plds_ek;
static int MPI_Data_pldr_sk, MPI_Data_pldr_ek;

#define TAKE_TIME(KEY, I)  dplasma_profiling_trace(MPI_prof, (KEY), (I))
#else
#define TAKE_TIME(KEY, I)
#endif  /* DPLASMA_PROFILING */

int remote_dep_transport_init(dplasma_context_t* context)
{
#ifdef DPLASMA_PROFILING
    dplasma_profiling_add_dictionary_keyword( "MPI_ACTIVATE", "fill:#40B080",
                                             &MPI_Activate_sk, &MPI_Activate_ek);
    dplasma_profiling_add_dictionary_keyword( "MPI_DATA_CTL", "fill:#8080B0",
                                             &MPI_Data_ctl_sk, &MPI_Data_ctl_ek);
    dplasma_profiling_add_dictionary_keyword( "MPI_DATA_PLD_SND", "fill:#B08080",
                                             &MPI_Data_plds_sk, &MPI_Data_plds_ek);
    dplasma_profiling_add_dictionary_keyword( "MPI_DATA_PLD_RCV", "fill:#FF0000",
                                             &MPI_Data_pldr_sk, &MPI_Data_pldr_ek);

    MPI_prof = dplasma_profiling_thread_init( 4096, "MPI thread");
#endif /* DPLASMA_PROFILING */
    return remote_dep_init(context);
}

int remote_dep_transport_fini(dplasma_context_t* context)
{
    return remote_dep_fini(context);
}


int dplasma_remote_dep_activate_rank(dplasma_execution_unit_t* eu_context, 
                                     const dplasma_execution_context_t* origin,
                                     const param_t* origin_param,
                                     int rank, void** data)
{    
    assert(rank >= 0);
    assert(rank < eu_context->master_context->nb_nodes);
    if(dplasma_remote_dep_is_forwarded(eu_context, rank))
    {    
        return 0;
    }
    dplasma_remote_dep_mark_forwarded(eu_context, rank);
    
    /* make sure we don't leave before serving all data deps */
    //    dplasma_atomic_inc_32b( &(eu_context->master_context->taskstodo) );
    return remote_dep_send(origin, rank, data);
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
#define DEP_NB_CONCURENT 128
static MPI_Comm dep_comm;
static MPI_Request dep_req[4 * DEP_NB_CONCURENT];
static MPI_Request* dep_activate_req = &dep_req[0];
static MPI_Request* dep_get_req = &dep_req[DEP_NB_CONCURENT];
static MPI_Request* dep_put_rcv_req = &dep_req[2 * DEP_NB_CONCURENT];
static MPI_Request* dep_put_snd_req = &dep_req[3 * DEP_NB_CONCURENT];
/* TODO: fix heterogeneous restriction by using proper mpi datatypes */
#define dep_dtt MPI_BYTE
#define dep_count sizeof(dplasma_execution_context_t)
static dplasma_execution_context_t dep_activate_buff[DEP_NB_CONCURENT];
#define datakey_dtt MPI_LONG_LONG
static void* dep_get_buff[DEP_NB_CONCURENT];

static int remote_dep_mpi_init(dplasma_context_t* context)
{
    int i, np;
    MPI_Comm_dup(MPI_COMM_WORLD, &dep_comm);
    MPI_Comm_size(dep_comm, &np);

    TAKE_TIME(MPI_Activate_sk, 0);
    MPI_Barrier(dep_comm);
    TAKE_TIME(MPI_Activate_ek, 0);

    
    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {        
        MPI_Recv_init(&dep_activate_buff[i], dep_count, dep_dtt, MPI_ANY_SOURCE, REMOTE_DEP_ACTIVATE_TAG, dep_comm, &dep_activate_req[i]);
        MPI_Start(&dep_activate_req[i]);
        MPI_Recv_init(&dep_get_buff[i], 1, datakey_dtt, MPI_ANY_SOURCE, REMOTE_DEP_GET_DATA_TAG, dep_comm, &dep_get_req[i]);
        MPI_Start(&dep_get_req[i]);
        dep_put_rcv_req[i] = MPI_REQUEST_NULL;
        dep_put_snd_req[i] = MPI_REQUEST_NULL;
    }
    return np;
}

static int remote_dep_mpi_fini(dplasma_context_t* context)
{
    int i;

    for(i = 0; i < DEP_NB_CONCURENT; i++)
    {
        MPI_Cancel(&dep_activate_req[i]); MPI_Request_free(&dep_activate_req[i]);
        MPI_Cancel(&dep_get_req[i]); MPI_Request_free(&dep_get_req[i]);
        assert(MPI_REQUEST_NULL == dep_put_rcv_req[i]);
        assert(MPI_REQUEST_NULL == dep_put_snd_req[i]);
    }
    MPI_Barrier(dep_comm);
    MPI_Comm_free(&dep_comm);
    return 0;
}

#define TILE_SIZE (120 * 120)

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

static int remote_dep_mpi_release(dplasma_execution_unit_t* eu_context, dplasma_execution_context_t* exec_context, void** data)
{
    return exec_context->function->release_deps(eu_context, exec_context, 0, data);
}

static void remote_dep_mpi_put_data(void* data, int to, int i);
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
    
    do {
 //        TAKE_TIME(MPI_Test_any_sk);
        MPI_Testany(4 * DEP_NB_CONCURENT, dep_req, &i, &flag, &status);
 //        TAKE_TIME(MPI_Test_any_ek);
        if(flag)
        {
            if(REMOTE_DEP_ACTIVATE_TAG == status.MPI_TAG)
            {
                DEBUG(("FROM\t%d\tActivate\ti=%d\t%s\n", status.MPI_SOURCE, i, dplasma_service_to_string(&dep_activate_buff[i], tmp, 128)));
                remote_dep_mpi_get_data(&dep_activate_buff[i], status.MPI_SOURCE, i);
                
                remote_dep_release(eu_context, &dep_activate_buff[i], &dep_activate_buff[i].list_item.cache_friendly_emptiness);
                MPI_Start(&dep_activate_req[i]);

            } 
            else if(REMOTE_DEP_GET_DATA_TAG == status.MPI_TAG)
            {
                assert(0);
                i -= DEP_NB_CONCURENT; /* shift i */
                assert(i >= 0);
                remote_dep_mpi_put_data(dep_get_buff[i], status.MPI_SOURCE, i);
            }
            else 
            {
                assert(0);

                assert(REMOTE_DEP_PUT_DATA_TAG == status.MPI_TAG);
                i -= DEP_NB_CONCURENT * 2;
                assert(i >= 0);
                if(i < DEP_NB_CONCURENT)
                {
                    DEBUG(("FROM\t%d\tPut data\ti=%d\tunknown \trecv complete\n", status.MPI_SOURCE, i));
                    CRC_PRINT(dep_activate_buff[i].list_item.cache_friendly_emptiness, "R");
                    //TAKE_TIME(MPI_Data_pldr_ek, i);
                    remote_dep_release(eu_context, &dep_activate_buff[i], &dep_activate_buff[i].list_item.cache_friendly_emptiness);
                    MPI_Start(&dep_activate_req[i]);
                    ret++;
                }
                else
                {
                    /* We finished sending the data, allow for more requests 
                     * to be processed */
                    //TAKE_TIME(MPI_Data_plds_ek, i);
                    i -= DEP_NB_CONCURENT;
                    DEBUG(("TO\tna\tPut data\tj=%d\tunknown \tsend complete\n", i));
                    MPI_Start(&dep_get_req[i]);
                    /* Allow for termination if needed */
                    dplasma_atomic_dec_32b( &(eu_context->master_context->taskstodo) );
                }
            }
        }
    } while(0/*flag*/);
    return ret;
}


static void remote_dep_mpi_put_data(void* data, int to, int i)
{
    //TAKE_TIME(MPI_Data_plds_sk, i);
    DEBUG(("TO\t%d\tPut data\tj=%d\tunknown \twith data at %p\n", to, i, data));
    MPI_Isend(data, TILE_SIZE, MPI_DOUBLE, to, REMOTE_DEP_PUT_DATA_TAG, dep_comm, &dep_put_snd_req[i]);
}

static int get = 0;

static void remote_dep_mpi_get_data(dplasma_execution_context_t* task, int from, int i)
{
#ifdef DPLASMA_DEBUG
    char tmp[128];
#endif    
    void* datakey = task->list_item.cache_friendly_emptiness;
    task->list_item.cache_friendly_emptiness = malloc(sizeof(double) * TILE_SIZE);
    
/*    DEBUG(("TO\t%d\tGet data\ti=%d\t%s\twith data at %p\n", from, i, dplasma_service_to_string(task, tmp, 128), task->list_item.cache_friendly_emptiness));
    //TAKE_TIME(MPI_Data_pldr_sk, i);
    MPI_Irecv(task->list_item.cache_friendly_emptiness, TILE_SIZE, 
              MPI_DOUBLE, from, REMOTE_DEP_PUT_DATA_TAG, dep_comm, &dep_put_rcv_req[i]);

    TAKE_TIME(MPI_Data_ctl_sk, get);
    MPI_Send(&datakey, 1, datakey_dtt, from, REMOTE_DEP_GET_DATA_TAG, dep_comm);
    TAKE_TIME(MPI_Data_ctl_ek, get++);
*/
 }

static int activate_count = 1;

/* Send the activate tag */
static int remote_dep_mpi_send(const dplasma_execution_context_t* task, int rank, void **data)
{
#ifdef DPLASMA_DEBUG
    char tmp[128];
#endif    
    
    TAKE_TIME(MPI_Activate_sk, activate_count);
    DEBUG(("TO\t%d\tActivate\ti=na\t%s\twith data at %p\n", rank, dplasma_service_to_string(task, tmp, 128), data[0]));
    CRC_PRINT(((double**) data)[0], "S");
    
    ((dplasma_execution_context_t*) task)->list_item.cache_friendly_emptiness = data[0];
    MPI_Send((void*) task, dep_count, dep_dtt, rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
    TAKE_TIME(MPI_Activate_ek, activate_count++);
    
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
    DEP_FINI,
} dep_cmd_t;

typedef union dep_cmd_item_content_t
{
    struct {
        dplasma_execution_context_t origin;
        void* data;
        int rank;
    } activate;
    struct {
        dplasma_execution_unit_t* unit;
    } progress;
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
volatile int enable_self_progress;

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
        enable_self_progress = 0;
        np = 0;
        pthread_create(&dep_thread_id,
                       &thread_attr,
                       (void* (*)(void*))remote_dep_dequeue_main,
                       (void*)context);
    
        while(0 == np); /* wait until the thread inits MPI */
    }
    return np;
}

static int remote_dep_dequeue_fini(dplasma_context_t* context)
{
    if(1 < context->nb_nodes)
    {        
        dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));
        dplasma_context_t *ret;
    
        cmd->super.list_prev = (dplasma_list_item_t*) cmd;
        cmd->cmd = DEP_FINI;
    
        dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) cmd);
    
        pthread_join(dep_thread_id, (void**) &ret);
        assert(ret == context);
    }
    return 0;
}



static int remote_dep_dequeue_send(const dplasma_execution_context_t* task, int rank, void** data)
{
    dep_cmd_item_t* cmd = (dep_cmd_item_t*) calloc(1, sizeof(dep_cmd_item_t));

    cmd->super.list_prev = (dplasma_list_item_t*) cmd;
    cmd->cmd = DEP_ACTIVATE;
    cmd->u.activate.origin = *task;
    cmd->u.activate.data = data[0];
    cmd->u.activate.rank = rank;
    
    dplasma_dequeue_push_back(&dep_cmd_queue, (dplasma_list_item_t*) cmd);
    return 1;
}


static int remote_dep_dequeue_release(dplasma_execution_unit_t* eu_context, dplasma_execution_context_t* exec_context, void** data)
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
    
    enable_self_progress = 1;
    
    /* don't while, the thread is starving, let it go right away */
    if(NULL != (cmd = (dep_cmd_item_t*) dplasma_dequeue_pop_front(&dep_activate_queue)))
    {
        remote_dep_mpi_release(eu_context, &cmd->u.activate.origin, &cmd->u.activate.data);
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
            if(enable_self_progress)
            {
                remote_dep_mpi_progress(context->execution_units[0]);
            }
            nanosleep(&ts, NULL);
        }
        
        switch(cmd->cmd)
        {                
            case DEP_ACTIVATE:
                remote_dep_mpi_send(&cmd->u.activate.origin, cmd->u.activate.rank, &cmd->u.activate.data);
                break;
            case DEP_FINI:
                keep_probing = 0;
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
/* Down there is deprecated, keep for references only                         */

#elif defined(USE_MPI_THREAD_NOMUTEX)

#include <pthread.h>
#include <errno.h>
#include <sys/time.h>

#define YIELD_TIME 50000
static void init_ts(struct timespec* ts)
{
#if defined(__gnu_linux__)
    clock_gettime(CLOCK_REALTIME, ts);
#else
    gettimeofday((struct timeval*) ts, NULL);
    ts->tv_nsec *= 1000;
#endif
}

static inline void update_ts(struct timespec* ts, long nsec) 
{
    ts->tv_nsec += nsec;
    while(ts->tv_nsec > 1000000000)
    {
        ts->tv_sec += 1;
        ts->tv_nsec -= 1000000000;
    }
}

pthread_t dep_thread_id;
pthread_cond_t dep_msg_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t dep_msg_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t dep_seq_mutex = PTHREAD_MUTEX_INITIALIZER;
typedef enum {WANT_ZERO, WANT_SEND, WANT_RECV, WANT_FINI} dep_signal_reason_t;
volatile dep_signal_reason_t dep_signal_reason;
volatile int dep_ret;

volatile int enable_self_progress;
volatile int np;

dplasma_execution_context_t *dep_send_context;
void **dep_send_data;
int dep_send_rank;

dplasma_execution_unit_t *dep_recv_eu_context;

static void* remote_dep_thread_main(dplasma_context_t* context);

static int remote_dep_thread_init(dplasma_context_t* context)
{
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
    
    enable_self_progress = 0;
    np = 0;
    dep_signal_reason = WANT_ZERO;
    
    pthread_create( &dep_thread_id,
                   &thread_attr,
                   (void* (*)(void*))remote_dep_thread_main,
                   (void*)context);
    
    while(0 == np); /* wait until the thread inits MPI */
    return np;
}

static int remote_dep_thread_fini(dplasma_context_t* context)
{
    dplasma_context_t *ret;
    
    pthread_mutex_lock(&dep_seq_mutex);
    pthread_mutex_lock(&dep_msg_mutex);
    
    dep_signal_reason = WANT_FINI;
    
    pthread_cond_signal(&dep_msg_cond);
    pthread_mutex_unlock(&dep_msg_mutex);
    
    pthread_join(dep_thread_id, (void**) &ret);
    assert(ret == context);
    
    pthread_mutex_unlock(&dep_seq_mutex);
    
    return 0;
}


static int remote_dep_thread_send(const dplasma_execution_context_t* task, int rank, void** data)
{
    int ret; 
    
    pthread_mutex_lock(&dep_seq_mutex);
    pthread_mutex_lock(&dep_msg_mutex);
    
    dep_ret = -1;
    dep_send_context = (dplasma_execution_context_t*) task;
    dep_send_data = data;
    dep_send_rank = rank;
    dep_signal_reason = WANT_SEND;
    
    pthread_cond_signal(&dep_msg_cond);
    pthread_mutex_unlock(&dep_msg_mutex);
    
    while(-1 == dep_ret);
    ret = dep_ret;
    
    pthread_mutex_unlock(&dep_seq_mutex);
    return ret;
}

static int remote_dep_thread_progress(dplasma_execution_unit_t* eu_context)
{
    int ret;
    
    pthread_mutex_lock(&dep_seq_mutex);
    pthread_mutex_lock(&dep_msg_mutex);
    
    enable_self_progress = 1;
    
    dep_ret = -1;
    dep_recv_eu_context = eu_context;
    dep_signal_reason = WANT_RECV;
    
    pthread_cond_signal(&dep_msg_cond);
    pthread_mutex_unlock(&dep_msg_mutex);
    
    while(-1 == dep_ret);
    ret = dep_ret;
    
    pthread_mutex_unlock(&dep_seq_mutex);
    return ret;
}

static void* remote_dep_thread_main(dplasma_context_t* context)
{
    int ret;
    int keep_probing = 1;
    struct timespec ts;
    
    np = remote_dep_mpi_init(context);
    
    init_ts(&ts);
    
    pthread_mutex_lock(&dep_msg_mutex);
    do {
        switch(dep_signal_reason)
        {                
            case WANT_SEND:
                dep_ret = remote_dep_mpi_send(dep_send_context, dep_send_rank, dep_send_data);
                break;
            case WANT_RECV:
                dep_ret = remote_dep_mpi_progress(dep_recv_eu_context);
                break;
            case WANT_FINI:
                keep_probing = 0;
                break;
            case WANT_ZERO:
                if(enable_self_progress)
                {
                    remote_dep_mpi_progress(&context->execution_units[0]);
                }
                update_ts(&ts, YIELD_TIME);
                ret = pthread_cond_timedwait(&dep_msg_cond, &dep_msg_mutex, &ts);
                assert((0 == ret) || (ETIMEDOUT == ret));
                continue;
        }
        dep_signal_reason = WANT_ZERO;
    } while(keep_probing);
    pthread_mutex_unlock(&dep_msg_mutex);
    
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
