/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* /!\  THIS FILE IS NOT INTENDED TO BE COMPILED ON ITS OWN
 *      It should be included from remote_dep.c if USE_MPI is defined
 */

#include <mpi.h>

#define USE_MPI_THREAD_PROGRESS

static int __remote_dep_mpi_init(dplasma_context_t* context);
static int __remote_dep_mpi_fini(dplasma_context_t* context);
static int __remote_dep_send(const dplasma_execution_context_t* task, int rank);
static int __remote_dep_progress(dplasma_execution_unit_t* eu_context);

#ifdef USE_MPI_THREAD_PROGRESS
    static int remote_dep_thread_init(dplasma_context_t* context);
    static int remote_dep_thread_fini(dplasma_context_t* fini);
    static int remote_dep_thread_send(const dplasma_execution_context_t* task, int rank);
    static int remote_dep_thread_progress(dplasma_execution_unit_t* eu_context);
#   define remote_dep_mpi_init(ctx) remote_dep_thread_init(ctx)
#   define remote_dep_mpi_fini(ctx) remote_dep_thread_fini(ctx)
#   define remote_dep_send(task, rank) remote_dep_thread_send(task, rank)
#   define remote_dep_progress(ctx) remote_dep_thread_progress(ctx)
#else
#   define remote_dep_mpi_init(ctx) __remote_dep_mpi_init(ctx)
#   define remote_dep_mpi_fini(ctx) __remote_dep_mpi_fini(ctx)
#   define remote_dep_send(task, rank) __remote_dep_send(task, rank)
#   define remote_dep_progress(ctx) __remote_dep_progress(ctx)
#endif 


/* TODO: smart use of dplasma context instead of ugly globals */
static MPI_Comm dep_comm;
static MPI_Request dep_req;
/* TODO: fix heterogeneous restriction by using mpi datatypes */ 
#define dep_dtt MPI_BYTE
#define dep_count sizeof(dplasma_execution_context_t)
static dplasma_execution_context_t dep_buff;

int __remote_dep_init(dplasma_context_t* context)
{
    return remote_dep_mpi_init(context);
}

int __remote_dep_fini(dplasma_context_t* context)
{
    return remote_dep_mpi_fini(context);
}


int dplasma_remote_dep_activate_rank(dplasma_execution_unit_t* eu_context, 
                                     const dplasma_execution_context_t* origin,
                                     const param_t* origin_param,
                                     const dplasma_execution_context_t* exec_context,
                                     const param_t* new_param,
                                     int rank)
{
#ifdef _DEBUG
    char tmp[128];
    char tmp2[128];
#endif
    
    assert(rank >= 0);
    assert(rank < eu_context->master_context->nb_nodes);
    if(dplasma_remote_dep_is_forwarded(eu_context, rank))
    {    
        return 0;
    }
    dplasma_remote_dep_mark_forwarded(eu_context, rank);
    DEBUG(("%s -> %s\ttrigger REMOTE process rank %d\n", dplasma_service_to_string(origin, tmp2, 128), dplasma_service_to_string(exec_context, tmp, 128), rank ));
    return remote_dep_send(origin, rank);
}

int dplasma_remote_dep_activate(dplasma_execution_unit_t* eu_context,
                                const dplasma_execution_context_t* origin,
                                const param_t* origin_param,
                                const dplasma_execution_context_t* exec_context,
                                const param_t* dest_param )
{
    int rank; 
#ifdef _DEBUG
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
    return remote_dep_send(origin, rank);
}


int dplasma_remote_dep_progress(dplasma_execution_unit_t* eu_context)
{
    return remote_dep_progress(eu_context);
}


static int __remote_dep_mpi_init(dplasma_context_t* context)
{
    int np;
    MPI_Comm_dup(MPI_COMM_WORLD, &dep_comm);
    MPI_Comm_size(dep_comm, &np);
    MPI_Recv_init(&dep_buff, dep_count, dep_dtt, MPI_ANY_SOURCE, REMOTE_DEP_ACTIVATE_TAG, dep_comm, &dep_req);
    MPI_Start(&dep_req);
    return np;    
}

static int __remote_dep_mpi_fini(dplasma_context_t* context)
{
    MPI_Request_free(&dep_req);
    MPI_Comm_free(&dep_comm);
    return 0;
}

static int __remote_dep_send(const dplasma_execution_context_t* task, int rank)
{
    return MPI_Send((void*) task, dep_count, dep_dtt, rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
}

static int __remote_dep_progress(dplasma_execution_unit_t* eu_context)
{
#ifdef _DEBUG
    char tmp[128];
#endif
    MPI_Status status;
    int flag;
    
    MPI_Test(&dep_req, &flag, &status);
    if(flag)
    {
        DEBUG(("%s -> local\tFROM REMOTE process rank %d\n", dplasma_service_to_string(&dep_buff, tmp, 128), status.MPI_SOURCE));
        fprintf(stderr, 
                "TODO: currently, I'm calling the last parameter with NULL (%s:%d),\n"
                "it MUST be an array of pointers of output variables produced by this task\n"
                "-- expect segfault in the next call\n"
                "-- Thomas\n", __FILE__, __LINE__);
        dep_buff.function->release_deps(eu_context, &dep_buff, 0, NULL);
        MPI_Start(&dep_req);
        return 1;
    }
    return 0;
}


#ifdef USE_MPI_THREAD_PROGRESS

#include <pthread.h>
#include <errno.h>

#define YIELD_TIME 50000
static inline void update_ts(struct timespec *ts, long nsec) 
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
volatile dep_signal_reason_t dep_signal_reason = WANT_ZERO;
volatile int dep_ret;

volatile int enable_progress = 0;
volatile int np = 0;

dplasma_execution_context_t *dep_send_context;
int dep_send_rank;

dplasma_execution_unit_t *dep_recv_eu_context;


static void* remote_dep_thread_main(dplasma_context_t* context)
{
    int ret;
    struct timespec ts;
    
    np = __remote_dep_mpi_init(context);
    
    clock_gettime(CLOCK_REALTIME, &ts);
    
    pthread_mutex_lock(&dep_msg_mutex);
    do {
        switch(dep_signal_reason)
        {                
            case WANT_SEND:
                dep_ret = __remote_dep_send(dep_send_context, dep_send_rank);
                dep_signal_reason = WANT_ZERO;
                goto sleep;
            case WANT_RECV:
                dep_ret = __remote_dep_progress(dep_recv_eu_context);
                dep_signal_reason = WANT_ZERO;
                goto sleep;
            case WANT_FINI:
                goto fini;
            case WANT_ZERO:
                if(enable_progress)
                {
                    __remote_dep_progress(&context->execution_units[0]);
                }
sleep:
                update_ts(&ts, YIELD_TIME);
                ret = pthread_cond_timedwait(&dep_msg_cond, &dep_msg_mutex, &ts);
                assert((0 == ret) || (ETIMEDOUT == ret));
                continue;
        }
    } while(1);
fini:
    pthread_mutex_unlock(&dep_msg_mutex);

    __remote_dep_mpi_fini(context);    
    return context;
}

static int remote_dep_thread_init(dplasma_context_t* context)
{
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
#ifdef __linux
    pthread_setconcurrency(context->nb_cores + 1);
#endif  /* __linux */

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

static int remote_dep_thread_send(const dplasma_execution_context_t* task, int rank)
{
    int ret; 
    
    pthread_mutex_lock(&dep_seq_mutex);
    pthread_mutex_lock(&dep_msg_mutex);
    
    dep_signal_reason = WANT_SEND;
    dep_ret = -1;
    dep_send_context = (dplasma_execution_context_t*) task;
    dep_send_rank = rank;
    
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
    
    enable_progress = 1;
    
    dep_signal_reason = WANT_RECV;
    dep_ret = -1;
    dep_recv_eu_context = eu_context;

    pthread_cond_signal(&dep_msg_cond);
    pthread_mutex_unlock(&dep_msg_mutex);
    
    while(-1 == dep_ret);
    ret = dep_ret;
    
    pthread_mutex_unlock(&dep_seq_mutex);
    return ret;
}

#endif 

