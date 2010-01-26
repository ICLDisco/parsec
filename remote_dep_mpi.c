/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* /!\  THIS FILE IS NOT INTENDED TO BE COMPILED ON ITS OWN
 *      It should be included from remote_dep.c if USE_MPI is defined
 */

#include <mpi.h>

static int __remote_dep_mpi_init(dplasma_context_t* eu_context);
static int __remote_dep_send(const dplasma_execution_context_t* task, int rank);
static int __remote_dep_poll(dplasma_execution_context_t* task);

#ifdef USE_MPI_THREAD_PROGRESS
    static int remote_dep_thread_init(dplasma_context_t* eu_context);
    static int remote_dep_thread_send(const dplasma_execution_context_t* task, int rank);
    static int remote_dep_thread_poll(const dplasma_execution_context_t* task);
#   define remote_dep_thread_init(ctx) remote_dep_thread_init(ctx)
#   define remote_dep_send(task, rank) remote_dep_thread_send(task, rank)
#   define remote_dep_poll(task) remote_dep_thread_poll(task)
#else
#   define remote_dep_mpi_init(ctx) __remote_dep_mpi_init(ctx)
#   define remote_dep_send(task, rank) __remote_dep_send(task, rank)
#   define remote_dep_poll(task) __remote_dep_poll(task)
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
    MPI_Request_free(&dep_req);
    MPI_Comm_free(&dep_comm);
    return 0;
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
    return MPI_Send((void*) origin, dep_count, dep_dtt, rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
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
    dplasma_execution_context_t origin;
#ifdef _DEBUG
    char tmp[128];
#endif
    
    if(remote_dep_poll(&origin))
    {
        DEBUG(("%s -> local\tFROM REMOTE process rank %d\n", dplasma_service_to_string(&dep_buff, tmp, 128), status.MPI_SOURCE));
        dep_buff.function->release_deps(eu_context, &origin, 0);
        return 1;
    }
    return 0;
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

static int __remote_dep_send(const dplasma_execution_context_t* task, int rank)
{
    return MPI_Send((void*) task, dep_count, dep_dtt, rank, REMOTE_DEP_ACTIVATE_TAG, dep_comm);
}

static int __remote_dep_poll(dplasma_execution_context_t* task)
{
    MPI_Status status;
    int flag;
    
    MPI_Test(&dep_req, &flag, &status);
    if(flag)
    {
        memcpy(task, &dep_buff, sizeof(dep_buff)); 
        MPI_Start(&dep_req);
        return 1;
    }
    return 0;
}


#ifdef USE_MPI_THREAD_PROGRESS

static int remote_dep_thread_init(dplasma_execution_unit_t* eu_context)
{
    return __remote_dep_mpi_init(context);
}

static int remote_dep_thread_send(task, rank)
{
    return 0;
}

static int remote_dep_thread_poll(task)
{
    return 0;
}

#endif 

