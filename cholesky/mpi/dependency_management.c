/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dependency_management.h"
#include "data_management.h"
#include <mpi.h>

enum {
    DEP_SATISFY_TAG,
} dependency_management_type_t;

static MPI_Comm dep_comm;
static MPI_Request dep_req;
#define dep_dtt MPI_INT
static dplasma_execution_context_t dep_buff

int dependency_management_init(void)
{
    MPI_Comm_dup(MPI_COMM_WORLD, &dep_comm);
    MPI_Irecv_init(&dep_buff, 1, dep_dtt, MPI_ANY_SOURCE, DEP_SATISFY_TAG, dep_comm, &dep_req);
    MPI_Start(&dep_req, 1);
}

int dependency_management_fini(void)
{
    MPI_Request_cancel(&dep_req);
    MPI_Request_free(&dep_req);
    MPI_Comm_free(&dep_comm);
}

int dependency_management_satisfy(DPLASMA_desc * Ddesc, const dplasma_execution_context_t *orig)
{
    int trank;
    
    trank = get_rank_for_tile(Ddesc, tm, tn);
    if(trank == Ddesc->my_rank)
    {
        
        
    }
    else {
        MPI_Send(buff, 1, DEP_DTT, trank, DEP_SATISFY_TAG, dep_comm);
    }
}

int dependency_management_test_activation(DPLASMA_desc *Ddesc, dplasma_execution_context_t *new)
{
    MPI_Status status;
    int flag;
    
    MPI_Test(&dep_req, &flag, &status);
    if(flag)
    {
        
        
    }
    else
    {
        new = NULL; 
        
    }
    
}

