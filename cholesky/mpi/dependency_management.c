/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dependency_management.h"
#include "data_management.h"
#include <mpi.h>

MPI_Comm dep_comm;

enum {
    DEP_SATISFY_TAG,
} dependency_management_type_t;

#define DEP_DTT MPI_INT


int dependency_management_init(void)
{
    MPI_Comm_dup(MPI_COMM_WORLD, &dep_comm);
    
}

int dependency_management_fini(void)
{
    
}

int dependency_management_satisfy(DPLASMA_desc * Ddesc, int tm, int tn, int lm, int ln)
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

