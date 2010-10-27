/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#ifdef USE_MPI
#include <mpi.h>
#endif  /* defined(USE_MPI) */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <math.h>
#include <cblas.h>
#include <plasma.h>
#include <lapacke.h>

#include "dague.h"
#include "scheduling.h"
#include "profiling.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"
#include "dplasma.h"

#include "common.h"
#include "common_timing.h"

//#ifdef VTRACE
//#include "vt_user.h"
//#endif

#define _FMULS_GETRF(M, N) ( 0.5 * (DagDouble_t)(N) * ( (DagDouble_t)(N) * ((DagDouble_t)(M) - (1. / 3.) * (DagDouble_t)(N)) - (DagDouble_t)(N) ) )
#define _FADDS_GETRF(M, N) ( 0.5 * (DagDouble_t)(N) * ( (DagDouble_t)(N) * ((DagDouble_t)(M) - (1. / 3.) * (DagDouble_t)(N))                    ) )

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    DagDouble_t flops, gflops;

    /* Set defaults for non argv iparams */
    iparam_default_solve(iparam);
    iparam[IPARAM_NGPUS] = -1;
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);

    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int cores = iparam[IPARAM_NCORES];
    //int gpus  = iparam[IPARAM_NGPUS];
    int prio  = iparam[IPARAM_PRIO];
    int P     = iparam[IPARAM_P];
    int Q     = iparam[IPARAM_Q];
    int M     = iparam[IPARAM_M];
    int N     = iparam[IPARAM_N];
    int LDA   = iparam[IPARAM_LDA];
    int LDB   = iparam[IPARAM_LDB];
    int MB    = iparam[IPARAM_MB];
    int NB    = iparam[IPARAM_NB];
    int IB    = iparam[IPARAM_IB];
    int SMB   = iparam[IPARAM_SMB];
    int SNB   = iparam[IPARAM_SNB];
    int loud  = iparam[IPARAM_VERBOSE];
    int mt    = (M%MB==0) ? (M/MB) : (M/MB+1);
    int nt    = (N%NB==0) ? (N/NB) : (N/NB+1);
    int info;

    
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
    flops = 2.*_FADDS_GETRF(M, N) + 6.*_FMULS_GETRF(M, N);
#else
    flops = _FADDS_GETRF(M, N) + _FMULS_GETRF(M, N);
#endif

    /* initializing matrix structure */
    two_dim_block_cyclic_t ddescA;
    two_dim_block_cyclic_init(&ddescA, matrix_ComplexDouble, nodes, cores, rank, MB, NB, M, N, 0, 0, LDA, N, SMB, SNB, P);
    ddescA.mat = dague_data_allocate((size_t)ddescA.super.nb_local_tiles * (size_t)ddescA.super.bsiz * (size_t)ddescA.super.mtype);
    generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);
    
    two_dim_block_cyclic_t ddescL;
    two_dim_block_cyclic_init(&ddescL, matrix_ComplexDouble, nodes, cores, rank, IB, NB, mt*IB, N, 0, 0, mt*IB, N,  SMB, SNB, P);
    ddescL.mat = dague_data_allocate((size_t)ddescL.super.nb_local_tiles * (size_t)ddescL.super.bsiz * (size_t)ddescL.super.mtype);
    generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescL);
    
    two_dim_block_cyclic_t ddescIPIV;
    two_dim_block_cyclic_init(&ddescIPIV, matrix_Integer,       nodes, cores, rank, MB,  1, M,     nt, 0, 0, M,     nt, SMB, SNB, P);
    ddescIPIV.mat = dague_data_allocate((size_t)ddescIPIV.super.nb_local_tiles * (size_t)ddescIPIV.super.bsiz * (size_t)ddescIPIV.super.mtype);
    generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescIPIV);


#if defined(DAGUE_PROFILING)
    ddescA.super.super.key = strdup("A");
    ddescL.super.super.key = strdup("L");
    ddescIPIV.super.super.key = strdup("IPIV");
#endif

    if(iparam[IPARAM_CHECK] == 0) 
    {
        /* Create GETRF DAGuE */
        if(loud) printf("Generate GETRF DAG ... ");
        TIME_START();
        dague_object_t* dague_zgetrf = 
            dplasma_zgetrf_New((tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescL,
                               (tiled_matrix_desc_t*)&ddescIPIV,
                               &info );
        dague_enqueue(dague, dague_zgetrf);
        if(loud) printf("Done\n");
        if(loud) TIME_PRINT(rank, ("DAG creation: %u total tasks enqueued\n", dague->taskstodo));

        /* lets rock! */
        SYNC_TIME_START();
        TIME_START();
        dague_progress(dague);
        if(loud) TIME_PRINT(rank, ("Dague proc %d:\tcomputed %u tasks,\t%f task/s\n",
                    rank, dague_zgetrf->nb_local_tasks, 
                    dague_zgetrf->nb_local_tasks/time_elapsed));
        SYNC_TIME_PRINT(rank, ("Dague progress:\t%d %d %f gflops\n", N, NB,
                         gflops = (flops/1e9)/(sync_time_elapsed)));
    }
    
    dague_data_free(ddescA.mat);
    dague_data_free(ddescL.mat);
    dague_data_free(ddescIPIV.mat);

    cleanup_dague(dague);

#if defined(DAGUE_PROFILING)
    free(ddescA.super.super.key);
    free(ddescL.super.super.key);
    free(ddescIPIV.super.super.key);
#endif
    return EXIT_SUCCESS;
}
