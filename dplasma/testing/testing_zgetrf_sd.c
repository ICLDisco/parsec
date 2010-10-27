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
    int iparam[IPARAM_SIZEOF];
    dague_context_t* dague;

    /* Set defaults for non argv iparams */
    iparam_default_solve(iparam);
    iparam[IPARAM_NGPUS] = -1;
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    
    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int cores = iparam[IPARAM_NCORES];
    int M     = iparam[IPARAM_M];
    int N     = iparam[IPARAM_N];
    int MB    = iparam[IPARAM_MB];
    int NB    = iparam[IPARAM_NB];
    int IB    = iparam[IPARAM_IB];
    int LDA   = iparam[IPARAM_LDA];
    int SMB   = iparam[IPARAM_SMB];
    int SNB   = iparam[IPARAM_SNB];
    int P     = iparam[IPARAM_P];
    int mt    = (M%MB==0) ? (M/MB) : (M/MB+1);
    int loud  = iparam[IPARAM_VERBOSE];
    int info;

    DagDouble_t flops, gflops;
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
    
    /* In each tile we store IPIV followed by L */
    two_dim_block_cyclic_t ddescLIPIV;
    two_dim_block_cyclic_init(&ddescLIPIV, matrix_ComplexDouble, nodes, cores, rank, IB+1, NB, mt*(IB+1), N, 0, 0, mt*(IB+1), N, SMB, SNB, P);
    ddescLIPIV.mat = dague_data_allocate((size_t)ddescLIPIV.super.nb_local_tiles * (size_t)ddescLIPIV.super.bsiz * (size_t)ddescLIPIV.super.mtype);
    generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescLIPIV);

#if defined(DAGUE_PROFILING)
    ddescA.super.super.key = strdup("A");
    ddescLIPIV.super.super.key = strdup("LIPIV");
#endif

    if(iparam[IPARAM_CHECK] == 0) {
        /* Create GETRF DAGuE */
        if(loud) printf("Generate GETRF DAG ... ");
        SYNC_TIME_START();
        dague_object_t* dague_zgetrf = 
            dplasma_zgetrf_sd_New((tiled_matrix_desc_t*)&ddescA,
                                  (tiled_matrix_desc_t*)&ddescLIPIV,
                                  &info);
        dague_enqueue( dague, (dague_object_t*)dague_zgetrf);
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
    dague_data_free(ddescLIPIV.mat);

    cleanup_dague(dague);
    return EXIT_SUCCESS;
}
