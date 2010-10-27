/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

#if defined(DAGUE_CUDA_SUPPORT) && defined(PRECISION_s)
#include "cuda_stsmqr.h"
#endif

#define _FMULS_GEQRF(M, N) ( ( (M) > (N) ) ? ((DagDouble_t)(N) * ( (DagDouble_t)(N) * ( 0.5 - (1. / 3.) * (DagDouble_t)(N) + (DagDouble_t)(M) ) + (DagDouble_t)(M) ) ) \
                             : ( (DagDouble_t)(M) * ( (DagDouble_t)(M) * ( -0.5 - (1. / 3.) * (DagDouble_t)(M) + (DagDouble_t)(N) ) + 2. * (DagDouble_t)(N) ) ) )
#define _FADDS_GEQRF(M, N) ( ( (M) > (N) ) ? ((DagDouble_t)(N) * ( (DagDouble_t)(N) * ( 0.5 - (1. / 3.) * (DagDouble_t)(N) + (DagDouble_t)(M) )                    ) ) \
                             : ( (DagDouble_t)(M) * ( (DagDouble_t)(M) * ( -0.5 - (1. / 3.) * (DagDouble_t)(M) + (DagDouble_t)(N) ) +      (DagDouble_t)(N) ) ) )

int main(int argc, char ** argv)
{
    int iparam[IPARAM_SIZEOF];
    dague_context_t* dague;

    /* Set defaults for non argv iparams */
    iparam_default_solve(iparam);
#if defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#else
    iparam[IPARAM_NGPUS] = -1;
#endif
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
    iparam[IPARAM_NB] = iparam[IPARAM_MB] = 144;
    iparam[IPARAM_IB] = 48;
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);

    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int cores = iparam[IPARAM_NCORES];
    int gpus  = iparam[IPARAM_NGPUS];
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

    DagDouble_t flops, gflops;
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
    flops = 2.*_FADDS_GEQRF(M, N) + 6.*_FMULS_GEQRF(M, N);
#else
    flops = _FADDS_GEQRF(M, N) + _FMULS_GEQRF(M, N);
#endif

    /* initializing matrix structure */
    two_dim_block_cyclic_t ddescA;
    two_dim_block_cyclic_init(&ddescA, matrix_ComplexDouble, nodes, cores, rank, MB, NB, M,     N, 0, 0, LDA,   N, SMB, SNB, P);
    ddescA.mat = dague_data_allocate((size_t)ddescA.super.nb_local_tiles * (size_t)ddescA.super.bsiz * (size_t)ddescA.super.mtype);
    generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);
    
    two_dim_block_cyclic_t ddescT;
    two_dim_block_cyclic_init(&ddescT, matrix_ComplexDouble, nodes, cores, rank, IB, NB, mt*IB, N, 0, 0, mt*IB, N, SMB, SNB, P);
    ddescT.mat = dague_data_allocate((size_t)ddescT.super.nb_local_tiles * (size_t)ddescT.super.bsiz * (size_t)ddescT.super.mtype);
    generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescT);

#if defined(DAGUE_PROFILING)
    ddescA.super.super.key = strdup("A");
    ddescT.super.super.key = strdup("T");
#endif

    /* load the GPU kernel */
#if defined(DAGUE_CUDA_SUPPORT) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0)
    {
        if(loud) printf("Load GPU kernels ... ");
        if(0 != stsmqr_cuda_init((tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescT)) 
        {
            fprintf(stderr, "Unable to load TSMQR operations.\n");
            exit(3);
        }
        if(loud) printf("Done\n");
    }
#endif

    if(iparam[IPARAM_CHECK] == 0)
    {
        /* Create GEQRF DAGuE */
        if(loud) printf("Generate GEQRF DAG ... ");
        TIME_START();
        dague_object_t* dague_zgeqrf = 
            dplasma_zgeqrf_New((tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescT);
        dague_enqueue(dague, dague_zgeqrf);
        if(loud) printf("Done\n");
        if(loud) TIME_PRINT(rank, ("DAG creation: %u total tasks enqueued\n", dague->taskstodo));

        /* lets rock! */
        SYNC_TIME_START();
        TIME_START();
        dague_progress(dague);
        if(loud) TIME_PRINT(rank, ("Dague proc %d:\tcomputed %u tasks,\t%f task/s\n",
                    rank, dague_zgeqrf->nb_local_tasks, 
                    dague_zgeqrf->nb_local_tasks/time_elapsed));
        SYNC_TIME_PRINT(rank, ("Dague progress:\t%d %d %f gflops\n", N, NB,
                         gflops = (flops/1e9)/(sync_time_elapsed)));
    }

#if defined(DAGUE_CUDA_SUPPORT) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0) 
    {
        stsmqr_cuda_fini();
    }
#endif

    dague_data_free(ddescA.mat);
    dague_data_free(ddescT.mat);

#if defined(DAGUE_PROFILING)
    free(ddescA.super.super.key);
    free(ddescT.super.super.key);
#endif

    cleanup_dague(dague);
    return EXIT_SUCCESS;
}

