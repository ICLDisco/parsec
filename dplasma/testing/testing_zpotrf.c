/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.h"
#if defined(DAGUE_CUDA_SUPPORT) && defined(PRECISION_s)
#include "cuda_sgemm.h"
#endif

#define _FMULS_POTRF(N) ((N) * (1.0 / 6.0 * (N) + 0.5) * (N))
#define _FADDS_POTRF(N) ((N) * (1.0 / 6.0 * (N)      ) * (N))

#define _FMULS_POTRS(N, NRHS) ( (NRHS) * ( (N) * ((N) + 1.) ) )
#define _FADDS_POTRS(N, NRHS) ( (NRHS) * ( (N) * ((N) - 1.) ) )

int main(int argc, char ** argv)
{
    int iparam[IPARAM_SIZEOF];
    dague_context_t* dague;

    /* Set defaults for non argv iparams */
    iparam_default_solve(iparam);
#if !defined(PRECISION_s) || !defined(DAGUE_CUDA_SUPPORT)
    iparam[IPARAM_NGPUS] = -1;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);

    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int cores = iparam[IPARAM_NCORES];
    //int M     = iparam[IPARAM_M];
    int N     = iparam[IPARAM_N];
    int NRHS  = iparam[IPARAM_K];
    int MB    = iparam[IPARAM_MB];
    int NB    = iparam[IPARAM_NB];
    int IB    = iparam[IPARAM_IB];
    int LDA   = iparam[IPARAM_LDA];
    int LDB   = iparam[IPARAM_LDB];
    int SMB   = iparam[IPARAM_SMB];
    int SNB   = iparam[IPARAM_SNB];
    int P     = iparam[IPARAM_P];
    int loud  = iparam[IPARAM_VERBOSE];
    int info;

    DagDouble_t flops, gflops;
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
    flops = 2.*_FADDS_POTRF(N) + 6.*_FMULS_POTRF(N);
#else
    flops = _FADDS_POTRF(N) + _FMULS_POTRF(N);
#endif
    
    //#ifdef VTRACE 
    //    VT_ON();
    //#endif
    
    /* initializing matrix structure */
    sym_two_dim_block_cyclic_t ddescA;
    sym_two_dim_block_cyclic_init(&ddescA, matrix_ComplexDouble, nodes, cores, rank,
                                  MB, NB, N, N, 0, 0, LDA, N, P);
    ddescA.mat = dague_data_allocate((size_t)ddescA.super.nb_local_tiles * (size_t)ddescA.super.bsiz * (size_t)ddescA.super.mtype);
#if defined(DAGUE_CUDA_SUPPORT) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0)
    {
        if(0 != spotrf_cuda_init((tiled_matrix_desc_t *) &ddescA)) {
            fprintf(stderr, "Unable to load GPU GEMM kernels.\n");
            exit(3);
        }
    }
#endif
    if(loud) printf("Generate matrices ... ");
    generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);
    if(loud) printf("Done\n");
#if defined(DAGUE_PROFILING)
    ddescA.super.super.key = strdup("A");
#endif

    if(iparam[IPARAM_CHECK] == 0) 
    {
        PLASMA_enum uplo = PlasmaLower;
        
        /* Create TRMM DAGuE */
        if(loud) printf("Generate ZPOTRF DAG ... ");
        SYNC_TIME_START();
#if defined(LLT_LL)
        dague_object_t* dague_zpotrf = 
            dplasma_zpotrf_ll_New(uplo, (tiled_matrix_desc_t*)&ddescA, &info);
#else
        dague_object_t* dague_zpotrf = 
            dplasma_zpotrf_New(uplo, (tiled_matrix_desc_t*)&ddescA, &info);
#endif
        dague_enqueue( dague, (dague_object_t*)dague_zpotrf);
        if(loud) printf("Done\n");
        if(loud) SYNC_TIME_PRINT(rank, ("DAG creation: %u total tasks enqueued\n", dague->taskstodo));

        /* lets rock! */
        SYNC_TIME_START();
        TIME_START();
        dague_progress(dague);
        if(loud) TIME_PRINT(rank, ("Dague proc %d:\tcomputed %u tasks,\t%f task/s\n",
                    rank, dague_zpotrf->nb_local_tasks,
                    dague_zpotrf->nb_local_tasks/time_elapsed));
        SYNC_TIME_PRINT(rank, ("Dague progress:\t%d %d %f gflops\n", N, NB,
                         gflops = (flops/1e9)/(sync_time_elapsed)));
    }
#if 0 /* Future check */
    else {

    }
#else
    /* Old checking by comparison: will be remove because doesn't check anything */
    if(iparam[IPARAM_CHECK] == 1) {
        char fname[20];
        sprintf(fname , "zposv_r%d", rank );
        printf("writing matrix to file\n");
        data_write((tiled_matrix_desc_t *) &ddescA, fname);
    } 
    else if( iparam[IPARAM_CHECK] == 2 ){
        char fname[20];
        sym_two_dim_block_cyclic_t ddescB;

        sym_two_dim_block_cyclic_init(&ddescB, matrix_ComplexDouble, nodes, cores, rank,
                                      MB, NB, N, N, 0, 0, LDA, N, P);
        ddescB.mat = dague_data_allocate((size_t)ddescB.super.nb_local_tiles * (size_t)ddescB.super.bsiz * (size_t)ddescB.super.mtype);
#if defined(DAGUE_PROFILING)
        ddescB.super.super.key = strdup("B");
#endif

        sprintf(fname , "sposv_r%d", rank );
        printf("reading matrix from file\n");
        data_read((tiled_matrix_desc_t *) &ddescB, fname);
        
        matrix_scompare_dist_data((tiled_matrix_desc_t *) &ddescA, (tiled_matrix_desc_t *) &ddescB);

        dague_data_free(ddescB.mat);
#if defined(DAGUE_PROFILING)
        free(ddescB.super.super.key);
#endif
    }
#endif

    dague_data_free(ddescA.mat);
#if defined(DAGUE_PROFILING)
    free(ddescA.super.super.key);
#endif

    cleanup_dague(dague);
    return EXIT_SUCCESS;
}
