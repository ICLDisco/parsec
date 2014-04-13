/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "flops.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#if defined(HAVE_CUDA)
#include "dplasma/cores/cuda_zgemm.h"
#endif

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaUpper;
    int info = 0;
    int ret = 0;
    int async = 1;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
#if defined(HAVE_CUDA)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZPOTRF, ((DagDouble_t)N));
    flops += FLOPS_ZPOTRI((DagDouble_t)N);
    flops += FLOPS_ZHEMM(PlasmaRight, (DagDouble_t)M, (DagDouble_t)N);

    /* initializing matrix structure */
    LDA = N;
    LDB = M;
    LDC = M;
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));

    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, N, 0, 0,
                               M, N, 1, 1, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( dague, (double)(N), uplo,
                   (tiled_matrix_desc_t *)&ddescA, random_seed);
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, random_seed + 1);
    if(loud > 3) printf("Done\n");

    /* load the GPU kernel */
#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        if(loud > 3) printf("+++ Load GPU kernel ... ");
        dague_gpu_data_register(dague,
                                (dague_ddesc_t*)&ddescA,
                                MT*NT, MB*NB*sizeof(dague_complex64_t) );
        if(loud > 3) printf("Done\n");
    }
#endif

    if (async) {
        PASTE_CODE_ENQUEUE_KERNEL(dague, zpoinv2,
                                  (uplo,
                                   (tiled_matrix_desc_t*)&ddescA,
                                   (tiled_matrix_desc_t*)&ddescB,
                                   (tiled_matrix_desc_t*)&ddescC,
                                   &info));
        PASTE_CODE_PROGRESS_KERNEL(dague, zpoinv2);
        dplasma_zpoinv2_Destruct( DAGUE_zpoinv2 );
    }
    else {
        SYNC_TIME_START();
        info = dplasma_zpoinv2_sync( dague, uplo,
                                     (tiled_matrix_desc_t*)&ddescA,
                                     (tiled_matrix_desc_t*)&ddescB,
                                     (tiled_matrix_desc_t*)&ddescC );
        SYNC_TIME_PRINT(rank, ("zpoinv2\tPxQ= %3d %-3d NB= %4d M= %7d N= %7d : %14f gflops\n",
                               P, Q, NB, M, N,
                               gflops=(flops/1e9)/sync_time_elapsed));
    }

#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        dague_gpu_data_unregister((dague_ddesc_t*)&ddescA);
    }
#endif

    if( 0 == rank && info != 0 ) {
        printf("-- Factorization is suspicious (info = %d) ! \n", info);
        ret |= 1;
    }
    if( !info && check ) {
        /* Check the factorization */
        PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
            two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile, nodes, rank,
                                   MB, NB, LDA, N, 0, 0, N, N, 1, 1, P));
        dplasma_zplghe( dague, (double)(N), PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA0, random_seed);

        ret |= check_zpoinv( dague, (rank == 0) ? loud : 0, uplo,
                            (tiled_matrix_desc_t *)&ddescA0,
                            (tiled_matrix_desc_t *)&ddescA );

        if (ret) {
            printf("-- Innversion is suspicious ! \n");
        }
        else
        {
            printf("-- Inversion is CORRECT ! \n");
        }

        /* Cleanup */
        dague_data_free(ddescA0.mat); ddescA0.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0 );
    }

    dague_data_free(ddescA.mat); ddescA.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    dague_data_free(ddescB.mat); ddescB.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);
    dague_data_free(ddescC.mat); ddescC.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC);

    cleanup_dague(dague, iparam);
    return ret;
}
