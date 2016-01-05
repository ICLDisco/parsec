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
    PLASMA_enum uplo = PlasmaLower;
    int info = 0;
    int ret = 0;

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

    /* initializing matrix structure */
    LDA = dplasma_imax( LDA, N );
    LDB = dplasma_imax( LDB, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( dague, (double)(N), uplo,
                    (tiled_matrix_desc_t *)&ddescA, random_seed);
    if(loud > 3) printf("Done\n");

    if((iparam[IPARAM_HNB] != iparam[IPARAM_NB]) || (iparam[IPARAM_HMB] != iparam[IPARAM_MB]))
    {

        SYNC_TIME_START();
        dague_handle_t* DAGUE_zpotrf = dplasma_zpotrf_New( uplo, (tiled_matrix_desc_t*)&ddescA, &info );
        /* Set the recursive size */
        dplasma_zpotrf_setrecursive( DAGUE_zpotrf, iparam[IPARAM_HMB] );
        dague_enqueue(dague, DAGUE_zpotrf);
        nb_local_tasks = DAGUE_zpotrf->nb_local_tasks;
        if( loud > 2 ) SYNC_TIME_PRINT(rank, ( "zpotrf\tDAG created\n"));

        PASTE_CODE_PROGRESS_KERNEL(dague, zpotrf);
        dplasma_zpotrf_Destruct( DAGUE_zpotrf );

        dague_handle_sync_ids(); /* recursive DAGs are not synchronous on ids */
    }
    else
    {
        PASTE_CODE_ENQUEUE_KERNEL(dague, zpotrf,
                                  (uplo, (tiled_matrix_desc_t*)&ddescA, &info));
        PASTE_CODE_PROGRESS_KERNEL(dague, zpotrf);

        dplasma_zpotrf_Destruct( DAGUE_zpotrf );
    }

    if( 0 == rank && info != 0 ) {
        printf("-- Factorization is suspicious (info = %d) ! \n", info);
        ret |= 1;
    }
    if( !info && check ) {
        /* Check the factorization */
        PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
            sym_two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble,
                                       nodes, rank, MB, NB, LDA, N, 0, 0,
                                       N, N, P, uplo));
        dplasma_zplghe( dague, (double)(N), uplo,
                        (tiled_matrix_desc_t *)&ddescA0, random_seed);

        ret |= check_zpotrf( dague, (rank == 0) ? loud : 0, uplo,
                             (tiled_matrix_desc_t *)&ddescA,
                             (tiled_matrix_desc_t *)&ddescA0);

        /* Check the solution */
        PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
            two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));
        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, random_seed+1);

        PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
            two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );

        dplasma_zpotrs(dague, uplo,
                       (tiled_matrix_desc_t *)&ddescA,
                       (tiled_matrix_desc_t *)&ddescX );

        ret |= check_zaxmb( dague, (rank == 0) ? loud : 0, uplo,
                            (tiled_matrix_desc_t *)&ddescA0,
                            (tiled_matrix_desc_t *)&ddescB,
                            (tiled_matrix_desc_t *)&ddescX);

        /* Cleanup */
        dague_data_free(ddescA0.mat); ddescA0.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0 );
        dague_data_free(ddescB.mat); ddescB.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB );
        dague_data_free(ddescX.mat); ddescX.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescX );
    }

    dague_data_free(ddescA.mat); ddescA.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);

    cleanup_dague(dague, iparam);
    return ret;
}
