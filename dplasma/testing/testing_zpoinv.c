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
#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaLower;
    int info = 0;
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
#if defined(PARSEC_HAVE_CUDA)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZPOTRF, ((DagDouble_t)N));
    flops += FLOPS_ZPOTRI((DagDouble_t)N);

    /* initializing matrix structure */
    LDA = dplasma_imax( LDA, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        sym_two_dim_block_cyclic, (&dcA, matrix_ComplexDouble,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( parsec, (double)(N), uplo,
                    (parsec_tiled_matrix_dc_t *)&dcA, random_seed);
    if(loud > 3) printf("Done\n");

    if (async) {
        PASTE_CODE_ENQUEUE_KERNEL(parsec, zpoinv,
                                  (uplo, (parsec_tiled_matrix_dc_t*)&dcA, &info));
        PASTE_CODE_PROGRESS_KERNEL(parsec, zpoinv);
        dplasma_zpoinv_Destruct( PARSEC_zpoinv );
    }
    else {
        SYNC_TIME_START();
        info = dplasma_zpoinv_sync( parsec, uplo, (parsec_tiled_matrix_dc_t*)&dcA );
        SYNC_TIME_PRINT(rank, ("zpoinv\tPxQ= %3d %-3d NB= %4d N= %7d : %14f gflops\n",
                               P, Q, NB, N,
                               gflops=(flops/1e9)/sync_time_elapsed));
    }

    if( 0 == rank && info != 0 ) {
        printf("-- Factorization is suspicious (info = %d) ! \n", info);
        ret |= 1;
    }
    if( !info && check ) {
        /* Check the factorization */
        PASTE_CODE_ALLOCATE_MATRIX(dcA0, check,
            two_dim_block_cyclic, (&dcA0, matrix_ComplexDouble, matrix_Tile, nodes, rank,
                                   MB, NB, LDA, N, 0, 0, N, N, 1, 1, P));
        dplasma_zplghe( parsec, (double)(N), PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcA0, random_seed);

        ret |= check_zpoinv( parsec, (rank == 0) ? loud : 0, uplo,
                             (parsec_tiled_matrix_dc_t *)&dcA0,
                             (parsec_tiled_matrix_dc_t *)&dcA );

        if (ret) {
            printf("-- Innversion is suspicious ! \n");
        }
        else
        {
            printf("-- Inversion is CORRECT ! \n");
        }

        /* Cleanup */
        parsec_data_free(dcA0.mat); dcA0.mat = NULL;
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0 );
    }

    parsec_data_free(dcA.mat); dcA.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

    cleanup_parsec(parsec, iparam);
    return ret;
}
