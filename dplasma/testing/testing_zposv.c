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
    int info = 0;
    int u, t1, t2;
    int info_solve = 0;
    int info_facto = 0;
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

    /* initializing matrix structure */
    LDA = max( LDA, N );
    LDB = max( LDB, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, 1,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               N, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, 1,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplghe( dague, (double)(N), PlasmaUpperLower,
                    (tiled_matrix_desc_t *)&ddescA0, 3872);
    if(loud > 2) printf("Done\n");

    for ( u=0; u<2; u++) {
        if ( uplo[u] == PlasmaUpper ) {
            t1 = PlasmaConjTrans; t2 = PlasmaNoTrans;
        } else {
            t1 = PlasmaNoTrans; t2 = PlasmaConjTrans;
        }

        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
            sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                       nodes, rank, MB, NB, LDA, N, 0, 0,
                                       N, N, P, uplo[u]));

        /*********************************************************************
         *               First Check ( ZPOSV )
         */
        if ( rank == 0 ) {
            printf("***************************************************\n");
        }
        /* Create A and X */
        dplasma_zlacpy( dague, uplo[u],
                        (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
        dplasma_zplrnt( dague, 0,
                        (tiled_matrix_desc_t *)&ddescB, 2354);
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB,  (tiled_matrix_desc_t *)&ddescX );

        /* Compute */
        if ( loud > 2 ) printf("Compute ... ... ");
        info = dplasma_zposv(dague, uplo[u],
                             (tiled_matrix_desc_t *)&ddescA,
                             (tiled_matrix_desc_t *)&ddescX );
        if ( loud > 2 ) printf("Done\n");
        if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

        /* Check the factorization */
        if ( info == 0 ) {
            info_facto = check_zpotrf( dague, (rank == 0) ? loud : 0, uplo[u],
                                       (tiled_matrix_desc_t *)&ddescA,
                                       (tiled_matrix_desc_t *)&ddescA0);

            info_solve = check_zaxmb( dague, (rank == 0) ? loud : 0, uplo[u],
                                      (tiled_matrix_desc_t *)&ddescA0,
                                      (tiled_matrix_desc_t *)&ddescB,
                                      (tiled_matrix_desc_t *)&ddescX);
        }
        if ( rank == 0 ) {
            if ( info_solve || info_facto || info ) {
                printf(" ----- TESTING ZPOSV (%s) ... FAILED !\n", uplostr[u]);
                ret |= 1;
            }
            else {
                printf(" ----- TESTING ZPOSV (%s) ....... PASSED !\n", uplostr[u]);
            }
            printf("***************************************************\n");
        }

        /*********************************************************************
         *               Second Check ( ZPOTRF + ZPOTRS )
         */
        if ( rank == 0 ) {
            printf("***************************************************\n");
        }

        /* Create A and X */
        dplasma_zlacpy( dague, uplo[u],
                        (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
        dplasma_zplrnt( dague, 0,
                        (tiled_matrix_desc_t *)&ddescB, 2354);
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB,  (tiled_matrix_desc_t *)&ddescX );

        /* Compute */
        if ( loud > 2 ) printf("Compute ... ... ");
        info = dplasma_zpotrf(dague, uplo[u],
                              (tiled_matrix_desc_t *)&ddescA );
        if ( info == 0 ) {
            dplasma_zpotrs(dague, uplo[u],
                           (tiled_matrix_desc_t *)&ddescA,
                           (tiled_matrix_desc_t *)&ddescX );
        }
        if ( loud > 2 ) printf("Done\n");
        if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

        /* Check the solution */
        if ( info == 0 ) {
            info_facto = check_zpotrf( dague, (rank == 0) ? loud : 0, uplo[u],
                                       (tiled_matrix_desc_t *)&ddescA,
                                       (tiled_matrix_desc_t *)&ddescA0 );

            info_solve = check_zaxmb( dague, (rank == 0) ? loud : 0, uplo[u],
                                      (tiled_matrix_desc_t *)&ddescA0,
                                      (tiled_matrix_desc_t *)&ddescB,
                                      (tiled_matrix_desc_t *)&ddescX );
        }
        if ( rank == 0 ) {
            if ( info_solve || info_facto || info ) {
                printf(" ----- TESTING ZPOTRF + ZPOTRS (%s) ... FAILED !\n", uplostr[u]);
                ret |= 1;
            }
            else {
                printf(" ----- TESTING ZPOTRF + ZPOTRS (%s) ....... PASSED !\n", uplostr[u]);
            }
            printf("***************************************************\n");
        }

        /*********************************************************************
         *               Third Check (ZPOTRF + ZTRSM + ZTRSM)
         */
        if ( rank == 0 ) {
            printf("***************************************************\n");
        }

        /* Create A and X */
        dplasma_zlacpy( dague, uplo[u],
                        (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
        dplasma_zplrnt( dague, 0,
                        (tiled_matrix_desc_t *)&ddescB, 2354);
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB,  (tiled_matrix_desc_t *)&ddescX );

        /* Compute */
        if ( loud > 2 ) printf("Compute ... ... ");
        info = dplasma_zpotrf(dague, uplo[u], (tiled_matrix_desc_t *)&ddescA );
        if ( info == 0 ) {
            dplasma_ztrsm(dague, PlasmaLeft, uplo[u], t1, PlasmaNonUnit, 1.0,
                          (tiled_matrix_desc_t *)&ddescA,
                          (tiled_matrix_desc_t *)&ddescX);
            dplasma_ztrsm(dague, PlasmaLeft, uplo[u], t2, PlasmaNonUnit, 1.0,
                          (tiled_matrix_desc_t *)&ddescA,
                          (tiled_matrix_desc_t *)&ddescX);
        }
        if ( loud > 2 ) printf("Done\n");
        if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

        /* Check the solution */
        if ( info == 0 ) {
            info_facto = check_zpotrf( dague, (rank == 0) ? loud : 0, uplo[u],
                                       (tiled_matrix_desc_t *)&ddescA,
                                       (tiled_matrix_desc_t *)&ddescA0 );

            info_solve = check_zaxmb( dague, (rank == 0) ? loud : 0, uplo[u],
                                      (tiled_matrix_desc_t *)&ddescA0,
                                      (tiled_matrix_desc_t *)&ddescB,
                                      (tiled_matrix_desc_t *)&ddescX );
        }

        if ( rank == 0 ) {
            if ( info_solve || info_facto || info ) {
                printf(" ----- TESTING ZPOTRF + ZTRSM + ZTRSM (%s) ... FAILED !\n", uplostr[u]);
                ret |= 1;
            }
            else {
                printf(" ----- TESTING ZPOTRF + ZTRSM + ZTRSM (%s) ....... PASSED !\n", uplostr[u]);
            }
            printf("***************************************************\n");
        }

        /*********************************************************************
         *               Fourth Check (ZPOTRF + ZPOTRI)
         */
        if ( rank == 0 ) {
            printf("***************************************************\n");
        }

        /* Create A and X */
        dplasma_zlacpy( dague, uplo[u],
                        (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );

        /* Compute */
        if ( loud > 2 ) printf("Compute ... ... ");
        info = dplasma_zpotrf(dague, uplo[u], (tiled_matrix_desc_t *)&ddescA );

        if ( info == 0 ) {
            info = dplasma_zpotri(dague, uplo[u], (tiled_matrix_desc_t *)&ddescA );
        }
        if ( loud > 2 ) printf("Done\n");
        if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

        /* Check the solution */
        if ( info == 0 ) {
            info_solve = check_zpoinv( dague, (rank == 0) ? loud : 0, uplo[u],
                                       (tiled_matrix_desc_t *)&ddescA0,
                                       (tiled_matrix_desc_t *)&ddescA);
        }

        if ( rank == 0 ) {
            if ( info_solve || info ) {
                printf(" ----- TESTING ZPOTRF + ZPOTRI (%s) ... FAILED !\n", uplostr[u]);
                ret |= 1;
            }
            else {
                printf(" ----- TESTING ZPOTRF + ZPOTRI (%s) ....... PASSED !\n", uplostr[u]);
            }
            printf("***************************************************\n");
        }

        dague_data_free(ddescA.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    }

    dague_data_free(ddescA0.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0);
    dague_data_free(ddescB.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);
    dague_data_free(ddescX.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescX);

    cleanup_dague(dague, iparam);

    return ret;
}
