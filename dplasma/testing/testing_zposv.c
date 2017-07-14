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

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info = 0;
    int u, t1, t2;
    int info_solve = 0;
    int info_facto = 0;
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

    /* initializing matrix structure */
    LDA = max( LDA, N );
    LDB = max( LDB, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(dcA0, 1,
        two_dim_block_cyclic, (&dcA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               N, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
        two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(dcX, 1,
        two_dim_block_cyclic, (&dcX, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplghe( parsec, (double)(N), PlasmaUpperLower,
                    (parsec_tiled_matrix_dc_t *)&dcA0, 3872);
    if(loud > 2) printf("Done\n");

    for ( u=0; u<2; u++) {
        if ( uplo[u] == PlasmaUpper ) {
            t1 = PlasmaConjTrans; t2 = PlasmaNoTrans;
        } else {
            t1 = PlasmaNoTrans; t2 = PlasmaConjTrans;
        }

        PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
            sym_two_dim_block_cyclic, (&dcA, matrix_ComplexDouble,
                                       nodes, rank, MB, NB, LDA, N, 0, 0,
                                       N, N, P, uplo[u]));

        /*********************************************************************
         *               First Check ( ZPOSV )
         */
        if ( rank == 0 ) {
            printf("***************************************************\n");
        }
        /* Create A and X */
        dplasma_zlacpy( parsec, uplo[u],
                        (parsec_tiled_matrix_dc_t *)&dcA0, (parsec_tiled_matrix_dc_t *)&dcA );
        dplasma_zplrnt( parsec, 0,
                        (parsec_tiled_matrix_dc_t *)&dcB, 2354);
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcB,  (parsec_tiled_matrix_dc_t *)&dcX );

        /* Compute */
        if ( loud > 2 ) printf("Compute ... ... ");
        info = dplasma_zposv(parsec, uplo[u],
                             (parsec_tiled_matrix_dc_t *)&dcA,
                             (parsec_tiled_matrix_dc_t *)&dcX );
        if ( loud > 2 ) printf("Done\n");
        if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

        /* Check the factorization */
        if ( info == 0 ) {
            info_facto = check_zpotrf( parsec, (rank == 0) ? loud : 0, uplo[u],
                                       (parsec_tiled_matrix_dc_t *)&dcA,
                                       (parsec_tiled_matrix_dc_t *)&dcA0);

            info_solve = check_zaxmb( parsec, (rank == 0) ? loud : 0, uplo[u],
                                      (parsec_tiled_matrix_dc_t *)&dcA0,
                                      (parsec_tiled_matrix_dc_t *)&dcB,
                                      (parsec_tiled_matrix_dc_t *)&dcX);
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
        dplasma_zlacpy( parsec, uplo[u],
                        (parsec_tiled_matrix_dc_t *)&dcA0, (parsec_tiled_matrix_dc_t *)&dcA );
        dplasma_zplrnt( parsec, 0,
                        (parsec_tiled_matrix_dc_t *)&dcB, 2354);
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcB,  (parsec_tiled_matrix_dc_t *)&dcX );

        /* Compute */
        if ( loud > 2 ) printf("Compute ... ... ");
        info = dplasma_zpotrf(parsec, uplo[u],
                              (parsec_tiled_matrix_dc_t *)&dcA );
        if ( info == 0 ) {
            dplasma_zpotrs(parsec, uplo[u],
                           (parsec_tiled_matrix_dc_t *)&dcA,
                           (parsec_tiled_matrix_dc_t *)&dcX );
        }
        if ( loud > 2 ) printf("Done\n");
        if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

        /* Check the solution */
        if ( info == 0 ) {
            info_facto = check_zpotrf( parsec, (rank == 0) ? loud : 0, uplo[u],
                                       (parsec_tiled_matrix_dc_t *)&dcA,
                                       (parsec_tiled_matrix_dc_t *)&dcA0 );

            info_solve = check_zaxmb( parsec, (rank == 0) ? loud : 0, uplo[u],
                                      (parsec_tiled_matrix_dc_t *)&dcA0,
                                      (parsec_tiled_matrix_dc_t *)&dcB,
                                      (parsec_tiled_matrix_dc_t *)&dcX );
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
        dplasma_zlacpy( parsec, uplo[u],
                        (parsec_tiled_matrix_dc_t *)&dcA0, (parsec_tiled_matrix_dc_t *)&dcA );
        dplasma_zplrnt( parsec, 0,
                        (parsec_tiled_matrix_dc_t *)&dcB, 2354);
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcB,  (parsec_tiled_matrix_dc_t *)&dcX );

        /* Compute */
        if ( loud > 2 ) printf("Compute ... ... ");
        info = dplasma_zpotrf(parsec, uplo[u], (parsec_tiled_matrix_dc_t *)&dcA );
        if ( info == 0 ) {
            dplasma_ztrsm(parsec, PlasmaLeft, uplo[u], t1, PlasmaNonUnit, 1.0,
                          (parsec_tiled_matrix_dc_t *)&dcA,
                          (parsec_tiled_matrix_dc_t *)&dcX);
            dplasma_ztrsm(parsec, PlasmaLeft, uplo[u], t2, PlasmaNonUnit, 1.0,
                          (parsec_tiled_matrix_dc_t *)&dcA,
                          (parsec_tiled_matrix_dc_t *)&dcX);
        }
        if ( loud > 2 ) printf("Done\n");
        if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

        /* Check the solution */
        if ( info == 0 ) {
            info_facto = check_zpotrf( parsec, (rank == 0) ? loud : 0, uplo[u],
                                       (parsec_tiled_matrix_dc_t *)&dcA,
                                       (parsec_tiled_matrix_dc_t *)&dcA0 );

            info_solve = check_zaxmb( parsec, (rank == 0) ? loud : 0, uplo[u],
                                      (parsec_tiled_matrix_dc_t *)&dcA0,
                                      (parsec_tiled_matrix_dc_t *)&dcB,
                                      (parsec_tiled_matrix_dc_t *)&dcX );
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
        dplasma_zlacpy( parsec, uplo[u],
                        (parsec_tiled_matrix_dc_t *)&dcA0, (parsec_tiled_matrix_dc_t *)&dcA );

        /* Compute */
        if ( loud > 2 ) printf("Compute ... ... ");
        info = dplasma_zpotrf(parsec, uplo[u], (parsec_tiled_matrix_dc_t *)&dcA );

        if ( info == 0 ) {
            info = dplasma_zpotri(parsec, uplo[u], (parsec_tiled_matrix_dc_t *)&dcA );
        }
        if ( loud > 2 ) printf("Done\n");
        if ( info != 0 ) printf("%d: Info = %d\n", rank, info);

        /* Check the solution */
        if ( info == 0 ) {
            info_solve = check_zpoinv( parsec, (rank == 0) ? loud : 0, uplo[u],
                                       (parsec_tiled_matrix_dc_t *)&dcA0,
                                       (parsec_tiled_matrix_dc_t *)&dcA);
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

        parsec_data_free(dcA.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    }

    parsec_data_free(dcA0.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0);
    parsec_data_free(dcB.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
    parsec_data_free(dcX.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcX);

    cleanup_parsec(parsec, iparam);

    return ret;
}
