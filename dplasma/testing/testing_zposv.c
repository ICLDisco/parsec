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

static int check_factorization( dague_context_t *dague, int loud, PLASMA_enum uplo,
                                tiled_matrix_desc_t *A,
                                tiled_matrix_desc_t *A0 );
static int check_solution( dague_context_t *dague, int loud, PLASMA_enum uplo,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX );

static int check_inverse( dague_context_t *dague, int loud,
                          PLASMA_enum uplo, int N,
                          tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *Ainv );

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
    dplasma_zplrnt( dague, 0,
                    (tiled_matrix_desc_t *)&ddescB, 2354);
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

        /* load the GPU kernel */
#if defined(HAVE_CUDA)
        if(iparam[IPARAM_NGPUS] > 0)
        {
            if(loud > 3) printf("+++ Load GPU kernel ... ");
            dague_gpu_data_register(dague,
                                    (dague_ddesc_t*)&ddescA,
                                    MT*NT, MB*NB*sizeof(dague_complex64_t) );
            if(loud > 3) printf("Done\n");
        }
#endif

        /*********************************************************************
         *               First Check ( ZPOSV )
         */
        if ( rank == 0 ) {
            printf("***************************************************\n");
        }
        /* Create A and X */
        dplasma_zlacpy( dague, uplo[u],
                        (tiled_matrix_desc_t *)&ddescA0, (tiled_matrix_desc_t *)&ddescA );
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
            info_facto = check_factorization( dague, (rank == 0) ? loud : 0, uplo[u],
                                              (tiled_matrix_desc_t *)&ddescA,
                                              (tiled_matrix_desc_t *)&ddescA0);

            info_solve = check_solution( dague, (rank == 0) ? loud : 0, uplo[u],
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
            info_facto = check_factorization( dague, (rank == 0) ? loud : 0, uplo[u],
                                              (tiled_matrix_desc_t *)&ddescA,
                                              (tiled_matrix_desc_t *)&ddescA0);

            info_solve = check_solution( dague, (rank == 0) ? loud : 0, uplo[u],
                                         (tiled_matrix_desc_t *)&ddescA0,
                                         (tiled_matrix_desc_t *)&ddescB,
                                         (tiled_matrix_desc_t *)&ddescX);
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
            info_facto = check_factorization( dague, (rank == 0) ? loud : 0, uplo[u],
                                              (tiled_matrix_desc_t *)&ddescA,
                                              (tiled_matrix_desc_t *)&ddescA0);

            info_solve = check_solution( dague, (rank == 0) ? loud : 0, uplo[u],
                                         (tiled_matrix_desc_t *)&ddescA0,
                                         (tiled_matrix_desc_t *)&ddescB,
                                         (tiled_matrix_desc_t *)&ddescX);
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
            info_solve = check_inverse( dague, (rank == 0) ? loud : 0, uplo[u], N,
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

#if defined(HAVE_CUDA)
        if(iparam[IPARAM_NGPUS] > 0) {
            dague_gpu_data_unregister((dague_ddesc_t*)&ddescA);
        }
#endif

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

static int check_factorization( dague_context_t *dague, int loud, PLASMA_enum uplo,
                                tiled_matrix_desc_t *A,
                                tiled_matrix_desc_t *A0 )
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A0;
    int info_factorization;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double result = 0.0;
    int M = A->m;
    int N = A->n;
    double eps = LAPACKE_dlamch_work('e');
    PLASMA_enum side;

    PASTE_CODE_ALLOCATE_MATRIX(L1, 1,
                               sym_two_dim_block_cyclic, (&L1, matrix_ComplexDouble,
                                                          A->super.nodes, twodA->grid.rank,
                                                          A->mb, A->nb, M, N, 0, 0,
                                                          M, N, twodA->grid.rows, uplo));
    PASTE_CODE_ALLOCATE_MATRIX(L2, 1,
                               two_dim_block_cyclic, (&L2, matrix_ComplexDouble, matrix_Tile,
                                                      A->super.nodes, twodA->grid.rank,
                                                      A->mb, A->nb, M, N, 0, 0,
                                                      M, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    dplasma_zlacpy( dague, uplo, A, (tiled_matrix_desc_t *)&L1 );
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0.,(tiled_matrix_desc_t *)&L2 );
    dplasma_zlacpy( dague, uplo, A, (tiled_matrix_desc_t *)&L2 );

    side = (uplo == PlasmaUpper ) ? PlasmaLeft : PlasmaRight;

    /* Compute L'L or U'U  */
    dplasma_ztrmm( dague, side, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0,
                   (tiled_matrix_desc_t*)&L1,
                   (tiled_matrix_desc_t*)&L2);

    /* compute L'L - A or U'U - A */
    dplasma_zgeadd( dague, uplo, -1.0, A0,
                   (tiled_matrix_desc_t*)&L2);

    Anorm = dplasma_zlanhe(dague, PlasmaInfNorm, uplo, A0);
    Rnorm = dplasma_zlanhe(dague, PlasmaInfNorm, uplo,
                           (tiled_matrix_desc_t*)&L2);

    result = Rnorm / ( Anorm * N * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Cholesky factorization \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e",
                    Anorm );
        if ( loud > 3 )
            printf( ", ||L'L-A||_oo = %e\n",
                    Rnorm );

        printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n", result);
    }

    if ( isnan(Rnorm)
         || isinf(Rnorm)
         || isnan(result)
         || isinf(result)
         || (result > 60.0) ) {
        if( loud ) printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else{
        if( loud ) printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    dague_data_free(L1.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&L1);
    dague_data_free(L2.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&L2);

    return info_factorization;
}

/*
 * This function destroy B
 */
static int check_solution( dague_context_t *dague, int loud, PLASMA_enum uplo,
                           tiled_matrix_desc_t *A,
                           tiled_matrix_desc_t *B,
                           tiled_matrix_desc_t *X )
{
    two_dim_block_cyclic_t *twodB = (two_dim_block_cyclic_t *)B;
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int N    = B->m;
    int NRHS = B->n;
    double eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(R, 1,
        two_dim_block_cyclic, (&R, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, twodB->grid.rank,
                               A->mb, A->nb, N, NRHS, 0, 0,
                               N, NRHS, twodB->grid.strows, twodB->grid.stcols, twodB->grid.rows));

    Anorm = dplasma_zlanhe(dague, PlasmaInfNorm, uplo, A);
    Bnorm = dplasma_zlange(dague, PlasmaInfNorm, B);
    Xnorm = dplasma_zlange(dague, PlasmaInfNorm, X);
    dplasma_zlacpy( dague, PlasmaUpperLower, B, (tiled_matrix_desc_t *)&R );

    /* Compute A*x */
    dplasma_zhemm( dague, PlasmaLeft, uplo, -1.0, A, X,
                   1.0, (tiled_matrix_desc_t *)&R);

    Rnorm = dplasma_zlange(dague, PlasmaInfNorm,
                           (tiled_matrix_desc_t *)&R);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * N * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );

        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
    }

    if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution is CORRECT ! \n");
        info_solution = 0;
    }

    dague_data_free(R.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&R);
    return info_solution;
}


/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_inverse( dague_context_t *dague, int loud,
                          PLASMA_enum uplo, int N,
                          tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *Ainv )
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A;
    int info_solution;
    double Anorm, Ainvnorm, Rnorm;
    double eps, result;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(Id, 1,
        two_dim_block_cyclic, (&Id, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, twodA->grid.rank,
                               A->mb, A->nb, N, N, 0, 0,
                               N, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&Id);

    /* Id - A^-1 * A */
    dplasma_zhemm(dague, PlasmaLeft, uplo,
                  -1., Ainv, A,
                  1., (tiled_matrix_desc_t *)&Id );

    Anorm    = dplasma_zlanhe( dague, PlasmaOneNorm, uplo, A );
    Ainvnorm = dplasma_zlanhe( dague, PlasmaOneNorm, uplo, Ainv );
    Rnorm    = dplasma_zlange( dague, PlasmaOneNorm, (tiled_matrix_desc_t*)&Id );

    result = Rnorm / ( (Anorm*Ainvnorm)*N*eps );
    if ( loud > 2 ) {
        printf("  ||A||_one = %e, ||A^(-1)||_one = %e, ||I - A * A^(-1)||_one = %e, result = %e\n",
               Anorm, Ainvnorm, Rnorm, result);
    }

    if ( isinf(Ainvnorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        info_solution = 1;
    }
    else {
        info_solution = 0;
    }

    dague_data_free(Id.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&Id);

    return info_solution;
}
