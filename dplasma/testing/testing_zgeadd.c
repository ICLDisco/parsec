/*
 * Copyright (c) 2009-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "flops.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dplasma/cores/dplasma_zcores.h"

static int check_tr_solution( parsec_context_t *parsec, int loud,
                              PLASMA_enum uplo, PLASMA_enum trans,
                              parsec_complex64_t alpha,
                              int Am, int An, parsec_tiled_matrix_dc_t *dcA,
                              parsec_complex64_t beta,
                              int M,  int N,  parsec_tiled_matrix_dc_t *dcC,
                              parsec_tiled_matrix_dc_t *dcC2 );

static int check_ge_solution( parsec_context_t *parsec, int loud,
                              PLASMA_enum trans,
                              parsec_complex64_t alpha,
                              int Am, int An, parsec_tiled_matrix_dc_t *dcA,
                              parsec_complex64_t beta,
                              int M,  int N,  parsec_tiled_matrix_dc_t *dcC,
                              parsec_tiled_matrix_dc_t *dcC2 );

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    int Aseed = 3872;
    int Cseed = 2873;
    int tA, u, Am, An;

    parsec_complex64_t alpha = 0.43;
    parsec_complex64_t beta  = 0.78;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
    beta  += I * 0.21;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(PARSEC_HAVE_CUDA) && 1
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    LDA = max(LDA, max(M, K));
    LDB = max(LDB, max(K, N));
    LDC = max(LDC, M);

    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        two_dim_block_cyclic, (&dcC, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));


    PASTE_CODE_ALLOCATE_MATRIX(dcC0, check,
        two_dim_block_cyclic, (&dcC0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));

    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC0, Cseed);

#if defined(PRECISION_z) || defined(PRECISION_c)
    for(tA=0; tA<3; tA++) {
#else
    for(tA=0; tA<2; tA++) {
#endif
        if ( trans[tA] == PlasmaNoTrans ) {
            Am = M; An = N;
        } else {
            Am = N; An = M;
        }

        PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
            two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDA, LDA, 0, 0,
                                   Am, An, SMB, SNB, P));

        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);

        for (u=0; u<2; u++) {
            if ( rank == 0 ) {
                printf("***************************************************\n");
                printf(" ----- TESTING ZTRADD (%s, %s) -------- \n",
                       uplostr[u], transstr[tA]);
            }

            /* matrix generation */
            if(loud) printf("Generate matrices ... ");
            dplasma_zlacpy( parsec, PlasmaUpperLower,
                            (parsec_tiled_matrix_dc_t *)&dcC0, (parsec_tiled_matrix_dc_t *)&dcC );
            if(loud) printf("Done\n");

            /* Create GEMM PaRSEC */
            if(loud) printf("Compute ... ... ");
            dplasma_ztradd(parsec, uplo[u], trans[tA],
                           (parsec_complex64_t)alpha,
                           (parsec_tiled_matrix_dc_t *)&dcA,
                           (parsec_complex64_t)beta,
                           (parsec_tiled_matrix_dc_t *)&dcC);
            if(loud) printf("Done\n");

            /* Check the solution */
            info_solution = check_tr_solution( parsec, (rank == 0) ? loud : 0,
                                               uplo[u], trans[tA],
                                               alpha, Am, An,
                                               (parsec_tiled_matrix_dc_t *)&dcA,
                                               beta,  M,  N,
                                               (parsec_tiled_matrix_dc_t *)&dcC0,
                                               (parsec_tiled_matrix_dc_t *)&dcC );
            if ( rank == 0 ) {
                if (info_solution == 0) {
                    printf(" ---- TESTING ZTRADD (%s, %s) ...... PASSED !\n",
                           uplostr[u], transstr[tA]);
                }
                else {
                    printf(" ---- TESTING ZTRADD (%s, %s) ... FAILED !\n",
                           uplostr[u], transstr[tA]);
                }
                printf("***************************************************\n");
            }
        }

        parsec_data_free(dcA.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    }
#if defined(_UNUSED_)
    }
#endif

#if defined(PRECISION_z) || defined(PRECISION_c)
    for(tA=0; tA<3; tA++) {
#else
    for(tA=0; tA<2; tA++) {
#endif
        if ( trans[tA] == PlasmaNoTrans ) {
            Am = M; An = N;
        } else {
            Am = N; An = M;
        }

        PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
                                   two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                                          nodes, rank, MB, NB, LDA, LDA, 0, 0,
                                                          Am, An, SMB, SNB, P));

        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);

        if ( rank == 0 ) {
            printf("***************************************************\n");
            printf(" ----- TESTING ZGEADD (%s) -------- \n",
                   transstr[tA]);
        }

        /* matrix generation */
        if(loud) printf("Generate matrices ... ");
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcC0, (parsec_tiled_matrix_dc_t *)&dcC );
        if(loud) printf("Done\n");

        /* Create GEMM PaRSEC */
        if(loud) printf("Compute ... ... ");
        dplasma_zgeadd(parsec, trans[tA],
                       (parsec_complex64_t)alpha,
                       (parsec_tiled_matrix_dc_t *)&dcA,
                       (parsec_complex64_t)beta,
                       (parsec_tiled_matrix_dc_t *)&dcC);
        if(loud) printf("Done\n");

        /* Check the solution */
        info_solution = check_ge_solution( parsec, (rank == 0) ? loud : 0,
                                           trans[tA],
                                           alpha, Am, An,
                                           (parsec_tiled_matrix_dc_t *)&dcA,
                                           beta,  M,  N,
                                           (parsec_tiled_matrix_dc_t *)&dcC0,
                                           (parsec_tiled_matrix_dc_t *)&dcC );
        if ( rank == 0 ) {
            if (info_solution == 0) {
                printf(" ---- TESTING ZGEADD (%s) ...... PASSED !\n",
                       transstr[tA]);
            }
            else {
                printf(" ---- TESTING ZGEADD (%s) ... FAILED !\n",
                       transstr[tA]);
            }
            printf("***************************************************\n");
        }

        parsec_data_free(dcA.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    }
#if defined(_UNUSED_)
    }
#endif

    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC);
    parsec_data_free(dcC0.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC0);

    cleanup_parsec(parsec, iparam);

    return info_solution;
}


/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_tr_solution( parsec_context_t *parsec, int loud,
                              PLASMA_enum uplo, PLASMA_enum trans,
                              parsec_complex64_t alpha,
                              int Am, int An, parsec_tiled_matrix_dc_t *dcA,
                              parsec_complex64_t beta,
                              int M,  int N,  parsec_tiled_matrix_dc_t *dcC,
                              parsec_tiled_matrix_dc_t *dcC2 )
{
    int info_solution = 1;
    double Anorm, Cinitnorm, Cdplasmanorm, Rnorm;
    double eps, result;
    parsec_complex64_t mzone = (parsec_complex64_t)-1.;
    int MB = dcC2->mb;
    int NB = dcC2->nb;
    int LDA = Am;
    int LDC = M;
    int rank  = dcC2->super.myrank;

    eps = LAPACKE_dlamch_work('e');

    if ( ((trans == PlasmaNoTrans) && (uplo == PlasmaLower)) ||
         ((trans != PlasmaNoTrans) && (uplo == PlasmaUpper)) )
    {
        Anorm = dplasma_zlantr( parsec, PlasmaFrobeniusNorm, PlasmaLower,
                                PlasmaNonUnit, dcA );
    }
    else
    {
        Anorm = dplasma_zlantr( parsec, PlasmaFrobeniusNorm, PlasmaUpper,
                                PlasmaNonUnit, dcA );
    }
    Cinitnorm    = dplasma_zlantr( parsec, PlasmaFrobeniusNorm, uplo, PlasmaNonUnit, dcC  );
    Cdplasmanorm = dplasma_zlantr( parsec, PlasmaFrobeniusNorm, uplo, PlasmaNonUnit, dcC2 );

    PASTE_CODE_ALLOCATE_MATRIX(localA, 1,
                               two_dim_block_cyclic, (&localA, matrix_ComplexDouble, matrix_Lapack,
                                                      1, rank, MB, NB, LDA, An, 0, 0,
                                                      Am, An, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(localC, 1,
                               two_dim_block_cyclic, (&localC, matrix_ComplexDouble, matrix_Lapack,
                                                      1, rank, MB, NB, LDC, N, 0, 0,
                                                      M, N, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(localC2, 1,
                               two_dim_block_cyclic, (&localC2, matrix_ComplexDouble, matrix_Lapack,
                                                      1, rank, MB, NB, LDC, N, 0, 0,
                                                      M, N, 1, 1, 1));

    dplasma_zlacpy( parsec, PlasmaUpperLower, dcA,  (parsec_tiled_matrix_dc_t *)&localA  );
    dplasma_zlacpy( parsec, PlasmaUpperLower, dcC,  (parsec_tiled_matrix_dc_t *)&localC  );
    dplasma_zlacpy( parsec, PlasmaUpperLower, dcC2, (parsec_tiled_matrix_dc_t *)&localC2 );

    if ( rank == 0 ) {
        parsec_complex64_t *A  = localA.mat;
        parsec_complex64_t *C  = localC.mat;
        parsec_complex64_t *C2 = localC2.mat;

        dplasma_core_ztradd( uplo, trans, M, N, alpha, A, LDA, beta, C, LDC );
        cblas_zaxpy( LDC * N, CBLAS_SADDR(mzone), C, 1, C2, 1);
    }

    Rnorm = dplasma_zlange( parsec, PlasmaMaxNorm, (parsec_tiled_matrix_dc_t*)&localC2 );

    result = Rnorm / (Cinitnorm * eps);

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||C||_inf = %e\n"
                   "  ||dplasma(a*A*C)||_inf = %e, ||R||_m = %e, res = %e\n",
                   Anorm, Cinitnorm, Cdplasmanorm, Rnorm, result);
        }

        if (  isinf(Cdplasmanorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
            info_solution = 1;
        }
        else {
            info_solution = 0;
        }
    }

#if defined(PARSEC_HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    parsec_data_free(localA.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&localA);
    parsec_data_free(localC.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&localC);
    parsec_data_free(localC2.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&localC2);

    return info_solution;
}

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_ge_solution( parsec_context_t *parsec, int loud,
                              PLASMA_enum trans,
                              parsec_complex64_t alpha,
                              int Am, int An, parsec_tiled_matrix_dc_t *dcA,
                              parsec_complex64_t beta,
                              int M,  int N,  parsec_tiled_matrix_dc_t *dcC,
                              parsec_tiled_matrix_dc_t *dcC2 )
{
    int info_solution = 1;
    double Anorm, Cinitnorm, Cdplasmanorm, Rnorm;
    double eps, result;
    parsec_complex64_t mzone = (parsec_complex64_t)-1.;
    int MB = dcC2->mb;
    int NB = dcC2->nb;
    int LDA = Am;
    int LDC = M;
    int rank  = dcC2->super.myrank;

    eps = LAPACKE_dlamch_work('e');

    Anorm        = dplasma_zlange( parsec, PlasmaFrobeniusNorm, dcA  );
    Cinitnorm    = dplasma_zlange( parsec, PlasmaFrobeniusNorm, dcC  );
    Cdplasmanorm = dplasma_zlange( parsec, PlasmaFrobeniusNorm, dcC2 );

    PASTE_CODE_ALLOCATE_MATRIX(localA, 1,
                               two_dim_block_cyclic, (&localA, matrix_ComplexDouble, matrix_Lapack,
                                                      1, rank, MB, NB, LDA, An, 0, 0,
                                                      Am, An, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(localC, 1,
                               two_dim_block_cyclic, (&localC, matrix_ComplexDouble, matrix_Lapack,
                                                      1, rank, MB, NB, LDC, N, 0, 0,
                                                      M, N, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(localC2, 1,
                               two_dim_block_cyclic, (&localC2, matrix_ComplexDouble, matrix_Lapack,
                                                      1, rank, MB, NB, LDC, N, 0, 0,
                                                      M, N, 1, 1, 1));

    dplasma_zlacpy( parsec, PlasmaUpperLower, dcA,  (parsec_tiled_matrix_dc_t *)&localA  );
    dplasma_zlacpy( parsec, PlasmaUpperLower, dcC,  (parsec_tiled_matrix_dc_t *)&localC  );
    dplasma_zlacpy( parsec, PlasmaUpperLower, dcC2, (parsec_tiled_matrix_dc_t *)&localC2 );

    if ( rank == 0 ) {
        parsec_complex64_t *A  = localA.mat;
        parsec_complex64_t *C  = localC.mat;
        parsec_complex64_t *C2 = localC2.mat;

        dplasma_core_zgeadd( trans, M, N, alpha, A, LDA, beta, C, LDC );
        cblas_zaxpy( LDC * N, CBLAS_SADDR(mzone), C, 1, C2, 1);
    }

    Rnorm = dplasma_zlange( parsec, PlasmaMaxNorm, (parsec_tiled_matrix_dc_t*)&localC2 );

    result = Rnorm / (Cinitnorm * eps);

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||C||_inf = %e\n"
                   "  ||dplasma(a*A*C)||_inf = %e, ||R||_m = %e, res = %e\n",
                   Anorm, Cinitnorm, Cdplasmanorm, Rnorm, result);
        }

        if (  isinf(Cdplasmanorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
            info_solution = 1;
        }
        else {
            info_solution = 0;
        }
    }

#if defined(PARSEC_HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    parsec_data_free(localA.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&localA);
    parsec_data_free(localC.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&localC);
    parsec_data_free(localC2.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&localC2);

    return info_solution;
}
