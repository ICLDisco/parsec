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
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dplasma/cores/dplasma_zcores.h"

static int check_tr_solution( dague_context_t *dague, int loud,
                              PLASMA_enum uplo, PLASMA_enum trans,
                              dague_complex64_t alpha,
                              int Am, int An, tiled_matrix_desc_t *ddescA,
                              dague_complex64_t beta,
                              int M,  int N,  tiled_matrix_desc_t *ddescC,
                              tiled_matrix_desc_t *ddescC2 );

static int check_ge_solution( dague_context_t *dague, int loud,
                              PLASMA_enum trans,
                              dague_complex64_t alpha,
                              int Am, int An, tiled_matrix_desc_t *ddescA,
                              dague_complex64_t beta,
                              int M,  int N,  tiled_matrix_desc_t *ddescC,
                              tiled_matrix_desc_t *ddescC2 );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    int Aseed = 3872;
    int Cseed = 2873;
    int tA, u, Am, An;

    dague_complex64_t alpha = 0.43;
    dague_complex64_t beta  = 0.78;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
    beta  += I * 0.21;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(HAVE_CUDA) && 1
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    LDA = max(LDA, max(M, K));
    LDB = max(LDB, max(K, N));
    LDC = max(LDC, M);

    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));


    PASTE_CODE_ALLOCATE_MATRIX(ddescC2, check,
        two_dim_block_cyclic, (&ddescC2, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));

    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC2, Cseed);

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

        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                                   two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                                          nodes, rank, MB, NB, LDA, LDA, 0, 0,
                                                          Am, An, SMB, SNB, P));

        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, Aseed);

        for (u=0; u<2; u++) {
            if ( rank == 0 ) {
                printf("***************************************************\n");
                printf(" ----- TESTING ZTRADD (%s, %s) -------- \n",
                       uplostr[u], transstr[tA]);
            }

            /* matrix generation */
            if(loud) printf("Generate matrices ... ");
            dplasma_zlacpy( dague, PlasmaUpperLower,
                            (tiled_matrix_desc_t *)&ddescC2, (tiled_matrix_desc_t *)&ddescC );
            if(loud) printf("Done\n");

            /* Create GEMM DAGuE */
            if(loud) printf("Compute ... ... ");
            dplasma_ztradd(dague, uplo[u], trans[tA],
                           (dague_complex64_t)alpha,
                           (tiled_matrix_desc_t *)&ddescA,
                           (dague_complex64_t)beta,
                           (tiled_matrix_desc_t *)&ddescC);
            if(loud) printf("Done\n");

            /* Check the solution */
            info_solution = check_tr_solution( dague, (rank == 0) ? loud : 0,
                                               uplo[u], trans[tA],
                                               alpha, Am, An,
                                               (tiled_matrix_desc_t *)&ddescA,
                                               beta,  M,  N,
                                               (tiled_matrix_desc_t *)&ddescC2,
                                               (tiled_matrix_desc_t *)&ddescC );
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

        dague_data_free(ddescA.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
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

        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                                   two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                                          nodes, rank, MB, NB, LDA, LDA, 0, 0,
                                                          Am, An, SMB, SNB, P));

        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, Aseed);

        if ( rank == 0 ) {
            printf("***************************************************\n");
            printf(" ----- TESTING ZGEADD (%s) -------- \n",
                   transstr[tA]);
        }

        /* matrix generation */
        if(loud) printf("Generate matrices ... ");
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescC2, (tiled_matrix_desc_t *)&ddescC );
        if(loud) printf("Done\n");

        /* Create GEMM DAGuE */
        if(loud) printf("Compute ... ... ");
        dplasma_zgeadd(dague, trans[tA],
                       (dague_complex64_t)alpha,
                       (tiled_matrix_desc_t *)&ddescA,
                       (dague_complex64_t)beta,
                       (tiled_matrix_desc_t *)&ddescC);
        if(loud) printf("Done\n");

        /* Check the solution */
        info_solution = check_ge_solution( dague, (rank == 0) ? loud : 0,
                                           trans[tA],
                                           alpha, Am, An,
                                           (tiled_matrix_desc_t *)&ddescA,
                                           beta,  M,  N,
                                           (tiled_matrix_desc_t *)&ddescC2,
                                           (tiled_matrix_desc_t *)&ddescC );
        if ( rank == 0 ) {
            if (info_solution == 0) {
                printf(" ---- TESTING ZGEADD (%s, %s) ...... PASSED !\n",
                       uplostr[u], transstr[tA]);
            }
            else {
                    printf(" ---- TESTING ZGEADD (%s, %s) ... FAILED !\n",
                           uplostr[u], transstr[tA]);
                }
                printf("***************************************************\n");
        }

        dague_data_free(ddescA.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    }
#if defined(_UNUSED_)
    }
#endif

    dague_data_free(ddescC.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC);
    dague_data_free(ddescC2.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC2);

    cleanup_dague(dague, iparam);

    return info_solution;
}


/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_tr_solution( dague_context_t *dague, int loud,
                              PLASMA_enum uplo, PLASMA_enum trans,
                              dague_complex64_t alpha,
                              int Am, int An, tiled_matrix_desc_t *ddescA,
                              dague_complex64_t beta,
                              int M,  int N,  tiled_matrix_desc_t *ddescC,
                              tiled_matrix_desc_t *ddescC2 )
{
    int info_solution = 1;
    double Anorm, Cinitnorm, Cdplasmanorm, Rnorm;
    double eps, result;
    dague_complex64_t mzone = (dague_complex64_t)-1.;
    int MB = ddescC2->mb;
    int NB = ddescC2->nb;
    int LDA = Am;
    int LDC = M;
    int rank  = ddescC2->super.myrank;

    eps = LAPACKE_dlamch_work('e');

    if ( ((trans == PlasmaNoTrans) && (uplo == PlasmaLower)) ||
         ((trans != PlasmaNoTrans) && (uplo == PlasmaUpper)) )
    {
        Anorm = dplasma_zlantr( dague, PlasmaFrobeniusNorm, PlasmaLower,
                                PlasmaNonUnit, ddescA );
    }
    else
    {
        Anorm = dplasma_zlantr( dague, PlasmaFrobeniusNorm, PlasmaUpper,
                                PlasmaNonUnit, ddescA );
    }
    Cinitnorm    = dplasma_zlantr( dague, PlasmaFrobeniusNorm, uplo, PlasmaNonUnit, ddescC  );
    Cdplasmanorm = dplasma_zlantr( dague, PlasmaFrobeniusNorm, uplo, PlasmaNonUnit, ddescC2 );

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

    dplasma_zlacpy( dague, PlasmaUpperLower, ddescA,  (tiled_matrix_desc_t *)&localA  );
    dplasma_zlacpy( dague, PlasmaUpperLower, ddescC,  (tiled_matrix_desc_t *)&localC  );
    dplasma_zlacpy( dague, PlasmaUpperLower, ddescC2, (tiled_matrix_desc_t *)&localC2 );

    if ( rank == 0 ) {
        dague_complex64_t *A  = localA.mat;
        dague_complex64_t *C  = localC.mat;
        dague_complex64_t *C2 = localC2.mat;

        dplasma_core_ztradd( uplo, trans, M, N, alpha, A, LDA, beta, C, LDC );
        cblas_zaxpy( LDC * N, CBLAS_SADDR(mzone), C, 1, C2, 1);
    }

    Rnorm = dplasma_zlange( dague, PlasmaMaxNorm, (tiled_matrix_desc_t*)&localC2 );

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

#if defined(HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    dague_data_free(localA.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&localA);
    dague_data_free(localC.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&localC);
    dague_data_free(localC2.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&localC2);

    return info_solution;
}

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_ge_solution( dague_context_t *dague, int loud,
                              PLASMA_enum trans,
                              dague_complex64_t alpha,
                              int Am, int An, tiled_matrix_desc_t *ddescA,
                              dague_complex64_t beta,
                              int M,  int N,  tiled_matrix_desc_t *ddescC,
                              tiled_matrix_desc_t *ddescC2 )
{
    int info_solution = 1;
    double Anorm, Cinitnorm, Cdplasmanorm, Rnorm;
    double eps, result;
    dague_complex64_t mzone = (dague_complex64_t)-1.;
    int MB = ddescC2->mb;
    int NB = ddescC2->nb;
    int LDA = Am;
    int LDC = M;
    int rank  = ddescC2->super.myrank;

    eps = LAPACKE_dlamch_work('e');

    Anorm        = dplasma_zlange( dague, PlasmaFrobeniusNorm, ddescA  );
    Cinitnorm    = dplasma_zlange( dague, PlasmaFrobeniusNorm, ddescC  );
    Cdplasmanorm = dplasma_zlange( dague, PlasmaFrobeniusNorm, ddescC2 );

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

    dplasma_zlacpy( dague, PlasmaUpperLower, ddescA,  (tiled_matrix_desc_t *)&localA  );
    dplasma_zlacpy( dague, PlasmaUpperLower, ddescC,  (tiled_matrix_desc_t *)&localC  );
    dplasma_zlacpy( dague, PlasmaUpperLower, ddescC2, (tiled_matrix_desc_t *)&localC2 );

    if ( rank == 0 ) {
        dague_complex64_t *A  = localA.mat;
        dague_complex64_t *C  = localC.mat;
        dague_complex64_t *C2 = localC2.mat;

        dplasma_core_zgeadd( trans, M, N, alpha, A, LDA, beta, C, LDC );
        cblas_zaxpy( LDC * N, CBLAS_SADDR(mzone), C, 1, C2, 1);
    }

    Rnorm = dplasma_zlange( dague, PlasmaMaxNorm, (tiled_matrix_desc_t*)&localC2 );

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

#if defined(HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    dague_data_free(localA.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&localA);
    dague_data_free(localC.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&localC);
    dague_data_free(localC2.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&localC2);

    return info_solution;
}
