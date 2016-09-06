/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                           dague_complex64_t alpha,
                           int Am, int An, int Aseed,
                           int M,  int N,  int Cseed,
                           two_dim_block_cyclic_t *ddescCfinal );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    int Aseed = 3872;
    int Cseed = 2873;
    dague_complex64_t alpha = 3.5;
    tiled_matrix_desc_t *ddescA;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 4.2;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(DAGUE_HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    /* initializing matrix structure */
    int Am = max(M, N);
    LDA = max(LDA, Am);
    LDC = max(LDC, M);
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, 1,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, Am, 0, 0,
                               Am, Am, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));

    if(!check)
    {
        PLASMA_enum side  = PlasmaLeft;
        PLASMA_enum uplo  = PlasmaLower;
        PLASMA_enum trans = PlasmaNoTrans;
        PLASMA_enum diag  = PlasmaUnit;

        PASTE_CODE_FLOPS(FLOPS_ZTRMM, (side, (DagDouble_t)M, (DagDouble_t)N));

        /* Make A square */
        if (side == PlasmaLeft) {
            ddescA = tiled_matrix_submatrix( (tiled_matrix_desc_t *)&ddescA0, 0, 0, M, M );
        } else {
            ddescA = tiled_matrix_submatrix( (tiled_matrix_desc_t *)&ddescA0, 0, 0, N, N );
        }

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplghe( dague, 0., uplo, ddescA, Aseed);
        dplasma_zplrnt( dague, 0,        (tiled_matrix_desc_t *)&ddescC, Cseed);
        if(loud > 2) printf("Done\n");

        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, ztrmm,
                                  (side, uplo, trans, diag,
                                   1.0, ddescA,
                                   (tiled_matrix_desc_t *)&ddescC));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, ztrmm);

        dplasma_ztrmm_Destruct( DAGUE_ztrmm );
        free(ddescA);
    }
    else
    {
        int s, u, t, d;
        int info_solution;

        PASTE_CODE_ALLOCATE_MATRIX(ddescC2, 1,
            two_dim_block_cyclic, (&ddescC2, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDC, N, 0, 0,
                                   M, N, SMB, SNB, P));

        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC2, Cseed);

        for (s=0; s<2; s++) {
            /* Make A square */
            if (side[s] == PlasmaLeft) {
                Am = M;
                ddescA = tiled_matrix_submatrix( (tiled_matrix_desc_t *)&ddescA0, 0, 0, M, M );
            } else {
                Am = N;
                ddescA = tiled_matrix_submatrix( (tiled_matrix_desc_t *)&ddescA0, 0, 0, N, N );
            }
            dplasma_zplghe( dague, 0., PlasmaUpperLower, ddescA, Aseed);

            for (u=0; u<2; u++) {
#if defined(PRECISION_z) || defined(PRECISION_c)
                for (t=0; t<3; t++) {
#else
                for (t=0; t<2; t++) {
#endif
                    for (d=0; d<2; d++) {

                        if ( rank == 0 ) {
                            printf("***************************************************\n");
                            printf(" ----- TESTING ZTRMM (%s, %s, %s, %s) -------- \n",
                                   sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                        }

                        /* matrix generation */
                        printf("Generate matrices ... ");
                        dplasma_zlacpy( dague, PlasmaUpperLower,
                                        (tiled_matrix_desc_t *)&ddescC2, (tiled_matrix_desc_t *)&ddescC );
                        printf("Done\n");

                        /* Compute */
                        printf("Compute ... ... ");
                        dplasma_ztrmm(dague, side[s], uplo[u], trans[t], diag[d],
                                      alpha, ddescA, (tiled_matrix_desc_t *)&ddescC);
                        printf("Done\n");

                        /* Check the solution */
                        info_solution = check_solution(dague, rank == 0 ? loud : 0,
                                                       side[s], uplo[u], trans[t], diag[d],
                                                       alpha, Am, Am, Aseed,
                                                              M,  N,  Cseed,
                                                       &ddescC);
                        if ( rank == 0 ) {
                            if (info_solution == 0) {
                                printf(" ---- TESTING ZTRMM (%s, %s, %s, %s) ...... PASSED !\n",
                                       sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                            }
                            else {
                                printf(" ---- TESTING ZTRMM (%s, %s, %s, %s) ... FAILED !\n",
                                       sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                                ret |= 1;
                            }
                            printf("***************************************************\n");
                        }
                    }
                }
#ifdef __UNUSED__
                }
#endif
            }
            free(ddescA);
        }
        dague_data_free(ddescC2.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC2);
    }

    dague_data_free(ddescA0.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0);
    dague_data_free(ddescC.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC);

    cleanup_dague(dague, iparam);

    return ret;
}


/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                           dague_complex64_t alpha,
                           int Am, int An, int Aseed,
                           int M,  int N,  int Cseed,
                           two_dim_block_cyclic_t *ddescCfinal )
{
    int info_solution = 1;
    double Anorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm;
    double eps, result;
    int MB = ddescCfinal->super.mb;
    int NB = ddescCfinal->super.nb;
    int LDA = Am;
    int LDC = M;
    int rank  = ddescCfinal->super.super.myrank;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDA, An, 0, 0,
                               Am, An, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1));

    dplasma_zplghe( dague, 0., PlasmaUpperLower, (tiled_matrix_desc_t *)&ddescA, Aseed);
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC, Cseed );

    Anorm        = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescA );
    Cinitnorm    = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescC );
    Cdplasmanorm = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)ddescCfinal );

    if ( rank == 0 ) {
        cblas_ztrmm(CblasColMajor,
                    (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
                    (CBLAS_TRANSPOSE)trans, (CBLAS_DIAG)diag,
                    M, N,
                    CBLAS_SADDR(alpha), ddescA.mat, LDA,
                                        ddescC.mat, LDC );
    }

    Clapacknorm = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescC );

    dplasma_zgeadd( dague, PlasmaNoTrans,
                    -1.0, (tiled_matrix_desc_t*)ddescCfinal,
                     1.0, (tiled_matrix_desc_t*)&ddescC );

    Rnorm = dplasma_zlange( dague, PlasmaMaxNorm, (tiled_matrix_desc_t*)&ddescC );

    result = Rnorm / (Clapacknorm * max(M,N) * eps);

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||C||_inf = %e\n"
                   "  ||lapack(a*A*C)||_inf = %e, ||dplasma(a*A*C)||_inf = %e, ||R||_m = %e, res = %e\n",
                   Anorm, Cinitnorm, Clapacknorm, Cdplasmanorm, Rnorm, result);
        }

        if (  isinf(Clapacknorm) || isinf(Cdplasmanorm) ||
              isnan(result) || isinf(result) || (result > 10.0) ) {
            info_solution = 1;
        }
        else {
            info_solution = 0;
        }
    }

#if defined(DAGUE_HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    dague_data_free(ddescA.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    dague_data_free(ddescC.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC);

    return info_solution;
}
