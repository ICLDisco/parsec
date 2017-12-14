/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */

#include "common.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h"

static int check_solution( parsec_context_t *parsec, int loud,
                           PLASMA_enum uplo, PLASMA_enum trans,
                           double alpha, int Am, int An, int Aseed,
                           double beta,  int M,  int N,  int Cseed,
                           sym_two_dim_block_cyclic_t *dcCfinal );

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    int Aseed = 3872;
    int Cseed = 2873;
    double alpha = 3.5;
    double beta  = -2.8;

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
    iparam[IPARAM_NGPUS] = 0;

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    M = N;
    LDC = max(LDC, N);

    if(!check)
    {
        PLASMA_enum uplo  = PlasmaLower;
        PLASMA_enum trans = PlasmaNoTrans;
        int Am = ( trans == PlasmaNoTrans ? N : K );
        int An = ( trans == PlasmaNoTrans ? K : N );
        LDA = max(LDA, Am);

        PASTE_CODE_FLOPS(FLOPS_ZHERK, ((DagDouble_t)K, (DagDouble_t)N));

        PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
            two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDA, An, 0, 0,
                                   Am, An, SMB, SNB, P));

        PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
            sym_two_dim_block_cyclic, (&dcC, matrix_ComplexDouble,
                                       nodes, rank, MB, NB, LDC, N, 0, 0,
                                       N, N, P, uplo));

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA,  Aseed);
        dplasma_zplghe( parsec, 0., uplo, (parsec_tiled_matrix_dc_t *)&dcC, Cseed);
        if(loud > 2) printf("Done\n");

        /* Create PaRSEC */
        PASTE_CODE_ENQUEUE_KERNEL(parsec, zherk,
                                  (uplo, trans,
                                   alpha, (parsec_tiled_matrix_dc_t *)&dcA,
                                   beta,  (parsec_tiled_matrix_dc_t *)&dcC));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(parsec, zherk);

        dplasma_zherk_Destruct( PARSEC_zherk );

        parsec_data_free(dcA.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
        parsec_data_free(dcC.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC);
    }
    else
    {
        int u, t;
        int info_solution;

        PASTE_CODE_ALLOCATE_MATRIX(dcC2, check,
            two_dim_block_cyclic, (&dcC2, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDC, N, 0, 0,
                                   N, N, SMB, SNB, P));

        if (loud > 2) printf("Generate matrices ... ");
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC2, Cseed);
        if (loud > 2) printf("Done\n");

        for (u=0; u<2; u++) {
            PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
                sym_two_dim_block_cyclic, (&dcC, matrix_ComplexDouble,
                                           nodes, rank, MB, NB, LDC, N, 0, 0,
                                           N, N, P, uplo[u]));

            for (t=0; t<2; t++) {
#if defined(PRECISION_z) || defined(PRECISION_c)
                if (t==1) t++;
#endif
                /* initializing matrix structure */
                int Am = ( trans[t] == PlasmaNoTrans ? N : K );
                int An = ( trans[t] == PlasmaNoTrans ? K : N );
                LDA = max(LDA, Am);

                PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
                    two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                           nodes, rank, MB, NB, LDA, An, 0, 0,
                                           Am, An, SMB, SNB, P));

                if (loud > 2) printf("Generate matrices ... ");
                dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);
                dplasma_zlacpy( parsec, uplo[u],
                                (parsec_tiled_matrix_dc_t *)&dcC2, (parsec_tiled_matrix_dc_t *)&dcC );
                if (loud > 2) printf("Done\n");

                /* Compute */
                if (loud > 2) printf("Compute ... ... ");
                dplasma_zherk(parsec, uplo[u], trans[t],
                              alpha, (parsec_tiled_matrix_dc_t *)&dcA,
                              beta,  (parsec_tiled_matrix_dc_t *)&dcC);
                if (loud > 2) printf("Done\n");

                /* Check the solution */
                info_solution = check_solution(parsec, rank == 0 ? loud : 0,
                                               uplo[u], trans[t],
                                               alpha, Am, An, Aseed,
                                               beta,  N,  N,  Cseed,
                                               &dcC);

                if ( rank == 0 ) {
                    if (info_solution == 0) {
                        printf(" ---- TESTING ZHERK (%s, %s) ...... PASSED !\n",
                               uplostr[u], transstr[t]);
                    }
                    else {
                        printf(" ---- TESTING ZHERK (%s, %s) ... FAILED !\n",
                               uplostr[u], transstr[t]);
                        ret |= 1;
                    }
                    printf("***************************************************\n");
                }

                parsec_data_free(dcA.mat);
                parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
            }
            parsec_data_free(dcC.mat);
            parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC);
        }

        parsec_data_free(dcC2.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC2);
    }

    cleanup_parsec(parsec, iparam);

    return ret;
}


/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( parsec_context_t *parsec, int loud,
                           PLASMA_enum uplo, PLASMA_enum trans,
                           double alpha, int Am, int An, int Aseed,
                           double beta,  int M,  int N,  int Cseed,
                           sym_two_dim_block_cyclic_t *dcCfinal )
{
    int info_solution = 1;
    double Anorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm;
    double eps, result;
    int MB = dcCfinal->super.mb;
    int NB = dcCfinal->super.nb;
    int LDA = Am;
    int LDC = M;
    int rank  = dcCfinal->super.super.myrank;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDA, An, 0, 0,
                               Am, An, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        two_dim_block_cyclic, (&dcC, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1));

    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC, Cseed );

    Anorm        = dplasma_zlange( parsec, PlasmaInfNorm, (parsec_tiled_matrix_dc_t*)&dcA );
    Cinitnorm    = dplasma_zlanhe( parsec, PlasmaInfNorm, uplo, (parsec_tiled_matrix_dc_t*)&dcC );
    Cdplasmanorm = dplasma_zlanhe( parsec, PlasmaInfNorm, uplo, (parsec_tiled_matrix_dc_t*)dcCfinal );

    if ( rank == 0 ) {
        cblas_zherk(CblasColMajor,
                    (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
                    N, (trans == PlasmaNoTrans) ? An : Am,
                    alpha, dcA.mat, LDA,
                    beta,  dcC.mat, LDC);
    }

    Clapacknorm = dplasma_zlanhe( parsec, PlasmaInfNorm, uplo, (parsec_tiled_matrix_dc_t*)&dcC );

    dplasma_ztradd( parsec, uplo, PlasmaNoTrans,
                    -1.0, (parsec_tiled_matrix_dc_t*)dcCfinal,
                     1.0, (parsec_tiled_matrix_dc_t*)&dcC );

    Rnorm = dplasma_zlanhe( parsec, PlasmaMaxNorm, uplo, (parsec_tiled_matrix_dc_t*)&dcC );

    result = Rnorm / (Clapacknorm * max(M,N) * eps);

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||C||_inf = %e\n"
                   "  ||lapack(a*A*At+b*C)||_inf = %e, ||dplasma(a*A*At+b*C)||_inf = %e, ||R||_m = %e, res = %e\n",
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

#if defined(PARSEC_HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC);

    return info_solution;
}
