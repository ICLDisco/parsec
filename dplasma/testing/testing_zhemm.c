/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"

static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum side, PLASMA_enum uplo,
                           dague_complex64_t alpha, int Am, int An, int Aseed,
                           dague_complex64_t beta,  int M,  int N,  int Bseed, int Cseed,
                           two_dim_block_cyclic_t *ddescCfinal );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    int Aseed = 3872;
    int Bseed = 4674;
    int Cseed = 2873;
    dague_complex64_t alpha =  3.5 - I * 4.2;
    dague_complex64_t beta  = -2.8 + I * 0.7;

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
    iparam[IPARAM_NGPUS] = 0;

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    LDB = max(LDB, M);
    LDC = max(LDC, M);

    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescC2, check,
        two_dim_block_cyclic, (&ddescC2, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));

    if (loud > 2) printf("Generate matrices ... ");
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB,  Bseed);
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC,  Cseed);
    if (check)
        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC2, Cseed);
    if (loud > 2) printf("Done\n");

    if(!check)
    {
        PLASMA_enum side  = PlasmaLeft;
        PLASMA_enum uplo  = PlasmaLower;
        int Am = ( side == PlasmaLeft ? M : N );
        LDA = max(LDA, Am);

        PASTE_CODE_FLOPS(FLOPS_ZHEMM, (side, (DagDouble_t)M, (DagDouble_t)N));

        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
            sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                       nodes, cores, rank, MB, NB, LDA, Am, 0, 0,
                                       Am, Am, P, uplo));

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplghe( dague, 0., uplo, (tiled_matrix_desc_t *)&ddescA, Aseed);
        if(loud > 2) printf("Done\n");

        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, zhemm,
                                  (side, uplo,
                                   alpha, (tiled_matrix_desc_t *)&ddescA,
                                          (tiled_matrix_desc_t *)&ddescB,
                                   beta,  (tiled_matrix_desc_t *)&ddescC));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, zhemm);

        dplasma_zhemm_Destruct( DAGUE_zhemm );

        dague_data_free(ddescA.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    }
    else
    {
        int s, u;
        int info_solution;

        for (s=0; s<2; s++) {
            /* initializing matrix structure */
            int Am = ( side[s] == PlasmaLeft ? M : N );
            LDA = max(LDA, Am);

            for (u=0; u<2; u++) {

                PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                    sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                               nodes, cores, rank, MB, NB, LDA, Am, 0, 0,
                                               Am, Am, P, uplo[u]));

                if (loud > 2) printf("Generate matrices ... ");
                dplasma_zplghe( dague, 0., uplo[u], (tiled_matrix_desc_t *)&ddescA, Aseed);
                dplasma_zlacpy( dague, PlasmaUpperLower,
                                (tiled_matrix_desc_t *)&ddescC2, (tiled_matrix_desc_t *)&ddescC );
                if (loud > 2) printf("Done\n");

                /* Compute */
                if (loud > 2) printf("Compute ... ... ");
                dplasma_zhemm(dague, side[s], uplo[u],
                              alpha, (tiled_matrix_desc_t *)&ddescA,
                                     (tiled_matrix_desc_t *)&ddescB,
                              beta,  (tiled_matrix_desc_t *)&ddescC);
                if (loud > 2) printf("Done\n");

                /* Check the solution */
                info_solution = check_solution(dague, rank == 0 ? loud : 0,
                                               side[s], uplo[u],
                                               alpha, Am, Am, Aseed,
                                               beta,  M,  N,  Bseed, Cseed,
                                               &ddescC);

                if ( rank == 0 ) {
                    if (info_solution == 0) {
                        printf(" ---- TESTING ZHEMM (%s, %s) ...... PASSED !\n",
                               sidestr[s], uplostr[u]);
                    }
                    else {
                        printf(" ---- TESTING ZHEMM (%s, %s) ... FAILED !\n",
                               sidestr[s], uplostr[u]);
                        ret |= 1;
                    }
                    printf("***************************************************\n");
                }

                dague_data_free(ddescA.mat);
                dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
            }
        }

        dague_data_free(ddescC2.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescC2);
    }

    dague_data_free(ddescB.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescB);
    dague_data_free(ddescC.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescC);

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
                           PLASMA_enum side, PLASMA_enum uplo,
                           dague_complex64_t alpha, int Am, int An, int Aseed,
                           dague_complex64_t beta,  int M,  int N,  int Bseed, int Cseed,
                           two_dim_block_cyclic_t *ddescCfinal )
{
    int info_solution;
    double Anorm, Bnorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm;
    double eps, result;
    int MB = ddescCfinal->super.mb;
    int NB = ddescCfinal->super.nb;
    int LDA = (Am%MB==0) ? Am : (Am/MB+1) * MB;
    int LDC = ( M%MB==0) ? M  : ( M/MB+1) * MB;
    int LDB = LDC;
    int cores = ddescCfinal->super.super.cores;
    int rank  = ddescCfinal->super.super.myrank;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Lapack,
                               1, cores, rank, MB, NB, LDA, An, 0, 0,
                               Am, An, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Lapack,
                               1, cores, rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Lapack,
                               1, cores, rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1));

    dplasma_zplghe( dague, 0., PlasmaUpperLower, (tiled_matrix_desc_t *)&ddescA, Aseed);
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, Bseed );
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC, Cseed );

    Anorm        = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescA );
    Bnorm        = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescB );
    Cinitnorm    = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescC );
    Cdplasmanorm = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)ddescCfinal );

    if ( rank == 0 ) {
        cblas_zhemm(CblasColMajor,
                    (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
                    M, N,
                    CBLAS_SADDR(alpha), ddescA.mat, LDA,
                                        ddescB.mat, LDB,
                    CBLAS_SADDR(beta),  ddescC.mat, LDC);
    }

    Clapacknorm = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescC );

    dplasma_zgeadd( dague, PlasmaUpperLower, -1.0, (tiled_matrix_desc_t*)ddescCfinal,
                                                   (tiled_matrix_desc_t*)&ddescC );

    Rnorm = dplasma_zlange( dague, PlasmaMaxNorm, (tiled_matrix_desc_t*)&ddescC );

    result = Rnorm / (Clapacknorm * max(M,N) * eps);

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||B||_inf = %e, ||C||_inf = %e\n"
                   "  ||lapack(a*A*B+b*C)||_inf = %e, ||dplasma(a*A*B+b*C)||_inf = %e, ||R||_m = %e, res = %e\n",
                   Anorm, Bnorm, Cinitnorm, Clapacknorm, Cdplasmanorm, Rnorm, result);
        }

        if (  isinf(Clapacknorm) || isinf(Cdplasmanorm) ||
              isnan(result) || isinf(result) || (result > 10.0) ) {
            info_solution = 1;
        }
        else {
            info_solution = 0;
        }
    }

#if defined(HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescB.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescB);
    dague_data_free(ddescC.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescC);

    return info_solution;
}
