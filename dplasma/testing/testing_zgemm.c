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
#if defined(DAGUE_HAVE_CUDA)
#include "dplasma/cores/cuda_zgemm.h"
#endif

static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum transA, PLASMA_enum transB,
                           dague_complex64_t alpha, int Am, int An, int Aseed,
                                                    int Bm, int Bn, int Bseed,
                           dague_complex64_t beta,  int M,  int N,  int Cseed,
                           two_dim_block_cyclic_t *ddescCfinal );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    int Aseed = 3872;
    int Bseed = 4674;
    int Cseed = 2873;
    int tA = PlasmaNoTrans;
    int tB = PlasmaNoTrans;
    dague_complex64_t alpha =  0.51;
    dague_complex64_t beta  = -0.42;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
    beta  += I * 0.21;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(DAGUE_HAVE_CUDA) && 1
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    PASTE_CODE_FLOPS(FLOPS_ZGEMM, ((DagDouble_t)M,(DagDouble_t)N,(DagDouble_t)K));

    LDA = max(LDA, max(M, K));
    LDB = max(LDB, max(K, N));
    LDC = max(LDC, M);

    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));

    /* initializing matrix structure */
    if(!check)
    {
        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
            two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDA, K, 0, 0,
                                   M, K, SMB, SNB, P));
        PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
            two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, N, 0, 0,
                                   K, N, SMB, SNB, P));

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, Aseed);
        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, Bseed);
        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC, Cseed);
        if(loud > 2) printf("Done\n");

        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, zgemm,
                                  (tA, tB, alpha,
                                   (tiled_matrix_desc_t *)&ddescA,
                                   (tiled_matrix_desc_t *)&ddescB,
                                   beta,
                                   (tiled_matrix_desc_t *)&ddescC));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, zgemm);

        dplasma_zgemm_Destruct( DAGUE_zgemm );

        dague_data_free(ddescA.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
        dague_data_free(ddescB.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);
    }
    else {
        int Am, An, Bm, Bn;
        PASTE_CODE_ALLOCATE_MATRIX(ddescC2, check,
            two_dim_block_cyclic, (&ddescC2, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDC, N, 0, 0,
                                   M, N, SMB, SNB, P));

        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC2, Cseed);

#if defined(PRECISION_z) || defined(PRECISION_c)
        for(tA=0; tA<3; tA++) {
            for(tB=0; tB<3; tB++) {
#else
        for(tA=0; tA<2; tA++) {
            for(tB=0; tB<2; tB++) {
#endif
                if ( trans[tA] == PlasmaNoTrans ) {
                    Am = M; An = K;
                } else {
                    Am = K; An = M;
                }
                if ( trans[tB] == PlasmaNoTrans ) {
                    Bm = K; Bn = N;
                } else {
                    Bm = N; Bn = K;
                }
                PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                    two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                           nodes, rank, MB, NB, LDA, LDA, 0, 0,
                                           Am, An, SMB, SNB, P));
                PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
                    two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                           nodes, rank, MB, NB, LDB, LDB, 0, 0,
                                           Bm, Bn, SMB, SNB, P));

                dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, Aseed);
                dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, Bseed);

                if ( rank == 0 ) {
                    printf("***************************************************\n");
                    printf(" ----- TESTING ZGEMM (%s, %s) -------- \n",
                           transstr[tA], transstr[tB]);
                }

                /* matrix generation */
                if(loud) printf("Generate matrices ... ");
                dplasma_zlacpy( dague, PlasmaUpperLower,
                                (tiled_matrix_desc_t *)&ddescC2, (tiled_matrix_desc_t *)&ddescC );
                if(loud) printf("Done\n");

                /* Create GEMM DAGuE */
                if(loud) printf("Compute ... ... ");
                dplasma_zgemm(dague, trans[tA], trans[tB],
                              (dague_complex64_t)alpha,
                              (tiled_matrix_desc_t *)&ddescA,
                              (tiled_matrix_desc_t *)&ddescB,
                              (dague_complex64_t)beta,
                              (tiled_matrix_desc_t *)&ddescC);
                if(loud) printf("Done\n");

                dague_data_free(ddescA.mat);
                tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
                dague_data_free(ddescB.mat);
                tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);

                /* Check the solution */
                info_solution = check_solution( dague, (rank == 0) ? loud : 0,
                                                trans[tA], trans[tB],
                                                alpha, Am, An, Aseed,
                                                       Bm, Bn, Bseed,
                                                beta,  M,  N,  Cseed,
                                                &ddescC);
                if ( rank == 0 ) {
                    if (info_solution == 0) {
                        printf(" ---- TESTING ZGEMM (%s, %s) ...... PASSED !\n",
                               transstr[tA], transstr[tB]);
                    }
                    else {
                        printf(" ---- TESTING ZGEMM (%s, %s) ... FAILED !\n",
                               transstr[tA], transstr[tB]);
                    }
                    printf("***************************************************\n");
                }
            }
        }
#if defined(_UNUSED_)
            }
        }
#endif
        dague_data_free(ddescC2.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC2);
    }

    dague_data_free(ddescC.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC);

    cleanup_dague(dague, iparam);

    return info_solution;
}

/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum transA, PLASMA_enum transB,
                           dague_complex64_t alpha, int Am, int An, int Aseed,
                                                    int Bm, int Bn, int Bseed,
                           dague_complex64_t beta,  int M,  int N,  int Cseed,
                           two_dim_block_cyclic_t *ddescCfinal )
{
    int info_solution = 1;
    double Anorm, Bnorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm;
    double eps, result;
    int K  = ( transA == PlasmaNoTrans ) ? An : Am ;
    int MB = ddescCfinal->super.mb;
    int NB = ddescCfinal->super.nb;
    int LDA = Am;
    int LDB = Bm;
    int LDC = M;
    int rank  = ddescCfinal->super.super.myrank;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDA, An, 0, 0,
                               Am, An, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDB, Bn, 0, 0,
                               Bm, Bn, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1));

    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, Aseed );
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, Bseed );
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC, Cseed );

    Anorm        = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescA );
    Bnorm        = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescB );
    Cinitnorm    = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescC );
    Cdplasmanorm = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)ddescCfinal );

    if ( rank == 0 ) {
        cblas_zgemm(CblasColMajor,
                    (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
                    M, N, K,
                    CBLAS_SADDR(alpha), ddescA.mat, LDA,
                                        ddescB.mat, LDB,
                    CBLAS_SADDR(beta),  ddescC.mat, LDC );
    }

    Clapacknorm = dplasma_zlange( dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&ddescC );

    dplasma_zgeadd( dague, PlasmaNoTrans, -1.0, (tiled_matrix_desc_t*)ddescCfinal,
                                           1.0, (tiled_matrix_desc_t*)&ddescC );

    Rnorm = dplasma_zlange( dague, PlasmaMaxNorm, (tiled_matrix_desc_t*)&ddescC);

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||B||_inf = %e, ||C||_inf = %e\n"
                   "  ||lapack(a*A*B+b*C)||_inf = %e, ||dplasma(a*A*B+b*C)||_inf = %e, ||R||_m = %e\n",
                   Anorm, Bnorm, Cinitnorm, Clapacknorm, Cdplasmanorm, Rnorm);
        }

        result = Rnorm / ((Anorm + Bnorm + Cinitnorm) * max(M,N) * eps);
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
    dague_data_free(ddescB.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);
    dague_data_free(ddescC.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC);

    return info_solution;
}
