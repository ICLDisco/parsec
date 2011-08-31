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

static int check_solution(PLASMA_enum transA, PLASMA_enum transB,
                          Dague_Complex64_t alpha, two_dim_block_cyclic_t *ddescA, two_dim_block_cyclic_t *ddescB, 
                          Dague_Complex64_t beta, two_dim_block_cyclic_t *ddescC, two_dim_block_cyclic_t *ddescCfinal);

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    PASTE_CODE_FLOPS(FLOPS_ZGEMM, ((DagDouble_t)M,(DagDouble_t)N,(DagDouble_t)K));

    int tA = PlasmaNoTrans;
    int tB = PlasmaNoTrans;
    Dague_Complex64_t alpha =  0.51;
    Dague_Complex64_t beta  = -0.42;

    LDA = max(LDA, max(M, K));
    LDB = max(LDB, max(K, N));
    LDC = max(LDC, M);

    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1, 
        two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, 
                               nodes, cores, rank, MB, NB, LDC, N, 0, 0, 
                               M, N, SMB, SNB, P));

    /* initializing matrix structure */
    if(!check) 
    {
        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
            two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                                   nodes, cores, rank, MB, NB, LDA, LDA, 0, 0, 
                                   M, K, SMB, SNB, P));
        PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1, 
            two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, 
                                   nodes, cores, rank, MB, NB, LDB, LDB, 0, 0, 
                                   K, N, SMB, SNB, P));

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, 3872);
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 4674);
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescC, 2873);
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
        dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
        dague_data_free(ddescB.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescB);
    } else if ( iparam[IPARAM_NNODES] > 1 ) {
        fprintf(stderr, "Checking doesn't work in distributed\n");
        info_solution = 1;
    } else {
        int Am, An, Bm, Bn;
        PASTE_CODE_ALLOCATE_MATRIX(ddescC2, check, 
            two_dim_block_cyclic, (&ddescC2, matrix_ComplexDouble, 
                                   nodes, cores, rank, MB, NB, LDC, N, 0, 0, 
                                   M, N, SMB, SNB, P));
                
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescC2, 2873);

/* Iterate on the transpose forms. TODO: LDB is set incorrecly for T and H */
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
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
                    two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                                           nodes, cores, rank, MB, NB, LDA, LDA, 0, 0, 
                                           Am, An, SMB, SNB, P));
                PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1, 
                    two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, 
                                           nodes, cores, rank, MB, NB, LDB, LDB, 0, 0, 
                                           Bm, Bn, SMB, SNB, P));

                dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA,  3872);
                dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB,  4674);

                if ( rank == 0 ) {
                    printf("***************************************************\n");
                    printf(" ----- TESTING DGEMM (%s, %s) -------- \n",
                           transstr[tA], transstr[tB]);
                }
                
                /* matrix generation */
                if(loud) printf("Generate matrices ... ");
                dplasma_zlacpy( dague, PlasmaUpperLower,
                                (tiled_matrix_desc_t *)&ddescC2, (tiled_matrix_desc_t *)&ddescC );
                if(loud) printf("Done\n");

                /* Create GEMM DAGuE */
                if(loud) printf("Compute ... ... ");
                dplasma_dgemm(dague, trans[tA], trans[tB],
                              (Dague_Complex64_t)alpha,
                              (tiled_matrix_desc_t *)&ddescA, 
                              (tiled_matrix_desc_t *)&ddescB, 
                              (Dague_Complex64_t)beta, 
                              (tiled_matrix_desc_t *)&ddescC);
                if(loud) printf("Done\n");
                
                /* Check the solution */
                info_solution = check_solution( trans[tA], trans[tB], 
                                                alpha, &ddescA,  &ddescB, 
                                                beta,  &ddescC2, &ddescC);
                if ( rank == 0 ) {
                    if (info_solution == 0) {
                        printf(" ---- TESTING DGEMM (%s, %s) ...... PASSED !\n",
                               transstr[tA], transstr[tB]);
                    }
                    else {
                        printf(" ---- TESTING DGEMM (%s, %s) ... FAILED !\n",
                               transstr[tA], transstr[tB]);
                    }
                    printf("***************************************************\n");
                }

                dague_data_free(ddescA.mat);
                dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
                dague_data_free(ddescB.mat);
                dague_ddesc_destroy((dague_ddesc_t*)&ddescB);
            }
        }

        dague_data_free(ddescC2.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescC2);
    }

    dague_data_free(ddescC.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescC);

    cleanup_dague(dague, iparam);

    return info_solution;
}


/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution(PLASMA_enum transA, PLASMA_enum transB,
                          Dague_Complex64_t alpha, two_dim_block_cyclic_t *ddescA, two_dim_block_cyclic_t *ddescB, 
                          Dague_Complex64_t beta, two_dim_block_cyclic_t *ddescC, two_dim_block_cyclic_t *ddescCfinal )
{
    int info_solution;
    double Anorm, Bnorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm;
    double eps, result;
    double *work;
    int Am, An, Bm, Bn;
    Dague_Complex64_t mzone = (Dague_Complex64_t)-1.0;
    Dague_Complex64_t *A, *B, *Cinit, *Cfinal;

    int M   = ddescC->super.m;
    int N   = ddescC->super.n;
    int K   = ( transA == PlasmaNoTrans ) ? ddescA->super.n : ddescA->super.m ;
    int LDA = ddescA->super.lm;
    int LDB = ddescB->super.lm;
    int LDC = ddescC->super.lm;

    eps = LAPACKE_dlamch_work('e');

    if (transA == PlasmaNoTrans) {
        Am = M; An = K;
    } else {
        Am = K; An = M;
    }
    if (transB == PlasmaNoTrans) {
        Bm = K; Bn = N;
    } else {
        Bm = N; Bn = K;
    }

    work  = (double *)malloc(max(K,max(M, N))* sizeof(double));
    A     = (Dague_Complex64_t *)malloc((ddescA->super.lm)*(ddescA->super.n)*sizeof(Dague_Complex64_t));
    B     = (Dague_Complex64_t *)malloc((ddescB->super.lm)*(ddescB->super.n)*sizeof(Dague_Complex64_t));
    Cinit = (Dague_Complex64_t *)malloc((ddescC->super.lm)*(ddescC->super.n)*sizeof(Dague_Complex64_t));
    Cfinal= (Dague_Complex64_t *)malloc((ddescC->super.lm)*(ddescC->super.n)*sizeof(Dague_Complex64_t));

    twoDBC_ztolapack( ddescA,      A,      LDA );
    twoDBC_ztolapack( ddescB,      B,      LDB );
    twoDBC_ztolapack( ddescC,      Cinit,  LDC );
    twoDBC_ztolapack( ddescCfinal, Cfinal, LDC );

    Anorm        = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', Am, An, A,      LDA, work);
    Bnorm        = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', Bm, Bn, B,      LDB, work);
    Cinitnorm    = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', M,  N,  Cinit,  LDC, work);
    Cdplasmanorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', M,  N,  Cfinal, LDC, work);

    cblas_zgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
                M, N, K, CBLAS_SADDR(alpha), A, LDA, B, LDB,
                CBLAS_SADDR(beta), Cinit, LDC);

    Clapacknorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', M, N, Cinit, LDC, work);

    cblas_zaxpy(LDC * N, CBLAS_SADDR(mzone), Cinit, 1, Cfinal, 1);
    Rnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', M, N, Cfinal, LDC, work);

    if (getenv("DPLASMA_TESTING_VERBOSE"))
        printf("Rnorm %e, Anorm %e, Bnorm %e, Cinit %e, Cdplasmanorm %e, Clapacknorm %e\n",
               Rnorm, Anorm, Bnorm, Cinitnorm, Cdplasmanorm, Clapacknorm);
    
    result = Rnorm / ((Anorm + Bnorm + Cinitnorm) * max(M,N) * eps);
    if (  isinf(Clapacknorm) || isinf(Cdplasmanorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        info_solution = 1;
    }
    else{
        info_solution = 0;
    }

    free(work);
    free(A);
    free(B);
    free(Cinit);
    free(Cfinal);

    return info_solution;
}
