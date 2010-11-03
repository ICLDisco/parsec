/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

static int check_solution(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                          Dague_Complex64_t alpha, two_dim_block_cyclic_t *ddescA, 
                          two_dim_block_cyclic_t *ddescB, two_dim_block_cyclic_t *ddescC );

#define FADDS(side, M, N) ( side == PlasmaLeft ? (0.5 * (N) * (M) * ((M)-1)) : (0.5 * (M) * (N) * ((N)-1)) )
#define FMULS(side, M, N) ( side == PlasmaLeft ? (0.5 * (N) * (M) * ((M)+1)) : (0.5 * (M) * (N) * ((N)+1)) )

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(DAGUE_CUDA_SUPPORT) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    int s = PlasmaLeft;
    PASTE_CODE_FLOPS_COUNT(FADDS, FMULS, (s, (DagDouble_t)M,(DagDouble_t)NRHS));
    /* initializing matrix structure */
    int Am = max(M, NRHS);
    LDA = max(LDA, Am);
    LDB = max(LDB, M);
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                               nodes, cores, rank, MB, NB, LDA, LDA, 0, 0, 
                               Am, Am, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1, 
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, 
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0, 
                               M, NRHS, SMB, SNB, P));

    if(!check) 
    {
        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);
        generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescB, 200);
        if(loud > 2) printf("Done\n");

        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, ztrsm,
                                  (s, PlasmaLower, PlasmaNoTrans, PlasmaUnit,
                                   (Dague_Complex64_t)1.0, 
                                   (tiled_matrix_desc_t *)&ddescA, 
                                   (tiled_matrix_desc_t *)&ddescB));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, ztrsm)
    }
    else
    { 
        int u, t, d;
        int info_solution;
        Dague_Complex64_t alpha = 3.5;

        PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1, 
            two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, 
                                   nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0, 
                                   M, NRHS, SMB, SNB, P));

        for (s=0; s<2; s++) {
            for (u=0; u<2; u++) {
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
                for (t=0; t<3; t++) {
#else
                for (t=0; t<2; t++) {
#endif
                    for (d=0; d<2; d++) {

                        printf("***************************************************\n");
                        printf(" ----- TESTING ZTRSM (%s, %s, %s, %s) -------- \n",
                                   sidestr[s], uplostr[u], transstr[t], diagstr[d]);

                        /* matrix generation */
                        printf("Generate matrices ... ");
                        generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 400);
                        generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescB, 200);
                        generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescC, 200);
                        printf("Done\n");

                        /* Create TRSM DAGuE */
                        printf("Compute ... ... ");
                        dplasma_ztrsm(dague, side[s], uplo[u], trans[t], diag[d], (Dague_Complex64_t)alpha,
                                      (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescC);
                        printf("Done\n");

                        /* Check the solution */
                        info_solution = check_solution(side[s], uplo[u], trans[t], diag[d],
                                                       alpha, &ddescA, &ddescB, &ddescC);
                       if (info_solution == 0) {
                            printf(" ---- TESTING ZTRSM (%s, %s, %s, %s) ...... PASSED !\n",
                                   sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                        }
                        else {
                            printf(" ---- TESTING ZTRSM (%s, %s, %s, %s) ... FAILED !\n",
                                   sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                        }
                        printf("***************************************************\n");
                    }
                }
#ifdef __UNUSED__
                }
#endif
            }
        }
        dague_data_free(ddescC.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescC);
    }

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescB.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescB);

    cleanup_dague(dague);
    return 0;
}


/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                          Dague_Complex64_t alpha, two_dim_block_cyclic_t *ddescA, two_dim_block_cyclic_t *ddescB, two_dim_block_cyclic_t *ddescC )
{
    int info_solution;
    double Anorm, Binitnorm, Bdaguenorm, Blapacknorm, Rnorm, result;
    Dague_Complex64_t *A, *B, *C;
    int M   = ddescB->super.m;
    int N   = ddescB->super.n;
    int LDA = ddescA->super.lm;
    int LDB = ddescB->super.lm;
    double eps = LAPACKE_dlamch_work('e');
    double *work = (double *)malloc(max(M, N)* sizeof(double));
    int Am;
    Dague_Complex64_t mzone = (Dague_Complex64_t)-1.0;

    if (side == PlasmaLeft) {
        Am = M;
    } else {
        Am = N;
    }

    A = (Dague_Complex64_t *)malloc((ddescA->super.lmt)*(ddescA->super.lnt)*(ddescA->super.bsiz)*sizeof(Dague_Complex64_t));
    B = (Dague_Complex64_t *)malloc((ddescB->super.lmt)*(ddescB->super.lnt)*(ddescB->super.bsiz)*sizeof(Dague_Complex64_t));
    C = (Dague_Complex64_t *)malloc((ddescC->super.lmt)*(ddescC->super.lnt)*(ddescC->super.bsiz)*sizeof(Dague_Complex64_t));

    twoDBC_to_lapack( ddescA, A, LDA );
    twoDBC_to_lapack( ddescB, B, LDB );
    twoDBC_to_lapack( ddescC, C, LDB );
    
    /* TODO: check lantr because it returns 0.0, it looks like a parameter is wrong */
    //Anorm      = LAPACKE_zlantr_work( LAPACK_COL_MAJOR, 'i', lapack_const(uplo), lapack_const(diag), Am, Am, A, LDA, work );
    Anorm      = LAPACKE_zlanhe_work( LAPACK_COL_MAJOR, 'i', lapack_const(uplo), Am, A, LDA, work );
    Binitnorm  = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'i', M,  N,  B, LDB, work );
    Bdaguenorm = LAPACKE_zlange_work( LAPACK_COL_MAJOR, 'i', M,  N,  C, LDB, work );

    cblas_ztrsm(CblasColMajor,
                (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans, (CBLAS_DIAG)diag,
                M, N, CBLAS_SADDR(alpha), A, LDA, B, LDB);

    Blapacknorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', M, N, B, LDB, work);

    cblas_zaxpy(LDB * N, CBLAS_SADDR(mzone), C, 1, B, 1);
    Rnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', M, N, B, LDB, work);

    if (getenv("DPLASMA_TESTING_VERBOSE"))
        printf("Rnorm %e, Anorm %e, Binitnorm %e, Bdaguenorm %e, Blapacknorm %e\n",
               Rnorm, Anorm, Binitnorm, Bdaguenorm, Blapacknorm);

    result = Rnorm / ((Anorm + Blapacknorm) * max(M,N) * eps);
    if (  isinf(Blapacknorm) || isinf(Bdaguenorm) || isnan(result) || isinf(result) || (result > 10.0) ) {
        printf("-- The solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        printf("-- The solution is CORRECT ! \n");
        info_solution = 0;
    }

    free(work);
    free(A);
    free(B);
    free(C);

    return info_solution;
}
