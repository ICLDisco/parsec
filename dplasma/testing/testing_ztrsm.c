/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <math.h>
#include <cblas.h>
#include <plasma.h>
#include <lapacke.h>
#include <core_blas.h>

#include "dague.h"
#include "scheduling.h"
#include "profiling.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"
#include "dplasma.h"

#include "common.h"
#include "common_timing.h"

#define _FMULS(side, M, N) ( side == PlasmaLeft ? ( 0.5 * (DagDouble_t)(N) * (DagDouble_t)(M) * (DagDouble_t)((M)+1) ) : ( 0.5 * (DagDouble_t)(M) * (DagDouble_t)(N) * (DagDouble_t)((N)+1) ) )
#define _FADDS(side, M, N) ( side == PlasmaLeft ? ( 0.5 * (DagDouble_t)(N) * (DagDouble_t)(M) * (DagDouble_t)((M)-1) ) : ( 0.5 * (DagDouble_t)(M) * (DagDouble_t)(N) * (DagDouble_t)((N)-1) ) )

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

    A = (Dague_Complex64_t *)malloc((ddescA->super.mt)*(ddescA->super.nt)*(ddescA->super.bsiz)*sizeof(Dague_Complex64_t));
    B = (Dague_Complex64_t *)malloc((ddescB->super.mt)*(ddescB->super.nt)*(ddescB->super.bsiz)*sizeof(Dague_Complex64_t));
    C = (Dague_Complex64_t *)malloc((ddescC->super.mt)*(ddescC->super.nt)*(ddescC->super.bsiz)*sizeof(Dague_Complex64_t));

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

int main(int argc, char ** argv)
{
    int iparam[IPARAM_SIZEOF];
    dague_context_t* dague;
    
    /* Set defaults for non argv iparam */
    iparam_default_solve(iparam);
    iparam[IPARAM_NGPUS] = -1;
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);

    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int cores = iparam[IPARAM_NCORES];
    int P     = iparam[IPARAM_P];
    int M     = iparam[IPARAM_M];
    int N     = iparam[IPARAM_N];
    int NRHS  = iparam[IPARAM_K];
    int LDA   = iparam[IPARAM_LDA];
    int LDB   = iparam[IPARAM_LDB];
    int MB    = iparam[IPARAM_MB];
    int NB    = iparam[IPARAM_NB];
    int SMB   = iparam[IPARAM_SMB];
    int SNB   = iparam[IPARAM_SNB];
    int loud  = iparam[IPARAM_VERBOSE];
    int mt    = (M%MB==0) ? (M/MB) : (M/MB+1);
    int nt    = (N%NB==0) ? (N/NB) : (N/NB+1);

    DagDouble_t flops, gflops;
    int s = PlasmaLeft;
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
    flops = 2.*_FADDS(s, M, NRHS) + 6.*_FMULS(s, M, NRHS);
#else
    flops = _FADDS(s, M, NRHS) + _FMULS(s, M, NRHS);
#endif

    /* initializing matrix structure */
    if(loud) printf("Generate matrices ... ");
    two_dim_block_cyclic_t ddescA;
    two_dim_block_cyclic_init(&ddescA, matrix_ComplexDouble, nodes, cores, rank, MB, NB, M, N,    0, 0, LDA, N,    SMB, SNB, P);
    ddescA.mat = dague_data_allocate((size_t)ddescA.super.nb_local_tiles * (size_t)ddescA.super.bsiz * (size_t)ddescA.super.mtype);
    generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);
    
    two_dim_block_cyclic_t ddescB;
    two_dim_block_cyclic_init(&ddescB, matrix_ComplexDouble, nodes, cores, rank, MB, NB, M, NRHS, 0, 0, LDB, NRHS, SMB, SNB, P);
    ddescB.mat = dague_data_allocate((size_t)ddescB.super.nb_local_tiles * (size_t)ddescB.super.bsiz * (size_t)ddescB.super.mtype);
    generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescB, 200);
    
    two_dim_block_cyclic_t work;
    two_dim_block_cyclic_init(&work, matrix_Integer, nodes, cores, rank, 1, 1, mt, nt, 0, 0, mt, nt, 1, 1, P);
    work.mat = dague_data_allocate((size_t)work.super.nb_local_tiles * (size_t)work.super.bsiz * (size_t)work.super.mtype);
    if(loud) printf("Done\n");

    if(iparam[IPARAM_CHECK] == 0) 
    {
        /* Create TRSM DAGuE */
        if(loud) printf("Generate ZTRSM DAG ... ");
        TIME_START();
        dague_object_t* dague_trsm = 
            dplasma_ztrsm_New(s, PlasmaLower, PlasmaNoTrans, PlasmaUnit,
                (Dague_Complex64_t)1.0, 
                (tiled_matrix_desc_t *)&ddescA, 
                (tiled_matrix_desc_t *)&ddescB); /*, 
                (tiled_matrix_desc_t *)&work);*/
        dague_enqueue(dague, dague_trsm);
        if(loud) printf("Done\n");
        if(loud) SYNC_TIME_PRINT(rank, ("DAG creation: %u total tasks enqueued\n", dague->taskstodo));

        /* lets rock! */
        SYNC_TIME_START();
        TIME_START();
        dague_progress(dague);
        if(loud) TIME_PRINT(rank, ("Dague proc %d:\tcomputed %u tasks,\t%f task/s\n",
                    rank, dague_trsm->nb_local_tasks,
                    dague_trsm->nb_local_tasks/time_elapsed));
        SYNC_TIME_PRINT(rank, ("Dague progress:\t%d %d %f gflops\n", N, NB,
                         gflops = (flops/1e9)/(sync_time_elapsed)));
    }
    else {
        int s, u, t, d;
        int info_solution;
        Dague_Complex64_t alpha = 1.0;
        two_dim_block_cyclic_t ddescC;

        two_dim_block_cyclic_init(&ddescC, matrix_ComplexDouble, nodes, cores, rank, MB, NB, M, NRHS, 0, 0, LDB, NRHS, SMB, SNB, P);
        ddescC.mat = dague_data_allocate((size_t)ddescC.super.nb_local_tiles * (size_t)ddescC.super.bsiz * (size_t)ddescC.super.mtype);

        for (s=0; s<2; s++) {
            for (u=0; u<2; u++) {
#if defined(PRECISIONS_z) || defined(PRECISIONS_c)
                for (t=0; t<3; t++) {
#else
                for (t=0; t<2; t++) {
#endif
                    for (d=0; d<2; d++) {

                        printf("***************************************************\n");
                        printf(" ----- TESTING DTRSM (%s, %s, %s, %s) -------- \n",
                                   sidestr[s], uplostr[u], transstr[t], diagstr[d]);

                        /* matrix generation */
                        printf("Generate matrices ... ");
			generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 400);
                        generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescB, 200);
                        generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescC, 200);
                        printf("Done\n");

                        /* Create TRSM DAGuE */
                        printf("Compute ... ... ");
                        dplasma_dtrsm(dague, side[s], uplo[u], trans[t], diag[d], (Dague_Complex64_t)alpha,
                                      (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescC);
                        printf("Done\n");

                        /* Check the solution */
                        info_solution = check_solution(side[s], uplo[u], trans[t], diag[d],
                                                       alpha, &ddescA, &ddescB, &ddescC);

                        if (info_solution == 0) {
                            printf(" ---- TESTING DTRSM (%s, %s, %s, %s) ...... PASSED !\n",
                                   sidestr[s], uplostr[u], transstr[t], diagstr[d]);
                        }
                        else {
                            printf(" ---- TESTING DTRSM (%s, %s, %s, %s) ... FAILED !\n",
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
    }

    dague_data_free(work.mat);
    dague_data_free(ddescA.mat);
    dague_data_free(ddescB.mat);

    cleanup_dague(dague);
    return 0;
}
