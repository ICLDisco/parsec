/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2015-2016 Inria, CNRS (LaBRI - UMR 5800), University of
 *                         Bordeaux and Bordeaux INP. All rights reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

static int check_solution(int N, const double *E1, const double *E2);

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    dplasma_qrtree_t qrtre0, qrtree, lqtree;
    tiled_matrix_desc_t *subA = NULL;
    double *s0 = NULL;
    double *s1;
    int minMN;
    int info_solution, rbidiag = 0;
    double time_ge2gb, time_gb2bd, time_solve = -1.;

    /* Ensure BLAS are sequential and set thread affinity for the master */
/* #if defined(__ICC) || defined(__INTEL_COMPILER) */
/*     kmp_set_defaults("KMP_AFFINITY=disabled"); */
/*     mkl_set_num_threads( 1 ); */
/* #endif */

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 32, 200, 200);
    iparam[IPARAM_SMB] = 1;
    iparam[IPARAM_SNB] = 1;
    iparam[IPARAM_HMB] = 2;
    iparam[IPARAM_HNB] = 2;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);

    /* Make sure SMB and SNB are set to 1, since it conflicts with HQR */
    iparam[IPARAM_SMB] = 1;
    iparam[IPARAM_SNB] = 1;

    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGEBRD, ((DagDouble_t)M, (DagDouble_t)N));

    rbidiag = iparam[IPARAM_QR_TSRR];
    LDA = max(M, LDA);

    if ( M < N ) {
        fprintf(stderr, "This testing can only perform SVD on matrices with M >= N\n");
        return EXIT_FAILURE;
    }
    minMN = dplasma_imin(M, N);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTS0, rbidiag,
        two_dim_block_cyclic, (&ddescTS0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTT0, rbidiag,
        two_dim_block_cyclic, (&ddescTT0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTS, 1,
        two_dim_block_cyclic, (&ddescTS, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTT, 1,
        two_dim_block_cyclic, (&ddescTT, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescBand, 1,
        two_dim_block_cyclic, (&ddescBand, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB+1, NB, MB+1, minMN, 0, 0,
                               MB+1, minMN, 1, 1, 1));

    /* Initialize the matrix */
    if(loud > 3) printf("+++ Generate matrices ... ");

    /* Generate the matrix on rank 0 */
    if ( check ) {

        /* Generate the singular values vector as in latms routines for check purpose */
        if (rank == 0)
        {
            double tmp = 1. / (double)N;
            double alp = ( 1. - tmp ) / ((double)( N - 1 ));
            int i;
            s0 = (double *) malloc(minMN * sizeof(double));

            s0[0] = 1.;
            for(i=1; i < minMN; i++){
                s0[i] = (double)(N-i-1) * alp + tmp;
            }
        }

        dplasma_zlatms( dague, PlasmaGeneral, (double)N, (tiled_matrix_desc_t *)&ddescA, 3872);
    }
    else {
        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, 3872);
    }

    if ( rbidiag ) {
        subA = tiled_matrix_submatrix( (tiled_matrix_desc_t *)&ddescA,
                                       0, 0, ddescA.super.n, ddescA.super.n );

        dplasma_hqr_init( &qrtre0,
                          PlasmaNoTrans, (tiled_matrix_desc_t *)&ddescA,
                          iparam[IPARAM_LOWLVL_TREE], iparam[IPARAM_HIGHLVL_TREE],
                          iparam[IPARAM_QR_TS_SZE],   P,
                          iparam[IPARAM_QR_DOMINO],   0 );

        dplasma_svd_init( &qrtree,
                          PlasmaNoTrans, subA,
                          iparam[IPARAM_HIGHLVL_TREE], P,
                          iparam[IPARAM_NCORES], iparam[IPARAM_HMB] );

        dplasma_svd_init( &lqtree,
                          PlasmaTrans, subA,
                          iparam[IPARAM_HIGHLVL_TREE], Q,
                          iparam[IPARAM_NCORES], iparam[IPARAM_HMB] );
    }
    else {
        dplasma_svd_init( &qrtree,
                          PlasmaNoTrans, (tiled_matrix_desc_t *)&ddescA,
                          iparam[IPARAM_HIGHLVL_TREE], P,
                          iparam[IPARAM_NCORES], iparam[IPARAM_HMB]);

        dplasma_svd_init( &lqtree,
                          PlasmaTrans, (tiled_matrix_desc_t *)&ddescA,
                          iparam[IPARAM_HIGHLVL_TREE], Q,
                          iparam[IPARAM_NCORES], iparam[IPARAM_HMB]);
    }

    /* Create DAGuE */
    PASTE_CODE_ENQUEUE_KERNEL(dague, zgebrd_ge2gb,
                              (rbidiag ? &qrtre0 : &qrtree,
                               &qrtree, &lqtree,
                               (tiled_matrix_desc_t*)&ddescA,
                               rbidiag ? (tiled_matrix_desc_t*)&ddescTS0 : (tiled_matrix_desc_t*)&ddescTS,
                               rbidiag ? (tiled_matrix_desc_t*)&ddescTT0 : (tiled_matrix_desc_t*)&ddescTT,
                               (tiled_matrix_desc_t*)&ddescTS,
                               (tiled_matrix_desc_t*)&ddescTT,
                               (tiled_matrix_desc_t*)&ddescBand));

    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    dague_context_wait(dague);
    SYNC_TIME_STOP();
    time_ge2gb = sync_time_elapsed;

    if( rank == 0 ) {
        double *e;

        s1 = (double*)malloc( minMN * sizeof(double));
        e  = (double*)malloc( minMN * sizeof(double));

/* #if defined(__ICC) || defined(__INTEL_COMPILER) */
/*         mkl_set_num_threads( iparam[IPARAM_NCORES] ); */
/* #endif */
        /* Reduce the band */
        TIME_START();
        info_solution = LAPACKE_zgbbrd( LAPACK_COL_MAJOR,
                                        'N',
                                        M, N,
                                        0, 0, NB,
                                        ddescBand.mat, MB+1,
                                        s1, e,
                                        NULL, 1,
                                        NULL, 1,
                                        NULL, 1 );
        TIME_STOP();
        time_gb2bd = time_elapsed;

        /* Solve the bidiagonal SVD problem */
        if (info_solution == 0){
            TIME_START();
            info_solution = LAPACKE_zbdsqr( LAPACK_COL_MAJOR, 'U',
                                            minMN, 0, 0, 0,
                                            s1, e,
                                            NULL, 1, NULL, 1, NULL, 1 );
            TIME_STOP();
            time_solve = time_elapsed;
        }
        free(e);

/* #if defined(__ICC) || defined(__INTEL_COMPILER) */
/*         mkl_set_num_threads( 1 ); */
/* #endif */
        fprintf(stderr, "WARNING: This code is using the non optimized Lapack zbdsqr subroutine to reduce the band to bi-diagonal form. Please replace this call by the multi-threaded PLASMA implementation in order to get performance\n");
        printf("zgeqrf GESVD computation NP= %d NC= %d P= %d IB= %d MB= %d NB= %d qr_a= %d qr_p = %d treel= %d treeh= %d domino= %d R-bidiag= %d M= %d N= %d : %e %e %e / %f gflops\n",
               iparam[IPARAM_NNODES],
               iparam[IPARAM_NCORES],
               iparam[IPARAM_P],
               iparam[IPARAM_IB],
               iparam[IPARAM_MB],
               iparam[IPARAM_NB],
               iparam[IPARAM_QR_TS_SZE],
               iparam[IPARAM_QR_HLVL_SZE],
               iparam[IPARAM_LOWLVL_TREE],
               iparam[IPARAM_HIGHLVL_TREE],
               iparam[IPARAM_QR_DOMINO],
               iparam[IPARAM_QR_TSRR],
               iparam[IPARAM_M],
               iparam[IPARAM_N],
               time_ge2gb, time_gb2bd, time_solve,
               gflops = (flops/1e9)/(time_ge2gb+time_gb2bd+time_solve));

#if defined(DAGUE_SIM)
        printf("zgeqrf GESVD simulation NP= %d NC= %d P= %d qr_a= %d qr_p = %d treel= %d treeh= %d domino= %d RR= %d MT= %d NT= %d : %d \n",
               iparam[IPARAM_NNODES],
               iparam[IPARAM_NCORES],
               iparam[IPARAM_P],
               iparam[IPARAM_QR_TS_SZE],
               iparam[IPARAM_QR_HLVL_SZE],
               iparam[IPARAM_LOWLVL_TREE],
               iparam[IPARAM_HIGHLVL_TREE],
               iparam[IPARAM_QR_DOMINO],
               iparam[IPARAM_QR_TSRR],
               MT, NT,
               dague_getsimulationdate( dague ));
#endif
    }

    dplasma_zgebrd_ge2gb_Destruct( DAGUE_zgebrd_ge2gb );

    if( check && (rank==0) ) {
        if (info_solution == 0 ) {
            info_solution = check_solution(minMN, s0, s1);
        }

        if (info_solution == 0) {
            printf("***************************************************\n"
                   " ---- TESTING ZGESVD .. M >= N ........... PASSED !\n"
                   "***************************************************\n");
        }
        else {
            printf("***************************************************\n"
                   " ---- TESTING ZGESVD .. M >= N .. FAILED !\n"
                   "***************************************************\n");
        }
        free(s1);
        free(s0);
    }

    dplasma_hqr_finalize( &qrtree );
    dplasma_hqr_finalize( &lqtree );

    if (rbidiag) {
        dplasma_hqr_finalize( &qrtre0 );
        free(subA);
        dague_data_free(ddescTS0.mat);
        dague_data_free(ddescTT0.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescTS0);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescTT0);
    }

    dague_data_free(ddescA.mat);
    dague_data_free(ddescTS.mat);
    dague_data_free(ddescTT.mat);
    dague_data_free(ddescBand.mat);

    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescTS);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescTT);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescBand);

    cleanup_dague(dague, iparam);

    return ret;
}

static double dplasma_dmax( double a, double b ) {
    if ( a < b ) {
        return b;
    } else {
        return a;
    }
}

/*------------------------------------------------------------
 *  Check the eigenvalues
 */
static int check_solution(int N, const double *E1, const double *E2)
{
    int info_solution, i;
    double resid;
    double maxtmp;
    double maxel = fabs( fabs(E1[0]) - fabs(E2[0]) );
    double maxeig = dplasma_dmax( fabs(E1[0]), fabs(E2[0]) );
    double eps = LAPACKE_dlamch_work('e');

    for (i = 1; i < N; i++){
        resid   = fabs(fabs(E1[i])-fabs(E2[i]));
        maxtmp  = dplasma_dmax(fabs(E1[i]), fabs(E2[i]));

        /* Update */
        maxeig = dplasma_dmax(maxtmp, maxeig);
        maxel  = dplasma_dmax(resid,  maxel );
    }

    maxel = maxel / (maxeig * N * eps);
    printf(" ======================================================\n");
    printf(" | S - singularcomputed | / (|S| * N * eps) : %e \n",  maxel );
    printf(" ======================================================\n");

    if ( isnan(maxel) || isinf(maxel) || (maxel > 100) ) {
        printf("-- The singular values are suspicious ! \n");
        info_solution = 1;
    }
    else{
        printf("-- The singular values are CORRECT ! \n");
        info_solution = 0;
    }
    return info_solution;
}
