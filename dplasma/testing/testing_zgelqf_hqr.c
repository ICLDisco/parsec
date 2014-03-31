/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

static int check_orthogonality(dague_context_t *dague, int loud,
                               tiled_matrix_desc_t *Q);
static int check_factorization(dague_context_t *dague, int loud,
                               tiled_matrix_desc_t *Aorig,
                               tiled_matrix_desc_t *A,
                               tiled_matrix_desc_t *Q);
static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    dplasma_qrtree_t qrtree;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 32, 200, 200);
    iparam[IPARAM_SMB] = 1;
    iparam[IPARAM_SNB] = 1;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'n';

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);

    /* Make sure SMB and SNB are set to 1, since it conflicts with HQR */
    iparam[IPARAM_SMB] = 1;
    iparam[IPARAM_SNB] = 1;

    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGELQF, ((DagDouble_t)M, (DagDouble_t)N));

    LDA = max(M, LDA);
    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTS, 1,
        two_dim_block_cyclic, (&ddescTS, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTT, 1,
        two_dim_block_cyclic, (&ddescTT, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescQ, check,
        two_dim_block_cyclic, (&ddescQ, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    /* Check the solution */
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zpltmg( dague, matrix_init, (tiled_matrix_desc_t *)&ddescA, random_seed );
    if( check )
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescA0 );
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTS);
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTT);
    if(loud > 3) printf("Done\n");

    dplasma_hqr_init( &qrtree,
                      PlasmaConjTrans, (tiled_matrix_desc_t *)&ddescA,
                      iparam[IPARAM_LOWLVL_TREE], iparam[IPARAM_HIGHLVL_TREE],
                      iparam[IPARAM_QR_TS_SZE],   iparam[IPARAM_QR_HLVL_SZE],
                      iparam[IPARAM_QR_DOMINO],   iparam[IPARAM_QR_TSRR] );

    /* Create DAGuE */
    PASTE_CODE_ENQUEUE_KERNEL(dague, zgelqf_param,
                              (&qrtree,
                               (tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescTS,
                               (tiled_matrix_desc_t*)&ddescTT));

    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    dague_progress(dague);

    SYNC_TIME_PRINT(rank,
                    ("zgelqf HQR computation NP= %d NC= %d P= %d IB= %d MB= %d NB= %d qr_a= %d qr_p = %d treel= %d treeh= %d domino= %d RR= %d M= %d N= %d : %f gflops\n",
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
                     gflops = (flops/1e9)/(sync_time_elapsed)));
    if(loud >= 5 && rank == 0) {
        printf("<DartMeasurement name=\"performance\" type=\"numeric/double\"\n"
               "                 encoding=\"none\" compression=\"none\">\n"
               "%g\n"
               "</DartMeasurement>\n",
               gflops);
    }

#if defined(DAGUE_SIM)
    if ( rank == 0 ) {
        printf("zgelqf HQR simulation NP= %d NC= %d P= %d qr_a= %d qr_p = %d treel= %d treeh= %d domino= %d RR= %d MT= %d NT= %d : %d \n",
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
    }
#endif

    dplasma_zgelqf_param_Destruct( DAGUE_zgelqf_param );

    if( check ) {
        if (N >= M) {
            if(loud > 2) printf("+++ Generate the Q ...");
            dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescQ);
            dplasma_zunglq_param( dague, &qrtree,
                                  (tiled_matrix_desc_t *)&ddescA,
                                  (tiled_matrix_desc_t *)&ddescTS,
                                  (tiled_matrix_desc_t *)&ddescTT,
                                  (tiled_matrix_desc_t *)&ddescQ);
            if(loud > 2) printf("Done\n");

            if(loud > 2) printf("+++ Solve the system ...");
            dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescX, random_seed+1);
            dplasma_zlacpy( dague, PlasmaUpperLower,
                            (tiled_matrix_desc_t *)&ddescX,
                            (tiled_matrix_desc_t *)&ddescB );
            dplasma_zgelqs_param( dague, &qrtree,
                                  (tiled_matrix_desc_t *)&ddescA,
                                  (tiled_matrix_desc_t *)&ddescTS,
                                  (tiled_matrix_desc_t *)&ddescTT,
                                  (tiled_matrix_desc_t *)&ddescX );
            if(loud > 2) printf("Done\n");

            /* Check the orthogonality, factorization and the solution */
            ret |= check_orthogonality( dague, (rank == 0) ? loud : 0,
                                        (tiled_matrix_desc_t *)&ddescQ);
            ret |= check_factorization( dague, (rank == 0) ? loud : 0,
                                        (tiled_matrix_desc_t *)&ddescA0,
                                        (tiled_matrix_desc_t *)&ddescA,
                                        (tiled_matrix_desc_t *)&ddescQ );
            ret |= check_solution( dague, (rank == 0) ? loud : 0,
                                   (tiled_matrix_desc_t *)&ddescA0,
                                   (tiled_matrix_desc_t *)&ddescB,
                                   (tiled_matrix_desc_t *)&ddescX );

        } else {
            printf("Check cannot be performed when M > N\n");
        }

        dague_data_free(ddescA0.mat);
        dague_data_free(ddescQ.mat);
        dague_data_free(ddescB.mat);
        dague_data_free(ddescX.mat);
        tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescA0);
        tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescQ);
        tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescB);
        tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescX);
    }

    dplasma_hqr_finalize( &qrtree );

    cleanup_dague(dague, iparam);

    dague_data_free(ddescA.mat);
    dague_data_free(ddescTS.mat);
    dague_data_free(ddescTT.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescA);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescTS);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescTT);

    return ret;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static int check_orthogonality(dague_context_t *dague, int loud, tiled_matrix_desc_t *Q)
{
    two_dim_block_cyclic_t *twodQ = (two_dim_block_cyclic_t *)Q;
    double normQ = 999999.0;
    double result;
    double eps = LAPACKE_dlamch_work('e');
    int info_ortho;
    int M = Q->m;
    int N = Q->n;
    int minMN = min(M, N);

    PASTE_CODE_ALLOCATE_MATRIX(Id, 1,
        two_dim_block_cyclic, (&Id, matrix_ComplexDouble, matrix_Tile,
                               Q->super.nodes, twodQ->grid.rank,
                               Q->mb, Q->nb, minMN, minMN, 0, 0,
                               minMN, minMN, twodQ->grid.strows, twodQ->grid.stcols, twodQ->grid.rows));

    dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&Id);

    /* Perform Id - Q'Q */
    if ( M >= N ) {
        dplasma_zherk( dague, PlasmaUpper, PlasmaConjTrans,
                       1.0, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    } else {
        dplasma_zherk( dague, PlasmaUpper, PlasmaNoTrans,
                       1.0, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    }

    normQ = dplasma_zlanhe(dague, PlasmaInfNorm, PlasmaUpper, (tiled_matrix_desc_t*)&Id);

    result = normQ / (minMN * eps);
    if ( loud ) {
        printf("============\n");
        printf("Checking the orthogonality of Q \n");
        printf("||Id-Q'*Q||_oo / (M*eps) = %e \n", result);
    }

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        if ( loud ) printf("-- Orthogonality is suspicious ! \n");
        info_ortho=1;
    }
    else {
        if ( loud ) printf("-- Orthogonality is CORRECT ! \n");
        info_ortho=0;
    }

    dague_data_free(Id.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&Id);
    return info_ortho;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static int
check_factorization(dague_context_t *dague, int loud,
                    tiled_matrix_desc_t *Aorig,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *Q)
{
    tiled_matrix_desc_t *subA;
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A;
    double Anorm, Rnorm;
    double result;
    double eps = LAPACKE_dlamch_work('e');
    int info_factorization;
    int M = A->m;
    int N = A->n;
    int minMN = min(M, N);

    PASTE_CODE_ALLOCATE_MATRIX(Residual, 1,
        two_dim_block_cyclic, (&Residual, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, twodA->grid.rank,
                               A->mb, A->nb, M, N, 0, 0,
                               M, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    PASTE_CODE_ALLOCATE_MATRIX(L, 1,
        two_dim_block_cyclic, (&L, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, twodA->grid.rank,
                               A->mb, A->nb, M, M, 0, 0,
                               M, M, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    /* Copy the original A in Residual */
    dplasma_zlacpy( dague, PlasmaUpperLower, Aorig, (tiled_matrix_desc_t *)&Residual );

    /* Extract the L */
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&L);

    subA = tiled_matrix_submatrix( A, 0, 0, M, M );
    dplasma_zlacpy( dague, PlasmaLower, subA, (tiled_matrix_desc_t *)&L );
    free(subA);

    /* Perform Residual = Aorig - L*Q */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans,
                   -1.0, (tiled_matrix_desc_t *)&L, Q,
                    1.0, (tiled_matrix_desc_t *)&Residual);

    /* Free R */
    dague_data_free(L.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&L);

    Rnorm = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&Residual);
    Anorm = dplasma_zlange(dague, PlasmaInfNorm, Aorig);

    result = Rnorm / ( Anorm * minMN * eps);

    if ( loud ) {
        printf("============\n");
        printf("Checking the LQ Factorization \n");
        printf("-- ||A-LQ||_oo/(||A||_oo.M.eps) = %e \n", result );
    }

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        if ( loud ) printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else {
        if ( loud ) printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    dague_data_free(Residual.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&Residual);
    return info_factorization;
}

static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX )
{
    tiled_matrix_desc_t *subB;
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    double eps = LAPACKE_dlamch_work('e');

    subB = tiled_matrix_submatrix( ddescB, 0, 0, ddescA->m, ddescB->n );

    Anorm = dplasma_zlange(dague, PlasmaInfNorm, ddescA);
    Bnorm = dplasma_zlange(dague, PlasmaInfNorm, subB);
    Xnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescX);

    /* Compute A*x-b */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, 1.0, ddescA, ddescX, -1.0, subB);

    /* Compute A' * ( A*x - b ) */
    dplasma_zgemm( dague, PlasmaConjTrans, PlasmaNoTrans,
                   1.0, ddescA, subB, 0., ddescX );

    Rnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescX );
    free(subB);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * ddescA->m * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );

        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).M.eps) = %e \n", result);
    }

    if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}
