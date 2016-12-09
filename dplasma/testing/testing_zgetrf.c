/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

static int check_solution( parsec_context_t *parsec, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX );

static int check_inverse( parsec_context_t *parsec, int loud,
                          tiled_matrix_desc_t *ddescA,
                          tiled_matrix_desc_t *ddescInvA,
                          tiled_matrix_desc_t *ddescI );

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info = 0;
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(PARSEC_HAVE_CUDA) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M,(DagDouble_t)N));

    LDA = max(M, LDA);

    if ( M != N && check ) {
        fprintf(stderr, "Check is impossible if M != N\n");
        check = 0;
    }

    if ( P > 1 ) {
        fprintf(stderr, "This function cannot be used with a 2D block cyclic distribution for now\n");
        return 1;
    }

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                               two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIV, 1,
        two_dim_block_cyclic, (&ddescIPIV, matrix_Integer, matrix_Tile,
                               nodes, rank, 1, NB, 1, dplasma_imin(M, N), 0, 0,
                               1, dplasma_imin(M, N), SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
                               two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    /* Random B check */
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
                               two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
                               two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    /* Inverse check */
    PASTE_CODE_ALLOCATE_MATRIX(ddescInvA, check_inv,
                               two_dim_block_cyclic, (&ddescInvA, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescI, check_inv,
                               two_dim_block_cyclic, (&ddescI, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescA, 3872);
    if ( check ) {
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA,
                        (tiled_matrix_desc_t *)&ddescA0 );
        dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescB, 2354);
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB,
                        (tiled_matrix_desc_t *)&ddescX );
    }
    if ( check_inv ) {
        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescI);
        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescInvA);
    }
    if(loud > 2) printf("Done\n");

    /* Create PaRSEC */
    if(loud > 2) printf("+++ Computing getrf ... ");
    PASTE_CODE_ENQUEUE_KERNEL(parsec, zgetrf,
                              ((tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescIPIV,
                               &info));
    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(parsec, zgetrf);
    dplasma_zgetrf_Destruct( PARSEC_zgetrf );
    if(loud > 2) printf("Done.\n");

    if ( info != 0 ) {
        if( rank == 0 && loud ) printf("-- Factorization is suspicious (info = %d) ! \n", info );
        ret |= 1;
    }
    else if ( check ) {
        /*
         * First check with a right hand side
         */
        dplasma_zgetrs(parsec, PlasmaNoTrans,
                       (tiled_matrix_desc_t *)&ddescA,
                       (tiled_matrix_desc_t *)&ddescIPIV,
                       (tiled_matrix_desc_t *)&ddescX );

        /* Check the solution */
        ret |= check_solution( parsec, (rank == 0) ? loud : 0,
                               (tiled_matrix_desc_t *)&ddescA0,
                               (tiled_matrix_desc_t *)&ddescB,
                               (tiled_matrix_desc_t *)&ddescX);

        /*
         * Second check with inverse
         */
        if ( check_inv ) {
            dplasma_zgetrs(parsec, PlasmaNoTrans,
                           (tiled_matrix_desc_t *)&ddescA,
                           (tiled_matrix_desc_t *)&ddescIPIV,
                           (tiled_matrix_desc_t *)&ddescInvA );

            /* Check the solution */
            ret |= check_inverse(parsec, (rank == 0) ? loud : 0,
                                 (tiled_matrix_desc_t *)&ddescA0,
                                 (tiled_matrix_desc_t *)&ddescInvA,
                                 (tiled_matrix_desc_t *)&ddescI);
        }
    }

    if ( check ) {
        parsec_data_free(ddescA0.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0);
        parsec_data_free(ddescB.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);
        parsec_data_free(ddescX.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescX);
        if ( check_inv ) {
            parsec_data_free(ddescInvA.mat);
            tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescInvA);
            parsec_data_free(ddescI.mat);
            tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescI);
        }
    }

    parsec_data_free(ddescA.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    parsec_data_free(ddescIPIV.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescIPIV);

    cleanup_parsec(parsec, iparam);

    return ret;
}

static int check_solution( parsec_context_t *parsec, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int m = ddescB->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescA);
    Bnorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescB);
    Xnorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescX);

    /* Compute b - A*x */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0, ddescA, ddescX, 1.0, ddescB);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescB);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * m * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );
        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
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

static int check_inverse( parsec_context_t *parsec, int loud,
                          tiled_matrix_desc_t *ddescA,
                          tiled_matrix_desc_t *ddescInvA,
                          tiled_matrix_desc_t *ddescI )
{
    int info_solution;
    double Anorm    = 0.0;
    double InvAnorm = 0.0;
    double Rnorm, result;
    int m = ddescA->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm    = dplasma_zlange(parsec, PlasmaInfNorm, ddescA   );
    InvAnorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescInvA);

    /* Compute I - A*A^{-1} */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0, ddescA, ddescInvA, 1.0, ddescI);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, ddescI);

    result = Rnorm / ( ( Anorm * InvAnorm ) * m * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||A^{-1}||_oo = %e, ||A A^{-1} - I||_oo = %e\n",
                    Anorm, InvAnorm, Rnorm );
        printf("-- ||AA^{-1}-I||_oo/((||A||_oo||A^{-1}||_oo).N.eps) = %e \n", result);
    }

    if (  isnan(Rnorm) || isinf(Rnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}
