/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */
#include "dplasma.h"
#include <math.h>
#include <lapacke.h>

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_check
 *
 * check_zpotrf - Check the correctness of the Cholesky factorization computed
 * Cholesky functions with the following criteria:
 *
 *    \f[ ||L'L-A||_oo/(||A||_oo.N.eps) < 60. \f]
 *
 *  or
 *
 *    \f[ ||UU'-A||_oo/(||A||_oo.N.eps) < 60. \f]
 *
 *  where A is the original matrix, and L, or U, the result of the Cholesky
 *  factorization.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] loud
 *          The level of verbosity required.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A and A0 are referenced;
 *          = PlasmaLower: Lower triangle of A and A0 are referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A result of the Cholesky
 *          factorization. Holds L or U. If uplo == PlasmaUpper, the only the
 *          upper part is referenced, otherwise if uplo == PlasmaLower, the
 *          lower part is referenced.
 *
 * @param[in] A0
 *          Descriptor of the original distributed matrix A before
 *          factorization. If uplo == PlasmaUpper, the only the upper part is
 *          referenced, otherwise if uplo == PlasmaLower, the lower part is
 *          referenced.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 1, if the result is incorrect
 *          \retval 0, if the result is correct
 *
 ******************************************************************************/
int check_zpotrf( dague_context_t *dague, int loud,
                  PLASMA_enum uplo,
                  tiled_matrix_desc_t *A,
                  tiled_matrix_desc_t *A0 )
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A0;
    two_dim_block_cyclic_t LLt;
    int info_factorization;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double result = 0.0;
    int M = A->m;
    int N = A->n;
    double eps = LAPACKE_dlamch_work('e');
    PLASMA_enum side;

    two_dim_block_cyclic_init(&LLt, matrix_ComplexDouble, matrix_Tile,
                              A->super.nodes, twodA->grid.rank,
                              A->mb, A->nb, M, N, 0, 0,
                              M, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows);

    LLt.mat = dague_data_allocate((size_t)LLt.super.nb_local_tiles *
                                  (size_t)LLt.super.bsiz *
                                  (size_t)dague_datadist_getsizeoftype(LLt.super.mtype));

    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0.,(tiled_matrix_desc_t *)&LLt );
    dplasma_zlacpy( dague, uplo, A, (tiled_matrix_desc_t *)&LLt );

    /* Compute LL' or U'U  */
    side = (uplo == PlasmaUpper ) ? PlasmaLeft : PlasmaRight;
    dplasma_ztrmm( dague, side, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0,
                   A, (tiled_matrix_desc_t*)&LLt);

    /* compute LL' - A or U'U - A */
    dplasma_zgeadd( dague, uplo, -1.0, A0,
                    (tiled_matrix_desc_t*)&LLt);

    Anorm = dplasma_zlanhe(dague, PlasmaInfNorm, uplo, A0);
    Rnorm = dplasma_zlanhe(dague, PlasmaInfNorm, uplo,
                           (tiled_matrix_desc_t*)&LLt);

    result = Rnorm / ( Anorm * N * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Cholesky factorization \n");

        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||L'L-A||_oo = %e\n", Anorm, Rnorm );

        printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n", result);
    }

    if ( isnan(Rnorm)  || isinf(Rnorm)  ||
         isnan(result) || isinf(result) ||
         (result > 60.0) )
    {
        if( loud ) printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else
    {
        if( loud ) printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    dague_data_free(LLt.mat); LLt.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&LLt);

    return info_factorization;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_check
 *
 * check_zaxmb - Returns the result of the following test
 *
 *    \f[ (|| A x - b ||_oo / ((||A||_oo * ||x||_oo + ||b||_oo) * N * eps) ) < 60. \f]
 *
 *  where A is the original matrix, b the original right hand side, and x the
 *  solution computed through any factorization.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] loud
 *          The level of verbosity required.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A result of the Cholesky
 *          factorization. Holds L or U. If uplo == PlasmaUpper, the only the
 *          upper part is referenced, otherwise if uplo == PlasmaLower, the
 *          lower part is referenced.
 *
 * @param[in,out] b
 *          Descriptor of the original distributed right hand side b.
 *          On exit, b is overwritten by (b - A * x).
 *
 * @param[in] x
 *          Descriptor of the solution to the problem, x.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 1, if the result is incorrect
 *          \retval 0, if the result is correct
 *
 ******************************************************************************/
int check_zaxmb( dague_context_t *dague, int loud,
                 PLASMA_enum uplo,
                 tiled_matrix_desc_t *A,
                 tiled_matrix_desc_t *b,
                 tiled_matrix_desc_t *x )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int N = b->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm = dplasma_zlanhe(dague, PlasmaInfNorm, uplo, A);
    Bnorm = dplasma_zlange(dague, PlasmaInfNorm, b);
    Xnorm = dplasma_zlange(dague, PlasmaInfNorm, x);

    /* Compute b - A*x */
    dplasma_zhemm( dague, PlasmaLeft, uplo, -1.0, A, x, 1.0, b);

    Rnorm = dplasma_zlange(dague, PlasmaInfNorm, b);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * N * eps ) ;

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
