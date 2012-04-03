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

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };

static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info = 0;
    int ret = 0;
    int nbpivot = 0;
    double criteria = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M, (DagDouble_t)N))

    if ( M != N && check ) {
        fprintf(stderr, "Check cannot be perfomed with M != N\n");
        check = 0;
    }

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescLU, 1,
        two_dim_block_cyclic, (&ddescLU, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, 1,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, 7657);

    /* Initialize criteria */
    double eps = LAPACKE_dlamch_work('e');
    double Anorm = dplasma_zlange(dague, PlasmaMaxNorm, (tiled_matrix_desc_t *)&ddescA);
    criteria = eps * Anorm;

    /* ((Dague_Complex64_t *) ddescA.mat)[0] = criteria*0.1; */
    /* ((Dague_Complex64_t *) ddescA.mat)[1] = eps; */
    /* ((Dague_Complex64_t *) ddescA.mat)[((tiled_matrix_desc_t *)&ddescA)->mb+1] = eps; */

    dplasma_zlacpy( dague, PlasmaUpperLower,
                    (tiled_matrix_desc_t *)&ddescA,
                    (tiled_matrix_desc_t *)&ddescLU );
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 2354);

    if(loud > 2) printf("Done\n");

    if(loud > 2) printf("+++ Computing getrf_sp ... ");

    /* Computing LU */
    nbpivot = dplasma_zgetrf_sp(dague,criteria,(tiled_matrix_desc_t *)&ddescLU);
    printf("LU decomposition done with %d pivoting\n",nbpivot);

    /* Initialize X */
    int first_solution = 1;
    if(first_solution)
    {
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB,
                        (tiled_matrix_desc_t *)&ddescX );
        dplasma_ztrsm(dague, PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit,
                      1.0, (tiled_matrix_desc_t *)&ddescLU,
                      (tiled_matrix_desc_t *)&ddescX);
        dplasma_ztrsm(dague, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit,
                      1.0, (tiled_matrix_desc_t *)&ddescLU,
                      (tiled_matrix_desc_t *)&ddescX);
    }
    else /* Initialize X to null matrix */
    {
        tiled_matrix_desc_t *descX = (tiled_matrix_desc_t *)&ddescX;
        Dague_Complex64_t   *tab   = (Dague_Complex64_t *) ddescX.mat;
        int i,j;
        for(i = 0; i < (descX->lmt*descX->mb); i++ )
            for(j = 0; j < (descX->lnt*descX->nb); j++)
                tab[j*(descX->lmt*descX->mb)+i] = (Dague_Complex64_t) 0.;
    }

    dplasma_zgerfs(dague,
                   (tiled_matrix_desc_t*) &ddescA,
                   (tiled_matrix_desc_t*) &ddescLU,
                   (tiled_matrix_desc_t*) &ddescB,
                   (tiled_matrix_desc_t*) &ddescX);
    if(check)
    {
        Dague_Complex64_t *A, *LU, *B, *X;
        lapack_int ipiv[N];
        int i;
        for(i = 0; i < N; i++)
            ipiv[i] = i+1;
        double ferr[NRHS];
        double berr[NRHS];

        A = (Dague_Complex64_t *)malloc((ddescA.super.lm)*(ddescA.super.n)*sizeof(Dague_Complex64_t));
        LU = (Dague_Complex64_t *)malloc((ddescLU.super.lm)*(ddescLU.super.n)*sizeof(Dague_Complex64_t));
        B = (Dague_Complex64_t *)malloc((ddescB.super.lm)*(ddescB.super.n)*sizeof(Dague_Complex64_t));
        X = (Dague_Complex64_t *)malloc((ddescX.super.lm)*(ddescX.super.n)*sizeof(Dague_Complex64_t));

        twoDBC_ztolapack( &ddescA, A, LDA );
        twoDBC_ztolapack( &ddescLU, LU, LDA );
        twoDBC_ztolapack( &ddescB, B, LDB );
        twoDBC_ztolapack( &ddescX, X, LDB );

        int ret = LAPACKE_zgerfs( LAPACK_COL_MAJOR, 'N', N, NRHS, A, LDA, LU, LDA, ipiv, B, LDB, X, LDB, ferr, berr );
        printf("The refinement is %s\n",ret?"bad":"good");

        Dague_Complex64_t alpha =  -1.;
        Dague_Complex64_t beta  = 1.;

        cblas_zgemm(CblasColMajor,
                    (CBLAS_TRANSPOSE)CblasNoTrans, (CBLAS_TRANSPOSE)CblasNoTrans,
                    M, NRHS, M, CBLAS_SADDR(alpha), A, LDA, X, LDB,
                    CBLAS_SADDR(beta), B, LDB);

        double Bnorm       = dplasma_zlange(dague, PlasmaMaxNorm, (tiled_matrix_desc_t *)&ddescB);
        double Xnorm, Rnorm, Znorm;

        double *work  = (double *)malloc(M* sizeof(double));
        Xnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'm', M, NRHS, X, LDB, work);
        Rnorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'm', M, NRHS, B, LDB, work);

        double result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * M * eps ) ;

        if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 10.0) )
            printf("-- Lapack solution is suspicious ! \n");
        else
            printf("-- Lapack solution is CORRECT ! \n");
        printf("Lapack berr = %e\n",berr[0]);

        free(A);
        free(LU);
        free(B);
        free(X);
    }

    if(loud > 2) printf("Done.\n");

    if ( info < 0 ) {
        if( rank == 0 && loud ) printf("-- Factorization is suspicious (info = %d) ! \n", info );
        ret |= 1;
    }
    else if ( check ) {
        /* Check the solution */
        ret |= check_solution( dague, (rank == 0) ? loud : 0,
                               (tiled_matrix_desc_t *)&ddescA,
                               (tiled_matrix_desc_t *)&ddescB,
                               (tiled_matrix_desc_t *)&ddescX);
    }

    dague_data_free(ddescLU.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescLU);
    dague_data_free(ddescB.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescB);
    dague_data_free(ddescX.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescX);
    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);

    cleanup_dague(dague, iparam);

    return EXIT_SUCCESS;
}



static int check_solution( dague_context_t *dague, int loud,
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

    Anorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescA);
    Bnorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescB);
    Xnorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescX);

    /* Compute b - A*x */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, -1.0, ddescA, ddescX, 1.0, ddescB);

    Rnorm = dplasma_zlange(dague, PlasmaMaxNorm, ddescB);

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
