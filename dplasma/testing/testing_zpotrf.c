/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "flops.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#if defined(HAVE_CUDA) && defined(PRECISION_s)
#include "dplasma/cores/cuda_sgemm.h"
#endif

static int check_factorization( dague_context_t *dague, int loud, PLASMA_enum uplo,
                                tiled_matrix_desc_t *A,
                                tiled_matrix_desc_t *A0 );
static int check_solution( dague_context_t *dague, int loud, PLASMA_enum uplo,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX );

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaLower;
    int info = 0;
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZPOTRF, ((DagDouble_t)N));

    /* initializing matrix structure */
    LDA = max( LDA, N );
    LDB = max( LDB, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                   nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));

    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
        sym_two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble,
                                   nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));

    PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( dague, (double)(N), uplo,
                    (tiled_matrix_desc_t *)&ddescA, 1358);
    if ( check ) {
        dplasma_zlacpy( dague, uplo,
                        (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescA0 );
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 3872);
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );
    }
    if(loud > 3) printf("Done\n");

    /* load the GPU kernel */
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0)
        {
            if(loud > 3) printf("+++ Load GPU kernel ... ");
            if(0 != gpu_kernel_init_zgemm(dague, (tiled_matrix_desc_t *)&ddescA))
                {
                    printf("XXX Unable to load GPU kernel.\n");
                    exit(3);
                }
            dague_gpu_data_register(dague,
                                    (dague_ddesc_t*)&ddescA,
                                    MT*NT, MB*NB*sizeof(Dague_Complex64_t) );
            if(loud > 3) printf("Done\n");
        }
#endif

    PASTE_CODE_ENQUEUE_KERNEL(dague, zpotrf,
                              (uplo, (tiled_matrix_desc_t*)&ddescA, &info));
    PASTE_CODE_PROGRESS_KERNEL(dague, zpotrf);

    dplasma_zpotrf_Destruct( DAGUE_zpotrf );

    if ( info != 0 ) {
        if( rank == 0 && loud ) printf("-- Factorization is suspicious (info = %d) ! \n", info);
        ret |= 1;
    }
    else if ( check ) {

        dplasma_zpotrs(dague, uplo,
                       (tiled_matrix_desc_t *)&ddescA,
                       (tiled_matrix_desc_t *)&ddescX );

        /* Check the solution */
        ret |= check_factorization( dague, (rank == 0) ? loud : 0, uplo,
                                    (tiled_matrix_desc_t *)&ddescA,
                                    (tiled_matrix_desc_t *)&ddescA0);

        ret |= check_solution( dague, (rank == 0) ? loud : 0, uplo,
                               (tiled_matrix_desc_t *)&ddescA0,
                               (tiled_matrix_desc_t *)&ddescB,
                               (tiled_matrix_desc_t *)&ddescX);
    }

    if ( check ) {
        dague_data_free(ddescA0.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescA0 );
        dague_data_free(ddescB.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescB );
        dague_data_free(ddescX.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescX );
    }

#if defined(HAVE_CUDA) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0) {
        dague_gpu_data_unregister();
        dague_gpu_kernel_fini(dague, "zgemm");
    }
#endif
    cleanup_dague(dague, iparam);

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescA);

    return ret;
}

static int check_factorization( dague_context_t *dague, int loud, PLASMA_enum uplo,
                                tiled_matrix_desc_t *A,
                                tiled_matrix_desc_t *A0 )
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A0;
    int info_factorization;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double result = 0.0;
    int M = A->m;
    int N = A->n;
    double eps = LAPACKE_dlamch_work('e');
    PLASMA_enum side;

    PASTE_CODE_ALLOCATE_MATRIX(L1, 1,
        sym_two_dim_block_cyclic, (&L1, matrix_ComplexDouble,
                                   A->super.nodes, A->super.cores, twodA->grid.rank,
                                   A->mb, A->nb, M, N, 0, 0,
                                   M, N, twodA->grid.rows, uplo));
    PASTE_CODE_ALLOCATE_MATRIX(L2, 1,
        two_dim_block_cyclic, (&L2, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, A->super.cores, twodA->grid.rank,
                               A->mb, A->nb, M, N, 0, 0,
                               M, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    dplasma_zlacpy( dague, uplo, A, (tiled_matrix_desc_t *)&L1 );
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0.,(tiled_matrix_desc_t *)&L2 );
    dplasma_zlacpy( dague, uplo, A, (tiled_matrix_desc_t *)&L2 );

    side = (uplo == PlasmaUpper ) ? PlasmaLeft : PlasmaRight;

    /* Compute LL' or U'U  */
    dplasma_ztrmm( dague, side, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0,
                   (tiled_matrix_desc_t*)&L1,
                   (tiled_matrix_desc_t*)&L2);

    /* compute LL' - A or U'U - A */
    dplasma_zgeadd( dague, uplo, -1.0, A0,
                    (tiled_matrix_desc_t*)&L2);

    Anorm = dplasma_zlanhe(dague, PlasmaMaxNorm, uplo, A0);
    Rnorm = dplasma_zlanhe(dague, PlasmaMaxNorm, uplo,
                           (tiled_matrix_desc_t*)&L2);

    result = Rnorm / ( Anorm * N * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Cholesky factorization \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e",
                    Anorm );
        if ( loud > 3 )
            printf( ", ||L'L-A||_oo = %e\n",
                    Rnorm );

        printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n", result);
    }

    if ( isnan(Rnorm)
         || isinf(Rnorm)
         || isnan(result)
         || isinf(result)
         || (result > 60.0) ) {
        if( loud ) printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else{
        if( loud ) printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    dague_data_free(L1.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&L1);
    dague_data_free(L2.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&L2);

    return info_factorization;
}

/*
 * This function destroy B
 */
static int check_solution( dague_context_t *dague, int loud, PLASMA_enum uplo,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int N = ddescB->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm = dplasma_zlanhe(dague, PlasmaMaxNorm, uplo, ddescA);
    Bnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescB);
    Xnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescX);

    /* Compute A*x */
    dplasma_zhemm( dague, PlasmaLeft, uplo, -1.0, ddescA, ddescX, 1.0, ddescB);

    Rnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescB);

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
