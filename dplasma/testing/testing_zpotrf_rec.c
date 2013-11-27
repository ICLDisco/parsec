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
#if defined(HAVE_CUDA)
#include "dplasma/cores/cuda_zgemm.h"
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
#if defined(HAVE_CUDA)
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
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   N, N, P, uplo));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplghe( dague, (double)(N), uplo,
                    (tiled_matrix_desc_t *)&ddescA, 3872);
    if(loud > 3) printf("Done\n");

    /* load the GPU kernel */
#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        if(loud > 3) printf("+++ Load GPU kernel ... ");
        dague_gpu_data_register(dague,
                                (dague_ddesc_t*)&ddescA,
                                MT*NT, MB*NB*sizeof(dague_complex64_t) );
        if(loud > 3) printf("Done\n");
    }
#endif

    PASTE_CODE_ENQUEUE_KERNEL(dague, zpotrf_rec,
                              (uplo, (tiled_matrix_desc_t*)&ddescA, &info));
    PASTE_CODE_PROGRESS_KERNEL(dague, zpotrf_rec);

    dplasma_zpotrf_rec_Destruct( DAGUE_zpotrf_rec );
    dague_handle_sync_ids(); /* recursive DAGs are not synchronous on ids */
#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        dague_gpu_data_unregister((dague_ddesc_t*)&ddescA);
    }
#endif

    if( 0 == rank && info != 0 ) {
        printf("-- Factorization is suspicious (info = %d) ! \n", info);
        ret |= 1;
    }
    if( !info && check ) {
        /* Check the factorization */
        PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
            sym_two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble,
                                       nodes, rank, MB, NB, LDA, N, 0, 0,
                                       N, N, P, uplo));
        dplasma_zplghe( dague, (double)(N), uplo,
                        (tiled_matrix_desc_t *)&ddescA0, 3872);

        ret |= check_factorization( dague, (rank == 0) ? loud : 0, uplo,
                                    (tiled_matrix_desc_t *)&ddescA,
                                    (tiled_matrix_desc_t *)&ddescA0);

        /* Check the solution */
        PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
            two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));
        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, 2354);

        PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
            two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                   N, NRHS, SMB, SNB, P));
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescB, (tiled_matrix_desc_t *)&ddescX );

        dplasma_zpotrs(dague, uplo,
                       (tiled_matrix_desc_t *)&ddescA,
                       (tiled_matrix_desc_t *)&ddescX );

        ret |= check_solution( dague, (rank == 0) ? loud : 0, uplo,
                               (tiled_matrix_desc_t *)&ddescA0,
                               (tiled_matrix_desc_t *)&ddescB,
                               (tiled_matrix_desc_t *)&ddescX);

        /* Cleanup */
        dague_data_free(ddescA0.mat); ddescA0.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0 );
        dague_data_free(ddescB.mat); ddescB.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB );
        dague_data_free(ddescX.mat); ddescX.mat = NULL;
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescX );
    }

    dague_data_free(ddescA.mat); ddescA.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);

    cleanup_dague(dague, iparam);
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
                                   A->super.nodes, twodA->grid.rank,
                                   A->mb, A->nb, M, N, 0, 0,
                                   M, N, twodA->grid.rows, uplo));
    PASTE_CODE_ALLOCATE_MATRIX(L2, 1,
        two_dim_block_cyclic, (&L2, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, twodA->grid.rank,
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

    printf("zlanhe\n");

    Anorm = dplasma_zlanhe(dague, PlasmaInfNorm, uplo, A0);
    Rnorm = dplasma_zlanhe(dague, PlasmaInfNorm, uplo,
                           (tiled_matrix_desc_t*)&L2);
     printf("zlanhe end\n");

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

    dague_data_free(L1.mat); L1.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&L1);
    dague_data_free(L2.mat); L2.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&L2);

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

    Anorm = dplasma_zlanhe(dague, PlasmaInfNorm, uplo, ddescA);
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

