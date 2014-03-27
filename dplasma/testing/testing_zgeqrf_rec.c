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
#include "dague/devices/cuda/dev_cuda.h"
#include "../lib/memory_pool.h"
#include "../lib/zgeqrf_rec.h"

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

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 32, 200, 200);
    iparam[IPARAM_SMB] = 4;
    iparam[IPARAM_SNB] = 1;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGEQRF, ((DagDouble_t)M, (DagDouble_t)N));

    LDA = max(M, LDA);
    LDB = max(M, LDB);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1,
        two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, matrix_Tile,
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
                               M, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               M, NRHS, SMB, SNB, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, 3872);
    if( check )
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescA0 );
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescT);
    if(loud > 3) printf("Done\n");

    /* load the GPU kernel */
#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        if(loud > 3) printf("+++ Load GPU kernel ... ");
        dague_gpu_data_register(dague,
                                (dague_ddesc_t*)&ddescA,
                                MT*NT, MB*NB*sizeof(dague_complex64_t) );
        dague_gpu_data_register(dague,
                                (dague_ddesc_t*)&ddescT,
                                MT*NT, IB*NB*sizeof(dague_complex64_t) );
        if(loud > 3) printf("Done\n");
    }
#endif

    /* Create DAGuE */
    PASTE_CODE_ENQUEUE_KERNEL(dague, zgeqrf_rec,
                              ((tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescT));

    if (iparam[IPARAM_SMALL_NB] == 0) {
        iparam[IPARAM_SMALL_NB] = iparam[IPARAM_IB]; /* default small nb = ib */
    }

    {
        dague_zgeqrf_rec_handle_t* myzgeqrf_handle = (dague_zgeqrf_rec_handle_t*)DAGUE_zgeqrf_rec;
        myzgeqrf_handle->smallnb = iparam[IPARAM_SMALL_NB];
        if ( (myzgeqrf_handle->smallnb % iparam[IPARAM_IB] != 0) &&
             (myzgeqrf_handle->smallnb != iparam[IPARAM_NB]) )
        {
            myzgeqrf_handle->smallnb = (myzgeqrf_handle->smallnb / iparam[IPARAM_IB]) * iparam[IPARAM_IB];
            myzgeqrf_handle->smallnb = dplasma_imin( myzgeqrf_handle->smallnb, iparam[IPARAM_NB] );
            fprintf(stderr, "Small nb should be a muliple of IB or equal to NB: set to min( (snb/IB)*IB, NB ) = %d\n", myzgeqrf_handle->smallnb);
        }
    }

    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(dague, zgeqrf_rec);
    dplasma_zgeqrf_rec_Destruct( DAGUE_zgeqrf_rec );
#if defined(HAVE_CUDA)
    if(iparam[IPARAM_NGPUS] > 0) {
        dague_gpu_data_unregister((dague_ddesc_t*)&ddescA);
        dague_gpu_data_unregister((dague_ddesc_t*)&ddescT);
    }
#endif

    if( check ) {

        /* remove GPU devices, they are not required for check */
        for (int i = 1; i < dague_nb_devices; i++) {
                dague_device_remove(dague_devices_get(i));
        }

        if (M >= N) {
            if(loud > 2) printf("+++ Generate the Q ...");
            dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescQ);
            dplasma_zungqr( dague,
                            (tiled_matrix_desc_t *)&ddescA,
                            (tiled_matrix_desc_t *)&ddescT,
                            (tiled_matrix_desc_t *)&ddescQ);
            if(loud > 2) printf("Done\n");

            if(loud > 2) printf("+++ Solve the system ...");
            dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescX, 2354);
            dplasma_zlacpy( dague, PlasmaUpperLower,
                            (tiled_matrix_desc_t *)&ddescX,
                            (tiled_matrix_desc_t *)&ddescB );
            dplasma_zgeqrs( dague,
                            (tiled_matrix_desc_t *)&ddescA,
                            (tiled_matrix_desc_t *)&ddescT,
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
            printf("Check cannot be performed when N > M\n");
        }

        dague_data_free(ddescA0.mat);
        dague_data_free(ddescQ.mat);
        dague_data_free(ddescB.mat);
        dague_data_free(ddescX.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA0);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescQ);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescX);
    }

    dague_data_free(ddescA.mat);
    dague_data_free(ddescT.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescT);

    cleanup_dague(dague, iparam);

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
        printf("||Id-Q'*Q||_oo / (N*eps) = %e \n", result);
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
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&Id);
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

    PASTE_CODE_ALLOCATE_MATRIX(R, 1,
        two_dim_block_cyclic, (&R, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, twodA->grid.rank,
                               A->mb, A->nb, N, N, 0, 0,
                               N, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    /* Copy the original A in Residual */
    dplasma_zlacpy( dague, PlasmaUpperLower, Aorig, (tiled_matrix_desc_t *)&Residual );

    /* Extract the R */
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&R);

    subA = tiled_matrix_submatrix( A, 0, 0, N, N );
    dplasma_zlacpy( dague, PlasmaUpper, subA, (tiled_matrix_desc_t *)&R );
    free(subA);

    /* Perform Residual = Aorig - Q*R */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans,
                   -1.0, Q, (tiled_matrix_desc_t *)&R,
                    1.0, (tiled_matrix_desc_t *)&Residual);

    /* Free R */
    dague_data_free(R.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&R);

    Rnorm = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&Residual);
    Anorm = dplasma_zlange(dague, PlasmaInfNorm, Aorig);

    result = Rnorm / ( Anorm * minMN * eps);

    if ( loud ) {
        printf("============\n");
        printf("Checking the QR Factorization \n");
        printf("-- ||A-QR||_oo/(||A||_oo.N.eps) = %e \n", result );
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
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&Residual);
    return info_factorization;
}

static int check_solution( dague_context_t *dague, int loud,
                           tiled_matrix_desc_t *ddescA,
                           tiled_matrix_desc_t *ddescB,
                           tiled_matrix_desc_t *ddescX )
{
    tiled_matrix_desc_t *subX;
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    double eps = LAPACKE_dlamch_work('e');

    subX = tiled_matrix_submatrix( ddescX, 0, 0, ddescA->n, ddescX->n );

    Anorm = dplasma_zlange(dague, PlasmaInfNorm, ddescA);
    Bnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescB);
    Xnorm = dplasma_zlange(dague, PlasmaInfNorm, subX);

    /* Compute A*x-b */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, 1.0, ddescA, subX, -1.0, ddescB);

    /* Compute A' * ( A*x - b ) */
    dplasma_zgemm( dague, PlasmaConjTrans, PlasmaNoTrans,
                   1.0, ddescA, ddescB, 0., subX );

    Rnorm = dplasma_zlange( dague, PlasmaInfNorm, subX );
    free(subX);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * ddescA->n * eps ) ;

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
