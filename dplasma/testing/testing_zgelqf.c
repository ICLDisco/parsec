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

#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
#include "dplasma/cores/cuda_stsmlq.h"
#endif

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
    iparam_default_ibnbmb(iparam, 48, 192, 192);
    iparam[IPARAM_SNB] = 2;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGELQF, ((DagDouble_t)M, (DagDouble_t)N));

    LDA = max(M, LDA);
    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1,
        two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, IB, NB, MT*IB, N, 0, 0,
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescQ, check,
        two_dim_block_cyclic, (&ddescQ, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    /* Check the solution */
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, check,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescX, check,
        two_dim_block_cyclic, (&ddescX, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    /* load the GPU kernel */
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    if(iparam[IPARAM_NGPUS] > 0)
    {
        if(loud) printf("+++ Load GPU kernel ... ");
        if(0 != stsmlq_cuda_init(dague, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescT))
        {
            printf("XXX Unable to load GPU kernel.\n");
            exit(3);
        }
        if(loud > 3) printf("Done\n");
    }
#endif

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, 3872);
    if( check )
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescA0 );
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescT);
    if(loud > 3) printf("Done\n");

    /* Create DAGuE */
    PASTE_CODE_ENQUEUE_KERNEL(dague, zgelqf,
                              ((tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescT));

    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(dague, zgelqf);
    dplasma_zgelqf_Destruct( DAGUE_zgelqf );

    if( check ) {
        if(loud > 2) printf("+++ Generate the Q ...");
        dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescQ);
        dplasma_zunglq( dague, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescT,
                        (tiled_matrix_desc_t *)&ddescQ);
        if(loud > 2) printf("Done\n");

        /* if(loud > 2) printf("+++ Solve the system ..."); */
        /* dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescX, 2354); */
        /* dplasma_zlacpy( dague, PlasmaUpperLower, */
        /*                 (tiled_matrix_desc_t *)&ddescX, (tiled_matrix_desc_t *)&ddescB ); */
        /* dplasma_zgelqr( dague, */
        /*                (tiled_matrix_desc_t *)&ddescA, */
        /*                (tiled_matrix_desc_t *)&ddescT, */
        /*                (tiled_matrix_desc_t *)&ddescX); */
        /* if(loud > 2) printf("Done\n"); */

        /* Check the orthogonality, factorization and the solution */
        ret |= check_orthogonality(dague, (rank == 0) ? loud : 0,
                                   (tiled_matrix_desc_t *)&ddescQ);
        ret |= check_factorization(dague, (rank == 0) ? loud : 0,
                                   (tiled_matrix_desc_t *)&ddescA0,
                                   (tiled_matrix_desc_t *)&ddescA,
                                   (tiled_matrix_desc_t *)&ddescQ);
        /* ret |= check_solution(dague, (rank == 0) ? loud : 0, */
        /*                            (tiled_matrix_desc_t *)&ddescA0, */
        /*                            (tiled_matrix_desc_t *)&ddescB, */
        /*                            (tiled_matrix_desc_t *)&ddescX); */

        dague_data_free(ddescA0.mat);
        dague_data_free(ddescQ.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescA0);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescQ);
    }

#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    if(iparam[IPARAM_NGPUS] > 0)
    {
        stsmlq_cuda_fini(dague);
    }
#endif

    cleanup_dague(dague, iparam);

    dague_data_free(ddescA.mat);
    dague_data_free(ddescT.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescT);

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
                               Q->super.nodes, Q->super.cores, twodQ->grid.rank,
                               Q->mb, Q->nb, minMN, minMN, 0, 0,
                               minMN, minMN, twodQ->grid.strows, twodQ->grid.stcols, twodQ->grid.rows));

    dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&Id);

    /* Perform Id - Q'Q (could be done with Herk) */
    if ( M >= N ) {
      dplasma_zgemm( dague, PlasmaConjTrans, PlasmaNoTrans,
                     1.0, Q, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    } else {
      dplasma_zgemm( dague, PlasmaNoTrans, PlasmaConjTrans,
                     1.0, Q, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    }

    normQ = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&Id);

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
    dague_ddesc_destroy((dague_ddesc_t*)&Id);
    return info_ortho;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */

static int check_factorization(dague_context_t *dague, int loud, tiled_matrix_desc_t *Aorig, tiled_matrix_desc_t *A, tiled_matrix_desc_t *Q)
{
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
                               A->super.nodes, A->super.cores, twodA->grid.rank,
                               A->mb, A->nb, M, N, 0, 0,
                               M, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    PASTE_CODE_ALLOCATE_MATRIX(R, 1,
        two_dim_block_cyclic, (&R, matrix_ComplexDouble, matrix_Tile,
                               A->super.nodes, A->super.cores, twodA->grid.rank,
                               A->mb, A->nb, N, N, 0, 0,
                               N, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    /* Copy the original A in Residual */
    dplasma_zlacpy( dague, PlasmaUpperLower, Aorig, (tiled_matrix_desc_t *)&Residual );

    /* Extract the R */
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&R);
    dplasma_zlacpy( dague, PlasmaLower, A, (tiled_matrix_desc_t *)&R );

    /* Perform Residual = Aorig - L*Q */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans,
                   -1.0, (tiled_matrix_desc_t *)&R, Q,
                   1.0, (tiled_matrix_desc_t *)&Residual);

    /* Free R */
    dague_data_free(R.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&R);

    Rnorm = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&Residual);
    Anorm = dplasma_zlange(dague, PlasmaInfNorm, Aorig);

    result = Rnorm / ( Anorm * minMN * eps);

    if ( loud ) {
        printf("============\n");
        printf("Checking the LQ Factorization \n");
        printf("-- ||A-LQ||_oo/(||A||_oo.N.eps) = %e \n", result );
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
    dague_ddesc_destroy((dague_ddesc_t*)&Residual);
    return info_factorization;
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

    Anorm = dplasma_zlange(dague, PlasmaInfNorm, ddescA);
    Bnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescB);
    Xnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescX);

    /* Compute b - A*x */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, -1.0, ddescA, ddescX, 1.0, ddescB);

    Rnorm = dplasma_zlange(dague, PlasmaInfNorm, ddescB);

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
