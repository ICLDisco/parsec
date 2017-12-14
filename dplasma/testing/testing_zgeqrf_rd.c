/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/two_dim_tabular.h"

static int check_orthogonality(parsec_context_t *parsec, int loud,
                               parsec_tiled_matrix_dc_t *Q);
static int check_factorization(parsec_context_t *parsec, int loud,
                               parsec_tiled_matrix_dc_t *Aorig,
                               parsec_tiled_matrix_dc_t *A,
                               parsec_tiled_matrix_dc_t *Q);
static int check_solution( parsec_context_t *parsec, int loud,
                           parsec_tiled_matrix_dc_t *dcA,
                           parsec_tiled_matrix_dc_t *dcB,
                           parsec_tiled_matrix_dc_t *dcX );

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    int seed;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 48, 192, 192);
    iparam[IPARAM_SMB] = 2;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGEQRF, ((DagDouble_t)M, (DagDouble_t)N));

    seed = getpid();
#if defined(PARSEC_HAVE_MPI)
    /* If we are in a distributed run, broadcast the seed of rank 0 */
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif  /* defined(PARSEC_HAVE_MPI) */

    LDA = max(M, LDA);
    /* initializing matrix structure */
    two_dim_tabular_t dcA;
    two_dim_tabular_init(&dcA, matrix_ComplexDouble,
                         nodes, rank, MB, NB, LDA, N, 0, 0,
                         M, N, NULL);
    two_dim_tabular_set_random_table(&dcA, seed);
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, "dcA");

    two_dim_tabular_t dcT;
    two_dim_tabular_init(&dcT, matrix_ComplexDouble,
                         nodes, rank, IB, NB, MT*IB, N, 0, 0,
                         MT*IB, N, NULL);
    two_dim_td_table_clone_table_structure(&dcA, &dcT);
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcT, "dcT");

    PASTE_CODE_ALLOCATE_MATRIX(dcA0, check,
        two_dim_block_cyclic, (&dcA0, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcQ, check,
        two_dim_block_cyclic, (&dcQ, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    /* Check the solution */
    PASTE_CODE_ALLOCATE_MATRIX(dcB, check,
        two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(dcX, check,
        two_dim_block_cyclic, (&dcX, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                               N, NRHS, SMB, SNB, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, 3872);
    if( check )
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcA, (parsec_tiled_matrix_dc_t *)&dcA0 );
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (parsec_tiled_matrix_dc_t *)&dcT);
    if(loud > 3) printf("Done\n");

    /* Create PaRSEC */
    PASTE_CODE_ENQUEUE_KERNEL(parsec, zgeqrf,
                              ((parsec_tiled_matrix_dc_t*)&dcA,
                               (parsec_tiled_matrix_dc_t*)&dcT));

    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(parsec, zgeqrf);
    dplasma_zgeqrf_Destruct( PARSEC_zgeqrf );

    if( check ) {
        if(loud > 2) printf("+++ Generate the Q ...");
        dplasma_zungqr( parsec,
                        (parsec_tiled_matrix_dc_t *)&dcA,
                        (parsec_tiled_matrix_dc_t *)&dcT,
                        (parsec_tiled_matrix_dc_t *)&dcQ);
        if(loud > 2) printf("Done\n");

        if(loud > 2) printf("+++ Solve the system ...");
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcX, 2354);
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcX, (parsec_tiled_matrix_dc_t *)&dcB );
        dplasma_zgeqrs( parsec,
                       (parsec_tiled_matrix_dc_t *)&dcA,
                       (parsec_tiled_matrix_dc_t *)&dcT,
                       (parsec_tiled_matrix_dc_t *)&dcX);
        if(loud > 2) printf("Done\n");

        /* Check the orthogonality, factorization and the solution */
        ret |= check_orthogonality(parsec, (rank == 0) ? loud : 0,
                                   (parsec_tiled_matrix_dc_t *)&dcQ);
        ret |= check_factorization(parsec, (rank == 0) ? loud : 0,
                                   (parsec_tiled_matrix_dc_t *)&dcA0,
                                   (parsec_tiled_matrix_dc_t *)&dcA,
                                   (parsec_tiled_matrix_dc_t *)&dcQ);
        ret |= check_solution(parsec, (rank == 0) ? loud : 0,
                                   (parsec_tiled_matrix_dc_t *)&dcA0,
                                   (parsec_tiled_matrix_dc_t *)&dcB,
                                   (parsec_tiled_matrix_dc_t *)&dcX);

        parsec_data_free(dcA0.mat);
        parsec_data_free(dcQ.mat);
        parsec_data_free(dcB.mat);
        parsec_data_free(dcX.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcQ);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcX);
    }

    two_dim_tabular_destroy( &dcA );
    two_dim_tabular_destroy( &dcT );

    cleanup_parsec(parsec, iparam);

    return ret;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */

static int check_orthogonality(parsec_context_t *parsec, int loud, parsec_tiled_matrix_dc_t *Q)
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

    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (parsec_tiled_matrix_dc_t *)&Id);

    /* Perform Id - Q'Q (could be done with Herk) */
    if ( M >= N ) {
      dplasma_zgemm( parsec, PlasmaConjTrans, PlasmaNoTrans,
                     1.0, Q, Q, -1.0, (parsec_tiled_matrix_dc_t*)&Id );
    } else {
      dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaConjTrans,
                     1.0, Q, Q, -1.0, (parsec_tiled_matrix_dc_t*)&Id );
    }

    normQ = dplasma_zlange(parsec, PlasmaInfNorm, (parsec_tiled_matrix_dc_t*)&Id);

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

    parsec_data_free(Id.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&Id);
    return info_ortho;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */

static int
check_factorization(parsec_context_t *parsec, int loud,
                    parsec_tiled_matrix_dc_t *Aorig,
                    parsec_tiled_matrix_dc_t *A,
                    parsec_tiled_matrix_dc_t *Q)
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)Aorig;
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
    dplasma_zlacpy( parsec, PlasmaUpperLower, Aorig, (parsec_tiled_matrix_dc_t *)&Residual );

    /* Extract the R */
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (parsec_tiled_matrix_dc_t *)&R);
    dplasma_zlacpy( parsec, PlasmaUpper, A, (parsec_tiled_matrix_dc_t *)&R );

    /* Perform Residual = Aorig - Q*R */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans,
                   -1.0, Q, (parsec_tiled_matrix_dc_t *)&R,
                   1.0, (parsec_tiled_matrix_dc_t *)&Residual);

    /* Free R */
    parsec_data_free(R.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&R);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, (parsec_tiled_matrix_dc_t*)&Residual);
    Anorm = dplasma_zlange(parsec, PlasmaInfNorm, Aorig);

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

    parsec_data_free(Residual.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&Residual);
    return info_factorization;
}

static int check_solution( parsec_context_t *parsec, int loud,
                           parsec_tiled_matrix_dc_t *dcA,
                           parsec_tiled_matrix_dc_t *dcB,
                           parsec_tiled_matrix_dc_t *dcX )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int m = dcB->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm = dplasma_zlange(parsec, PlasmaInfNorm, dcA);
    Bnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcB);
    Xnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcX);

    /* Compute b - A*x */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0, dcA, dcX, 1.0, dcB);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcB);

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
