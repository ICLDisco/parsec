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

static int check_orthogonality(dague_context_t *dague, int loud, tiled_matrix_desc_t *Q);
static int check_factorization(dague_context_t *dague, int loud, tiled_matrix_desc_t *Aorig, tiled_matrix_desc_t *A, tiled_matrix_desc_t *Q);

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info_ortho = 0, info_facto = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 48, 144, 144);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS(FLOPS_ZGEQRF, ((DagDouble_t)M,(DagDouble_t)N))
      
    LDA = max(M, LDA);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTS, 1, 
        two_dim_block_cyclic, (&ddescTS, matrix_ComplexDouble, 
                               nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTT, 1, 
        two_dim_block_cyclic, (&ddescTT, matrix_ComplexDouble, 
                               nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check, 
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, 
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescQ, check, 
        two_dim_block_cyclic, (&ddescQ, matrix_ComplexDouble, 
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                               M, N, SMB, SNB, P));


#if defined(DAGUE_PROF_TRACE)
    ddescA.super.super.key = strdup("A");
    ddescTS.super.super.key = strdup("TS");
    ddescTT.super.super.key = strdup("TT");
    ddescA0.super.super.key = strdup("A0");
    ddescQ.super.super.key = strdup("Q");
#endif

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, 3872);
    if( check )
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescA0 );
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTS);
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTT);
    if(loud > 2) printf("Done\n");
    
    /* Create DAGuE */
    PASTE_CODE_ENQUEUE_KERNEL(dague, zgeqrf_param,
                              (iparam[IPARAM_LOWLVL_TREE], iparam[IPARAM_HIGHLVL_TREE],
                               iparam[IPARAM_QR_TS_SZE], iparam[IPARAM_QR_HLVL_SZE],
                               (tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescTS,
                               (tiled_matrix_desc_t*)&ddescTT));
    
    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(dague, zgeqrf_param);
    dplasma_zgeqrf_param_Destruct( DAGUE_zgeqrf_param );
    
    if( check ) {
        if(loud > 2) printf("+++ Generate the Q ...");
        dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescQ);
        dplasma_zungqr_param( dague, 
                              iparam[IPARAM_LOWLVL_TREE], iparam[IPARAM_HIGHLVL_TREE],
                              iparam[IPARAM_QR_TS_SZE], iparam[IPARAM_QR_HLVL_SZE],
                              (tiled_matrix_desc_t *)&ddescA, 
                              (tiled_matrix_desc_t *)&ddescTS, 
                              (tiled_matrix_desc_t *)&ddescTT, 
                              (tiled_matrix_desc_t *)&ddescQ);
        if(loud > 2) printf("Done\n");
        
        /* Check the orthogonality, factorization and the solution */
        info_ortho = check_orthogonality(dague, (rank == 0) ? loud : 0,
                                         (tiled_matrix_desc_t *)&ddescQ);
        info_facto = check_factorization(dague, (rank == 0) ? loud : 0,
                                         (tiled_matrix_desc_t *)&ddescA0, 
                                         (tiled_matrix_desc_t *)&ddescA, 
                                         (tiled_matrix_desc_t *)&ddescQ);
        
        dague_data_free(ddescA0.mat);
        dague_data_free(ddescQ.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescA0);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescQ);
    }
    
    dague_data_free(ddescA.mat);
    dague_data_free(ddescTS.mat);
    dague_data_free(ddescTT.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescTS);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescTT);
    
    cleanup_dague(dague, iparam);
    
    return info_ortho || info_facto;
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
        two_dim_block_cyclic, (&Id, matrix_ComplexDouble, 
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

    normQ = dplasma_zlange(dague, PlasmaMaxNorm, (tiled_matrix_desc_t*)&Id);

    result = normQ / (minMN * eps);
    if ( loud ) {
        printf("============\n");
        printf("Checking the orthogonality of Q \n");
        printf("||Id-Q'*Q||_oo / (N*eps) = %e \n", result);
    }

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Orthogonality is suspicious ! \n");
        info_ortho=1;
    }
    else {
        if( loud ) printf("-- Orthogonality is CORRECT ! \n");
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
        two_dim_block_cyclic, (&Residual, matrix_ComplexDouble, 
                               A->super.nodes, A->super.cores, twodA->grid.rank, 
                               A->mb, A->nb, M, N, 0, 0, 
                               M, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    PASTE_CODE_ALLOCATE_MATRIX(R, 1, 
        two_dim_block_cyclic, (&R, matrix_ComplexDouble, 
                               A->super.nodes, A->super.cores, twodA->grid.rank, 
                               A->mb, A->nb, N, N, 0, 0, 
                               N, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    /* Extract the L */
    dplasma_zlacpy( dague, PlasmaUpperLower, Aorig, (tiled_matrix_desc_t *)&Residual );
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&R);

    /* Extract the R */
    dplasma_zlacpy( dague, PlasmaUpper, A, (tiled_matrix_desc_t *)&R );
        
    /* Perform Residual = Aorig - Q*R */
    dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, 
                   -1.0, Q, (tiled_matrix_desc_t *)&R, 
                    1.0, (tiled_matrix_desc_t *)&Residual);

    /* Free R */
    dague_data_free(R.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&R);
    
    Rnorm = dplasma_zlange(dague, PlasmaMaxNorm, (tiled_matrix_desc_t*)&Residual);
    Anorm = dplasma_zlange(dague, PlasmaMaxNorm, Aorig);

    result = Rnorm / ( Anorm * minMN * eps);

    if( loud ) {
        printf("============\n");
        printf("Checking the QR Factorization \n");
        printf("-- ||A-QR||_oo/(||A||_oo.N.eps) = %e \n", result );
    }

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else {
        if( loud ) printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    dague_data_free(Residual.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&Residual);
    return info_factorization;
}
