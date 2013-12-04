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
    dague_ddesc_destroy((dague_ddesc_t*)&Id);
    return info_ortho;
}

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    dplasma_qrtree_t qrtree;
    double Cnorm, Rnorm, result;
    double threshold = 10.;
    double eps = LAPACKE_dlamch_work('e');
    char *resultstr;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    int s, t;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 48, 192, 192);
    iparam[IPARAM_SMB] = 1;
    iparam[IPARAM_SNB] = 1;
    iparam[IPARAM_LDA] = -'m';

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    if (M < K) {
        printf("WARNING: M must be greater or equal to K (Set M = K)\n");
        M = K;
        MT = KT;
    }

    LDA = max(M, LDA);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, K, 0, 0,
                               M, K, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTS, 1,
        two_dim_block_cyclic, (&ddescTS, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, IB, NB, MT*IB, K, 0, 0,
                               MT*IB, K, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTT, 1,
        two_dim_block_cyclic, (&ddescTT, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, IB, NB, MT*IB, K, 0, 0,
                               MT*IB, K, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescQ, 1,
        two_dim_block_cyclic, (&ddescQ, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, M, 0, 0,
                               M, M, SMB, SNB, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, 3872);
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTS);
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTT);
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescQ);
    if(loud > 3) printf("Done\n");

    dplasma_hqr_init( &qrtree,
                      (tiled_matrix_desc_t *)&ddescA,
                      iparam[IPARAM_LOWLVL_TREE], iparam[IPARAM_HIGHLVL_TREE],
                      iparam[IPARAM_QR_TS_SZE],   iparam[IPARAM_QR_HLVL_SZE],
                      iparam[IPARAM_QR_DOMINO],   iparam[IPARAM_QR_TSRR] );

    if(loud > 3) printf("+++ Factorize A ... ");
    dplasma_zgeqrf_param( dague, &qrtree,
                          (tiled_matrix_desc_t*)&ddescA,
                          (tiled_matrix_desc_t*)&ddescTS,
                          (tiled_matrix_desc_t*)&ddescTT );
    if(loud > 3) printf("Done\n");

    if(loud > 3) printf("+++ Generate Q ... ");
    dplasma_zungqr_param( dague, &qrtree,
                          (tiled_matrix_desc_t *)&ddescA,
                          (tiled_matrix_desc_t *)&ddescTS,
                          (tiled_matrix_desc_t *)&ddescTT,
                          (tiled_matrix_desc_t *)&ddescQ);
    if(loud > 3) printf("Done\n");

    ret |= check_orthogonality( dague, (rank == 0) ? loud : 0,
                                (tiled_matrix_desc_t *)&ddescQ);
    if (ret)
        return ret;

    for (s=0; s<2; s++) {

        int Cm = (side[s] == PlasmaLeft) ? M : N;
        int Cn = (side[s] == PlasmaLeft) ? N : M;
        LDC = max(LDC, Cm);

        PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
            two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Tile,
                                   nodes, cores, rank, MB, NB, LDC, Cn, 0, 0,
                                   Cm, Cn, SMB, SNB, P));
        PASTE_CODE_ALLOCATE_MATRIX(ddescC0, 1,
            two_dim_block_cyclic, (&ddescC0, matrix_ComplexDouble, matrix_Tile,
                                   nodes, cores, rank, MB, NB, LDC, Cn, 0, 0,
                                   Cm, Cn, SMB, SNB, P));

        dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescC0, 2354);
        Cnorm = dplasma_zlange(dague, PlasmaOneNorm, (tiled_matrix_desc_t *)&ddescC0);

        if (Cnorm == 0.)
            Cnorm = 1.;

        for (t=0; t<2; t++) {
#if defined(PRECISION_z) || defined(PRECISION_c)
            if (t==1) t++;
#endif

            dplasma_zlacpy( dague, PlasmaUpperLower,
                            (tiled_matrix_desc_t *)&ddescC0,
                            (tiled_matrix_desc_t *)&ddescC);

            dplasma_zunmqr_param( dague,
                                  side[s], trans[t], &qrtree,
                                  (tiled_matrix_desc_t *)&ddescA,
                                  (tiled_matrix_desc_t *)&ddescTS,
                                  (tiled_matrix_desc_t *)&ddescTT,
                                  (tiled_matrix_desc_t *)&ddescC);

            if (side[s] == PlasmaLeft ) {
                dplasma_zgemm( dague, trans[t], PlasmaNoTrans,
                               -1., (tiled_matrix_desc_t *)&ddescQ,
                                    (tiled_matrix_desc_t *)&ddescC0,
                               1.,  (tiled_matrix_desc_t *)&ddescC);
            } else {
                dplasma_zgemm( dague, PlasmaNoTrans, trans[t],
                               -1., (tiled_matrix_desc_t *)&ddescC0,
                                    (tiled_matrix_desc_t *)&ddescQ,
                               1.,  (tiled_matrix_desc_t *)&ddescC);
            }

            Rnorm = dplasma_zlange(dague, PlasmaOneNorm, (tiled_matrix_desc_t *)&ddescC);
            result = Rnorm / ((double)M * Cnorm * eps);

            if (loud && rank == 0) {
                printf("***************************************************\n");
                if ( loud > 3 ) {
                    printf( "-- ||C||_1 = %e, ||R||_1 = %e, ||R||_1 / (M * ||C||_1 * eps) = %e\n",
                            Cnorm, Rnorm, result );
                }

                if (  isnan(Rnorm) || isinf(Rnorm) ||
                      isnan(result) || isinf(result) || (result >= threshold) ) {
                    resultstr = " FAILED";
                    ret |= 1;
                }
                else{
                    resultstr = "... PASSED";
                }
                printf(" ---- TESTING ZUNMQR (%s, %s) ...%s !\n",
                       sidestr[s], transstr[t], resultstr);
            }
        }

        dague_data_free(ddescC0.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescC0);
        dague_data_free(ddescC.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescC);
    }

    dplasma_hqr_finalize( &qrtree );

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescTS.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescTS);
    dague_data_free(ddescTT.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescTT);
    dague_data_free(ddescQ.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescQ);

    cleanup_dague(dague, iparam);

    return ret;
}
