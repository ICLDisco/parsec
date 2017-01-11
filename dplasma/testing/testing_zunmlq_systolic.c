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
static int check_orthogonality(parsec_context_t *parsec, int loud, tiled_matrix_desc_t *Q)
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

    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&Id);

    /* Perform Id - Q'Q */
    if ( M >= N ) {
        dplasma_zherk( parsec, PlasmaUpper, PlasmaConjTrans,
                       1.0, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    } else {
        dplasma_zherk( parsec, PlasmaUpper, PlasmaNoTrans,
                       1.0, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    }

    normQ = dplasma_zlanhe(parsec, PlasmaInfNorm, PlasmaUpper, (tiled_matrix_desc_t*)&Id);

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
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&Id);
    return info_ortho;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
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

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    if (N < K) {
        printf("WARNING: N must be greater or equal to K (Set N = K)\n");
        N = K;
        NT = KT;
    }

    LDA = max(N, LDA);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               K, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTS, 1,
        two_dim_block_cyclic, (&ddescTS, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               KT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescTT, 1,
        two_dim_block_cyclic, (&ddescTT, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               KT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescQ, 1,
        two_dim_block_cyclic, (&ddescQ, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               N, N, SMB, SNB, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescA, 3872);
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTS);
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTT);
    if(loud > 3) printf("Done\n");

    dplasma_systolic_init( &qrtree,
                           PlasmaConjTrans, (tiled_matrix_desc_t *)&ddescA,
                           iparam[IPARAM_P],
                           iparam[IPARAM_Q] );

    if(loud > 3) printf("+++ Factorize A ... ");
    dplasma_zgelqf_param( parsec, &qrtree,
                          (tiled_matrix_desc_t*)&ddescA,
                          (tiled_matrix_desc_t*)&ddescTS,
                          (tiled_matrix_desc_t*)&ddescTT );
    if(loud > 3) printf("Done\n");

    if(loud > 3) printf("+++ Generate Q ... ");
    dplasma_zunglq_param( parsec, &qrtree,
                          (tiled_matrix_desc_t *)&ddescA,
                          (tiled_matrix_desc_t *)&ddescTS,
                          (tiled_matrix_desc_t *)&ddescTT,
                          (tiled_matrix_desc_t *)&ddescQ);
    if(loud > 3) printf("Done\n");

    ret |= check_orthogonality( parsec, (rank == 0) ? loud : 0,
                                (tiled_matrix_desc_t *)&ddescQ);
    if (ret)
        return ret;

    for (s=0; s<2; s++) {

        int Cm = (side[s] == PlasmaLeft) ? N : M;
        int Cn = (side[s] == PlasmaLeft) ? M : N;
        LDC = max(LDC, Cm);

        PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
            two_dim_block_cyclic, (&ddescC, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDC, Cn, 0, 0,
                                   Cm, Cn, SMB, SNB, P));
        PASTE_CODE_ALLOCATE_MATRIX(ddescC0, 1,
            two_dim_block_cyclic, (&ddescC0, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDC, Cn, 0, 0,
                                   Cm, Cn, SMB, SNB, P));

        dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescC0, 2354);
        Cnorm = dplasma_zlange(parsec, PlasmaOneNorm, (tiled_matrix_desc_t *)&ddescC0);

        if (Cnorm == 0.)
            Cnorm = 1.;

        for (t=0; t<2; t++) {
#if defined(PRECISION_z) || defined(PRECISION_c)
            if (t==1) t++;
#endif

            dplasma_zlacpy( parsec, PlasmaUpperLower,
                            (tiled_matrix_desc_t *)&ddescC0,
                            (tiled_matrix_desc_t *)&ddescC);

            dplasma_zunmlq_param( parsec,
                                  side[s], trans[t], &qrtree,
                                  (tiled_matrix_desc_t *)&ddescA,
                                  (tiled_matrix_desc_t *)&ddescTS,
                                  (tiled_matrix_desc_t *)&ddescTT,
                                  (tiled_matrix_desc_t *)&ddescC);

            if (side[s] == PlasmaLeft ) {
                dplasma_zgemm( parsec, trans[t], PlasmaNoTrans,
                               -1., (tiled_matrix_desc_t *)&ddescQ,
                                    (tiled_matrix_desc_t *)&ddescC0,
                               1.,  (tiled_matrix_desc_t *)&ddescC);
            } else {
                dplasma_zgemm( parsec, PlasmaNoTrans, trans[t],
                               -1., (tiled_matrix_desc_t *)&ddescC0,
                                    (tiled_matrix_desc_t *)&ddescQ,
                               1.,  (tiled_matrix_desc_t *)&ddescC);
            }

            Rnorm = dplasma_zlange(parsec, PlasmaOneNorm, (tiled_matrix_desc_t *)&ddescC);
            result = Rnorm / ((double)N * Cnorm * eps);

            if (loud && rank == 0) {
                printf("***************************************************\n");
                if ( loud > 3 ) {
                    printf( "-- ||C||_1 = %e, ||R||_1 = %e, ||R||_1 / (N * ||C||_1 * eps) = %e\n",
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
                printf(" ---- TESTING ZUNMLQ (%s, %s) ...%s !\n",
                       sidestr[s], transstr[t], resultstr);
            }
        }

        parsec_data_free(ddescC0.mat);
        tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescC0);
        parsec_data_free(ddescC.mat);
        tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescC);
    }

    dplasma_systolic_finalize( &qrtree );

    parsec_data_free(ddescA.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescA);
    parsec_data_free(ddescTS.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescTS);
    parsec_data_free(ddescTT.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescTT);
    parsec_data_free(ddescQ.mat);
    tiled_matrix_desc_destroy((tiled_matrix_desc_t*)&ddescQ);

    cleanup_parsec(parsec, iparam);

    return ret;
}
