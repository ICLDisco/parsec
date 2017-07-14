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

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
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
    iparam[IPARAM_SNB] = 4;
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
    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               K, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcT, 1,
        two_dim_block_cyclic, (&dcT, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, N, 0, 0,
                               KT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcQ, 1,
        two_dim_block_cyclic, (&dcQ, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, N, 0, 0,
                               N, N, SMB, SNB, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, 3872);
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (parsec_tiled_matrix_dc_t *)&dcT);
    if(loud > 3) printf("Done\n");

    if(loud > 3) printf("+++ Factorize A ... ");
    dplasma_zgelqf(parsec,
                   (parsec_tiled_matrix_dc_t*)&dcA,
                   (parsec_tiled_matrix_dc_t*)&dcT);
    if(loud > 3) printf("Done\n");

    if(loud > 3) printf("+++ Generate Q ... ");
    dplasma_zunglq( parsec,
                    (parsec_tiled_matrix_dc_t *)&dcA,
                    (parsec_tiled_matrix_dc_t *)&dcT,
                    (parsec_tiled_matrix_dc_t *)&dcQ);
    if(loud > 3) printf("Done\n");

    for (s=0; s<2; s++) {

        int Cm = (side[s] == PlasmaLeft) ? N : M;
        int Cn = (side[s] == PlasmaLeft) ? M : N;
        LDC = max(LDC, Cm);

        PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
            two_dim_block_cyclic, (&dcC, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDC, Cn, 0, 0,
                                   Cm, Cn, SMB, SNB, P));
        PASTE_CODE_ALLOCATE_MATRIX(dcC0, 1,
            two_dim_block_cyclic, (&dcC0, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDC, Cn, 0, 0,
                                   Cm, Cn, SMB, SNB, P));

        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC0, 2354);
        Cnorm = dplasma_zlange(parsec, PlasmaOneNorm, (parsec_tiled_matrix_dc_t *)&dcC0);

        if (Cnorm == 0.)
            Cnorm = 1.;

        for (t=0; t<2; t++) {
#if defined(PRECISION_z) || defined(PRECISION_c)
            if (t==1) t++;
#endif

            dplasma_zlacpy( parsec, PlasmaUpperLower,
                            (parsec_tiled_matrix_dc_t *)&dcC0,
                            (parsec_tiled_matrix_dc_t *)&dcC);

            dplasma_zunmlq( parsec, side[s], trans[t],
                            (parsec_tiled_matrix_dc_t *)&dcA,
                            (parsec_tiled_matrix_dc_t *)&dcT,
                            (parsec_tiled_matrix_dc_t *)&dcC);

            if (side[s] == PlasmaLeft ) {
                dplasma_zgemm( parsec, trans[t], PlasmaNoTrans,
                               -1., (parsec_tiled_matrix_dc_t *)&dcQ,
                                    (parsec_tiled_matrix_dc_t *)&dcC0,
                               1.,  (parsec_tiled_matrix_dc_t *)&dcC);
            } else {
                dplasma_zgemm( parsec, PlasmaNoTrans, trans[t],
                               -1., (parsec_tiled_matrix_dc_t *)&dcC0,
                                    (parsec_tiled_matrix_dc_t *)&dcQ,
                               1.,  (parsec_tiled_matrix_dc_t *)&dcC);
            }

            Rnorm = dplasma_zlange(parsec, PlasmaOneNorm, (parsec_tiled_matrix_dc_t *)&dcC);
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

        parsec_data_free(dcC0.mat);
        parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcC0);
        parsec_data_free(dcC.mat);
        parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcC);
    }

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcA);
    parsec_data_free(dcT.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcT);
    parsec_data_free(dcQ.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcQ);

    cleanup_parsec(parsec, iparam);

    return ret;
}
