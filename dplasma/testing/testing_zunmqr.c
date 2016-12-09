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
    iparam[IPARAM_SMB] = 4;
    iparam[IPARAM_SNB] = 1;
    iparam[IPARAM_LDA] = -'m';

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
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
                               nodes, rank, MB, NB, LDA, K, 0, 0,
                               M, K, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1,
        two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, IB, NB, MT*IB, K, 0, 0,
                               MT*IB, K, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescQ, 1,
        two_dim_block_cyclic, (&ddescQ, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDA, M, 0, 0,
                               M, M, SMB, SNB, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( parsec, 0, (tiled_matrix_desc_t *)&ddescA, 3872);
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescT);
    if(loud > 3) printf("Done\n");

    if(loud > 3) printf("+++ Factorize A ... ");
    dplasma_zgeqrf(parsec,
                   (tiled_matrix_desc_t*)&ddescA,
                   (tiled_matrix_desc_t*)&ddescT);
    if(loud > 3) printf("Done\n");

    if(loud > 3) printf("+++ Generate Q ... ");
    dplasma_zungqr( parsec,
                    (tiled_matrix_desc_t *)&ddescA,
                    (tiled_matrix_desc_t *)&ddescT,
                    (tiled_matrix_desc_t *)&ddescQ);
    if(loud > 3) printf("Done\n");

    for (s=0; s<2; s++) {

        int Cm = (side[s] == PlasmaLeft) ? M : N;
        int Cn = (side[s] == PlasmaLeft) ? N : M;
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

            dplasma_zunmqr( parsec, side[s], trans[t],
                            (tiled_matrix_desc_t *)&ddescA,
                            (tiled_matrix_desc_t *)&ddescT,
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

        parsec_data_free(ddescC0.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC0);
        parsec_data_free(ddescC.mat);
        tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescC);
    }

    parsec_data_free(ddescA.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);
    parsec_data_free(ddescT.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescT);
    parsec_data_free(ddescQ.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescQ);

    cleanup_parsec(parsec, iparam);

    return ret;
}
