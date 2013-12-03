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
    dague_context_t* dague;
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

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    if (N < K) {
        printf("WARNING: N must be greater or equal to K (Set N = K)\n");
        N = K;
    }

    LDA = max(N, LDA);

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               K, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1,
        two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, IB, NB, MT*IB, N, 0, 0,
                               KT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescQ, 1,
        two_dim_block_cyclic, (&ddescQ, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               N, N, SMB, SNB, P));

    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, 3872);
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescT);
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescQ);
    if(loud > 3) printf("Done\n");

    if(loud > 3) printf("+++ Factorize A ... ");
    dplasma_zgelqf(dague,
                   (tiled_matrix_desc_t*)&ddescA,
                   (tiled_matrix_desc_t*)&ddescT);
    if(loud > 3) printf("Done\n");

    if(loud > 3) printf("+++ Generate Q ... ");
    dplasma_zunglq( dague,
                    (tiled_matrix_desc_t *)&ddescA,
                    (tiled_matrix_desc_t *)&ddescT,
                    (tiled_matrix_desc_t *)&ddescQ);
    if(loud > 3) printf("Done\n");

    for (s=0; s<2; s++) {

        int Cm = (side[s] == PlasmaLeft) ? N : M;
        int Cn = (side[s] == PlasmaLeft) ? M : N;
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

            dplasma_zunmlq( dague, side[s], trans[t],
                            (tiled_matrix_desc_t *)&ddescA,
                            (tiled_matrix_desc_t *)&ddescT,
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

        dague_data_free(ddescC0.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescC0);
        dague_data_free(ddescC.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescC);
    }

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_data_free(ddescT.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescT);
    dague_data_free(ddescQ.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescQ);

    cleanup_dague(dague, iparam);

    return ret;
}
