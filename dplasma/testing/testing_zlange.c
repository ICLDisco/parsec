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
    double norm = 0.0;
    int iparam[IPARAM_SIZEOF];
    int ret = 0;
    double eps = LAPACKE_dlamch_work('e');

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS(FLOPS_ZGEMV, ((DagDouble_t)M, (DagDouble_t)N))

    LDA = max( LDA, M );

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile,
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check,
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, matrix_Lapack,
                               1, cores, rank, MB, NB, LDA, N, 0, 0,
                               M, N, SMB, SNB, 1));
    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, 7657);
    if(loud > 2) printf("Done\n");

    if(loud > 2) printf("+++ Computing getrf_sp ... ");

    /* Computing the norm */
    norm = dplasma_zlange_inf(dague, PlasmaInfNorm,
                              (tiled_matrix_desc_t *)&ddescA);

    printf("%d: The infini norm of A is %e\n", rank, norm );

    if(check)
    {
        double *work;
        double normlap = 0.0;

        dplasma_zlacpy(dague,
                       PlasmaUpperLower,
                       (tiled_matrix_desc_t *)&ddescA,
                       (tiled_matrix_desc_t *)&ddescA0);

        if( rank == 0 ) {

            work    = (double *)malloc( max(M,N) * sizeof(double));
            normlap = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'i', M, N,
                                          (Dague_Complex64_t*)(ddescA0.mat), ddescA0.super.lm, work);

            printf("The infini Lapacke norm of A is %e\n", normlap );

            normlap = normlap - norm;
            if ( normlap < (N*eps) ) {
                printf( "The solution is correct\n" );
            } else {
                printf( "The solution is bad (%e)\n", norm - normlap );
            }

            free( work );
        }
        dague_data_free(ddescA0.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescA0);
    }

    if(loud > 2) printf("Done.\n");

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);

    cleanup_dague(dague, iparam);

    return EXIT_SUCCESS;
}
