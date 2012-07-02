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
    double *work = NULL;
    double normlap = 0.0;
    double normdag = 0.0;
    double eps = LAPACKE_dlamch_work('e');
    int iparam[IPARAM_SIZEOF];
    int i, ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    check = 1;
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
    dplasma_zlacpy(dague,
                   PlasmaUpperLower,
                   (tiled_matrix_desc_t *)&ddescA,
                   (tiled_matrix_desc_t *)&ddescA0);
    if(loud > 2) printf("Done\n");

    if( rank == 0 ) {
        work = (double *)malloc( max(M,N) * sizeof(double));
    }

    /* Computing the norm */
    for(i=0; i<4; i++) {
        if ( rank == 0 ) {
            printf("***************************************************\n");
        }
        if(loud > 2) printf("+++ Computing norm %s ... ", normsstr[i]);
        normdag = dplasma_zlange(dague, norms[i],
                                 (tiled_matrix_desc_t *)&ddescA);

        if ( rank == 0 ) {
            normlap = LAPACKE_zlange_work(LAPACK_COL_MAJOR, normsstr[i][0], M, N,
                                          (Dague_Complex64_t*)(ddescA0.mat), ddescA0.super.lm, work);
        }
        if(loud > 2) printf("Done.\n");

        if ( loud > 2 ) {
            printf( "%d: The norm %s of A is %e\n",
                    rank, normsstr[i], normdag);
        }

        if ( rank == 0 ) {
            if ( loud > 2 ) {
                printf( "The LAPACK norm %s of A is %e\n",
                        normsstr[i], normlap);
            }
            normdag = fabs(normdag - normlap) / normlap ;
            if ( normdag < ( 10 * (double)N * eps ) ) {
                printf(" ----- TESTING ZLANGE (%s) ... SUCCESS !\n", normsstr[i]);
            } else {
                printf(" ----- TESTING ZLANGE (%s) ... FAILED !\n", normsstr[i]);
                printf("       | Ndag - Nlap | / Nlap = %e\n", normdag);
                ret |= 1;
            }
        }
    }

    if ( rank == 0 ) {
        printf("***************************************************\n");
        free( work );
    }
    dague_data_free(ddescA0.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA0);

    dague_data_free(ddescA.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);

    cleanup_dague(dague, iparam);

    return ret;
}
