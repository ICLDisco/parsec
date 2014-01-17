/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "flops.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaLower;
    int i, info, ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    /* initializing matrix structure */
    LDA = max( LDA, M );
    LDB = max( LDB, M );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble,
                                   nodes, rank, MB, NB, LDA, N, 0, 0,
                                   M, N, P, uplo));

    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble,  matrix_Tile,
                               nodes, rank, MB, NB, LDB, N, 0, 0,
                               M, N, 1, 1, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    ret |= dplasma_zplghe( dague, (double)(N), uplo,
                    (tiled_matrix_desc_t *)&ddescA, 3872);
    ret |= dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, 2354);
    if(loud > 2) printf("Done\n");

    ret |= dplasma_zprint( dague, uplo,             (tiled_matrix_desc_t *)&ddescA );
    ret |= dplasma_zprint( dague, PlasmaUpperLower, (tiled_matrix_desc_t *)&ddescB );

    for(i=0; i<43; i++) {
        if ( rank == 0 ) {
            fprintf(stdout, "====== Generate Test Matrix %d ======\n", i);
            fflush(stdout);
        }
        info = dplasma_zpltmg( dague, i, (tiled_matrix_desc_t *)&ddescB, 5373 );
        if (info == 0)
            ret |= dplasma_zprint( dague, PlasmaUpperLower, (tiled_matrix_desc_t *)&ddescB );
    }

    dague_data_free(ddescB.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB );
    dague_data_free(ddescA.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);

    cleanup_dague(dague, iparam);

    return ret;
}
