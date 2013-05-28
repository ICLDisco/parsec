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

    dplasma_zlaset( dague, PlasmaUpperLower, 0., (double)minMN, (tiled_matrix_desc_t *)&Id);

    /* Perform Id - Q'Q (could be done with Herk) */
    if ( M >= N ) {
      dplasma_zgemm( dague, PlasmaConjTrans, PlasmaNoTrans,
                     1.0, Q, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    } else {
      dplasma_zgemm( dague, PlasmaNoTrans, PlasmaConjTrans,
                     1.0, Q, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    }

    normQ = dplasma_zlange(dague, PlasmaInfNorm, (tiled_matrix_desc_t*)&Id);

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
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaLower;
    int ret = 0;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    /* initializing matrix structure */
    LDA = max( LDA, M );
    LDB = max( LDB, M );
    SMB = 1;
    SNB = 1;

    /* PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, */
    /*     sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, */
    /*                                nodes, cores, rank, MB, NB, LDA, N, 0, 0, */
    /*                                M, N, P, uplo)); */

    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble,  matrix_Tile,
                               nodes, cores, rank, MB, NB, LDB, N, 0, 0,
                               M, N, 1, 1, P));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    /* dplasma_zplghe( dague, (double)(N), uplo, */
    /*                 (tiled_matrix_desc_t *)&ddescA, 3872); */
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescB, 2354);
    if(loud > 2) printf("Done\n");

    //    dplasma_zprint( dague, uplo, (tiled_matrix_desc_t *)&ddescA );
    // dplasma_zprint( dague, PlasmaUpperLower, (tiled_matrix_desc_t *)&ddescB );
    check_orthogonality( dague, 1,  (tiled_matrix_desc_t *)&ddescB );
    cleanup_dague(dague, iparam);

    dague_data_free(ddescB.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescB );
    /* dague_data_free(ddescA.mat); */
    /* dague_ddesc_destroy( (dague_ddesc_t*)&ddescA); */

    return ret;
}


