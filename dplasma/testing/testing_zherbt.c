/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <plasma.h>
#include "common.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

/* Including the bulge chassing */
#define FADDS_ZHERBT(__n) (((__n) * (-8 / 3 + (__n) * (1 + 2 / 3 * (__n)))) - 4)
#define FMULS_ZHERBT(__n) (((__n) * (-1 / 6 + (__n) * (5 / 2 + 2 / 3 * (__n)))) - 15)

int main(int argc, char *argv[])
{
    dague_context_t *dague;
    int iparam[IPARAM_SIZEOF];
    PLASMA_desc *plasmaDescA;
    PLASMA_desc *plasmaDescT;

     /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 48, 144, 144);
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS_COUNT(FADDS_ZHERBT, FMULS_ZHERBT, ((DagDouble_t)N))

    LDA = max(M, LDA);
    LDB = max( LDB, N );
    SMB = 1; SNB = 1;

    if( !check ) {
        PLASMA_Init(1);

        PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
                                   two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                                                          nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                                                          N, N, SMB, SNB, P))
        PLASMA_Desc_Create(&plasmaDescA, ddescA.mat, PlasmaComplexDouble, 
                           ddescA.super.mb, ddescA.super.nb, ddescA.super.bsiz, 
                           ddescA.super.lm, ddescA.super.ln, ddescA.super.i, ddescA.super.j, 
                           ddescA.super.m, ddescA.super.n);
        PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1, 
                                   two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, 
                                                          nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
                                                          MT*IB, N, SMB, SNB, P))
        PLASMA_Desc_Create(&plasmaDescT, ddescT.mat, PlasmaComplexDouble, 
                           ddescT.super.mb, ddescT.super.nb, ddescT.super.bsiz, 
                           ddescT.super.lm, ddescT.super.ln, ddescT.super.i, ddescT.super.j, 
                           ddescT.super.m, ddescT.super.n);
        PLASMA_enum uplo = PlasmaLower;
        generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);
        PASTE_CODE_ENQUEUE_KERNEL(dague, zherbt, 
                                  (uplo, IB, *plasmaDescA, (tiled_matrix_desc_t*)&ddescA, *plasmaDescT, (tiled_matrix_desc_t*)&ddescT));

        PASTE_CODE_PROGRESS_KERNEL(dague, zherbt);

        dplasma_zherbt_Destruct( DAGUE_zherbt );

        dague_data_free(ddescA.mat);
        dague_data_free(ddescT.mat);
    
        cleanup_dague(dague);
        
        dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescT);
        
        return EXIT_SUCCESS;

    } else {
        fprintf(stderr, "Run with checks not coded yet\n");
        /* Initialize with checks:
         *  LAPACK zlatms function that takes as input a set of parameters
         *  And create as output
         *     - corresponding eigenvalues
         *     - a matrix whose eigenvalue is this vector
         *  See testing_zeev.c in Hatem's branch
         *   Look at line 261: call to LAPACKE_zlatms_work
         *     A1 is the matrix
         *     W1 is the eigenvalues for A1
         *
         *  Sort the eigenvalues with dlasrt_ (see same file)
         */
        
        /* Scatter A1 (the remaining in assuming a _L operation) */
        
        /* Call dague_L_zherbt */
        
        /* Gather everything in the lower part of A1.
         * The upper part is irrelevant */
        
        /* In the lower part of A1, we get the Band,
         * and some data similar to the reflector vectors of QR
         * These reflectors must be zeroed.
         * Everything below the band of radius nb+2 inclusive must be zeroed.
         */
        
        /* Call LAPACK_zheev to compute the eigenvalues from the Band
         * See */
        
        /* Compare resulting vector with result W1 of zlatms
         * using the check_solution function of testing_zeev.c */

        cleanup_dague(dague);
    }
}
