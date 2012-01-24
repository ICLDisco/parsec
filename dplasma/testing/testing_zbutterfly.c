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
#if defined(HAVE_CUDA) && defined(PRECISION_s)
#include "dplasma/cores/cuda_sgemm.h"
#endif

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info = 0;
    int ret = 0;
    int uplo = PlasmaLower;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    /* TODO: compute flops for butterfly */
    PASTE_CODE_FLOPS(FLOPS_ZPOTRF, ((DagDouble_t)N));

    /* initializing matrix structure */
    LDA = max( LDA, N );
    LDB = max( LDB, N );
    SMB = 1;
    SNB = 1;

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
                               sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                                                          nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                                                          N, N, P, uplo));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplghe( dague, (double)(N), uplo, 
                    (tiled_matrix_desc_t *)&ddescA, 1358);
    if(loud > 2) printf("Done\n");

    if(loud > 2) printf("+++ Computing Butterfly ... ");
    PASTE_CODE_ENQUEUE_KERNEL(dague, zbutterfly, 
                              ((tiled_matrix_desc_t*)&ddescA, &info));
    PASTE_CODE_PROGRESS_KERNEL(dague, zbutterfly);

    dplasma_zbutterfly_Destruct( DAGUE_zbutterfly );
    if(loud > 2) printf("Done.\n");

    if ( info != 0 ) {
        if( rank == 0 && loud ) printf("-- The butterfly failed (info = %d) ! \n", info);
        ret |= 1;
    }
    dague_data_free(ddescA.mat);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescA);

    cleanup_dague(dague, iparam);

    return ret;
}

