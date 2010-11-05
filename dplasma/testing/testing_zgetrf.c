/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

#define FMULS_GETRF(M, N) (0.5 * (N) * ((N) * ((M) - (1./3.) * (N)) - (N)))
#define FADDS_GETRF(M, N) (0.5 * (N) * ((N) * ((M) - (1./3.) * (N))))

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    
    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 60, 200, 200);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS_COUNT(FADDS_GETRF, FMULS_GETRF, ((DagDouble_t)M,(DagDouble_t)N))
    /* initializing matrix structure */
    int info = 0;
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                                    nodes, cores, rank, MB, NB, M, N, 0, 0, 
                                    LDA, N, SMB, SNB, P))
    PASTE_CODE_ALLOCATE_MATRIX(ddescL, 1, 
        two_dim_block_cyclic, (&ddescL, matrix_ComplexDouble, 
                                    nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
                                    MT*IB, N, SMB, SNB, P))
    PASTE_CODE_ALLOCATE_MATRIX(ddescIPIV, 1, 
        two_dim_block_cyclic, (&ddescIPIV, matrix_Integer, 
                                    nodes, cores, rank, MB, 1, M, NT, 0, 0, 
                                    M, NT, SMB, SNB, P))

#if defined(DAGUE_PROF_TRACE)
    ddescA.super.super.key = strdup("A");
    ddescL.super.super.key = strdup("L");
    ddescIPIV.super.super.key = strdup("IPIV");
#endif

    if(!check)
    {
        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);
        generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescIPIV);
        if(loud > 2) printf("Done\n");
        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, zgetrf,
                                            ((tiled_matrix_desc_t*)&ddescA,
                                             (tiled_matrix_desc_t*)&ddescL,
                                             (tiled_matrix_desc_t*)&ddescIPIV,
                                             &info)) 
        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, zgetrf)
    }
    
    dague_data_free(ddescA.mat);
    dague_data_free(ddescL.mat);
    dague_data_free(ddescIPIV.mat);

    cleanup_dague(dague);

    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescL);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescIPIV);

    return EXIT_SUCCESS;
}
