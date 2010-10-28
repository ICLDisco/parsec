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
#if defined(DAGUE_CUDA_SUPPORT) && defined(PRECISION_s) && 0
    iparam[IPARAM_NGPUS] = 0;
#endif
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
    iparam_default_ibnbmb(iparam, 60, 200, 200);

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
    /* In each tile we store IPIV followed by L */
    PASTE_CODE_ALLOCATE_MATRIX(ddescLIPIV, 1, 
        two_dim_block_cyclic, (&ddescLIPIV, matrix_ComplexDouble, 
                                    nodes, cores, rank, IB+1, NB, MT*(IB+1), N, 0, 0, 
                                    MT*(IB+1), N, SMB, SNB, P))

#if defined(DAGUE_PROFILING)
    ddescA.super.super.key = strdup("A");
    ddescLIPIV.super.super.key = strdup("LIPIV");
#endif

    if(!check)
    {
        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);
        generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescLIPIV);
        if(loud > 2) printf("Done\n");
        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, zgetrf_sd,
                                            ((tiled_matrix_desc_t*)&ddescA,
                                             (tiled_matrix_desc_t*)&ddescLIPIV,
                                             &info)) 
        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, zgetrf_sd)
    }
    
    dague_data_free(ddescA.mat);
    dague_data_free(ddescLIPIV.mat);

    cleanup_dague(dague);

#if defined(DAGUE_PROFILING)
    free(ddescA.super.super.key);
    free(ddescLIPIV.super.super.key);
#endif
    return EXIT_SUCCESS;
}

