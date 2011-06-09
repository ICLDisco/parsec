/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#define FMULS_GEQRF(M, N) (((M) > (N)) ? ((N) * ((N) * (  0.5-(1./3.) * (N) + (M)) + (M))) \
                                       : ((M) * ((M) * ( -0.5-(1./3.) * (M) + (N)) + 2.*(N))))
#define FADDS_GEQRF(M, N) (((M) > (N)) ? ((N) * ((N) * (  0.5-(1./3.) * (N) + (M)))) \
                                       : ((M) * ((M) * ( -0.5-(1./3.) * (M) + (N)) + (N))))
#define FMULS_GELQF(__m, __n) FMULS_GEQRF(__m, __n)
#define FADDS_GELQF(__m, __n) FADDS_GEQRF(__m, __n)

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 48, 144, 144);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS_COUNT(FADDS_GELQF, FMULS_GELQF, ((DagDouble_t)M,(DagDouble_t)N))
      
    LDA = max(M, LDA);
    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                                    nodes, cores, rank, MB, NB, M, N, 0, 0, 
                                    LDA, N, SMB, SNB, P))

    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1, 
        two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, 
                                    nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
                                    MT*IB, N, SMB, SNB, P))
#if defined(DAGUE_PROF_TRACE)
    ddescA.super.super.key = strdup("A");
    ddescT.super.super.key = strdup("T");
#endif

    if(!check) 
    {
        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);
        generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescT);
        if(loud > 2) printf("Done\n");

        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, zgelqf, 
                                  ((tiled_matrix_desc_t*)&ddescA,
                                   (tiled_matrix_desc_t*)&ddescT));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, zgelqf);

        dplasma_zgelqf_Destruct( DAGUE_zgelqf );
    }


    dague_data_free(ddescA.mat);
    dague_data_free(ddescT.mat);

    cleanup_dague(dague, iparam);

    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescT);

    return EXIT_SUCCESS;
}

