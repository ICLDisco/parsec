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

#if defined(HAVE_CUDA) && defined(PRECISION_s)
#include "cuda_stsmqr.h"
#endif

#define FMULS_GEQRF(M, N) (((M) > (N)) ? ((N) * ((N) * (  0.5-(1./3.) * (N) + (M)) + (M))) \
                                       : ((M) * ((M) * ( -0.5-(1./3.) * (M) + (N)) + 2.*(N))))
#define FADDS_GEQRF(M, N) (((M) > (N)) ? ((N) * ((N) * (  0.5-(1./3.) * (N) + (M)))) \
                                       : ((M) * ((M) * ( -0.5-(1./3.) * (M) + (N)) + (N))))

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 48, 144, 144);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS_COUNT(FADDS_GEQRF, FMULS_GEQRF, ((DagDouble_t)M,(DagDouble_t)N))
      
    LDA = max(M, LDA);
    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                                    nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                                    M, N, SMB, SNB, P))
    PASTE_CODE_ALLOCATE_MATRIX(ddescTS, 1, 
        two_dim_block_cyclic, (&ddescTS, matrix_ComplexDouble, 
                                    nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
                                    MT*IB, N, SMB, SNB, P))

    PASTE_CODE_ALLOCATE_MATRIX(ddescTT, 1, 
        two_dim_block_cyclic, (&ddescTT, matrix_ComplexDouble, 
                                    nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
                                    MT*IB, N, SMB, SNB, P))
#if defined(DAGUE_PROF_TRACE)
    ddescA.super.super.key = strdup("A");
    ddescTS.super.super.key = strdup("TS");
    ddescTT.super.super.key = strdup("TT");
#endif

    if(!check) 
    {
        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        generate_tiled_random_mat((tiled_matrix_desc_t *) &ddescA, 100);
        generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescTS);
        generate_tiled_zero_mat((tiled_matrix_desc_t *) &ddescTT);

        /* dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, 3872); */
        /* dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTS); */
        /* dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescTT); */
        if(loud > 2) printf("Done\n");

        /* Create DAGuE */
        PASTE_CODE_ENQUEUE_KERNEL(dague, zgeqrf_param, 
                                  ((tiled_matrix_desc_t*)&ddescA,
                                   (tiled_matrix_desc_t*)&ddescTS,
                                   (tiled_matrix_desc_t*)&ddescTT));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, zgeqrf_param);

        /*dplasma_zgeqrf_param_Destruct( DAGUE_zgeqrf_param );*/
    }

    dague_data_free(ddescA.mat);
    dague_data_free(ddescTS.mat);
    dague_data_free(ddescTT.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescTS);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescTT);

    cleanup_dague(dague, iparam);

    return EXIT_SUCCESS;
}

