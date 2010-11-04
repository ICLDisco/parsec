/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.h"
#if defined(HAVE_CUDA) && defined(PRECISION_s)
#include "cuda_sgemm.h"
#endif


#define FMULS_POTRF(N) ((N) * (1.0 / 6.0 * (N) + 0.5) * (N))
#define FADDS_POTRF(N) ((N) * (1.0 / 6.0 * (N)      ) * (N))

#define FMULS_POTRS(N, NRHS) ( (NRHS) * ( (N) * ((N) + 1.) ) )
#define FADDS_POTRS(N, NRHS) ( (NRHS) * ( (N) * ((N) - 1.) ) )


int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PLASMA_enum uplo = PlasmaLower;
    PASTE_CODE_FLOPS_COUNT(FADDS_POTRF, FMULS_POTRF, ((DagDouble_t)N));

    /* initializing matrix structure */
    int info = 0;
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
        sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                                   nodes, cores, rank, MB, NB, N, N, 0, 0, 
                                   LDA, N, P));

    /* load the GPU kernel */
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0)
    {
        if(loud) printf("+++ Load GPU kernel ... ");
        if(0 != spotrf_cuda_init(dague, (tiled_matrix_desc_t *)&ddescA))
        {
            fprintf(stderr, "XXX Unable to load GPU kernel.\n");
            exit(3);
        }
        if(loud) printf("Done\n");
    }
#endif

    if(!check) 
    {
        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);
        if(loud > 2) printf("Done\n");
#if defined(LLT_LL)
        PASTE_CODE_ENQUEUE_KERNEL(dague, zpotrf_ll, 
                                           (uplo, 
                                            (tiled_matrix_desc_t*)&ddescA,
                                            &info));
        PASTE_CODE_PROGRESS_KERNEL(dague, zpotrf_ll);
#else
        PASTE_CODE_ENQUEUE_KERNEL(dague, zpotrf, 
                                           (uplo, 
                                            (tiled_matrix_desc_t*)&ddescA,
                                            &info));
        PASTE_CODE_PROGRESS_KERNEL(dague, zpotrf);
#endif
    }

    /* OLD UGLY CHECK GOES HERE */

#if defined(HAVE_CUDA) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0) 
    {
        spotrf_cuda_fini(dague);
    }
#endif

    dague_data_free(ddescA.mat);

    cleanup_dague(dague);
    dague_ddesc_destroy( (dague_ddesc_t*)&ddescA);

    return EXIT_SUCCESS;
}


#if 0 /* OLD ULGLY CHECK */
    /* Old checking by comparison: will be remove because doesn't check anything */
    if(iparam[IPARAM_CHECK] == 1) {
        char fname[20];
        sprintf(fname , "zposv_r%d", rank );
        printf("writing matrix to file\n");
        data_write((tiled_matrix_desc_t *) &ddescA, fname);
    } 
    else if( iparam[IPARAM_CHECK] == 2 ){
        char fname[20];
        sym_two_dim_block_cyclic_t ddescB;

        sym_two_dim_block_cyclic_init(&ddescB, matrix_ComplexDouble, nodes, cores, rank,
                                      MB, NB, N, N, 0, 0, LDA, N, P);
        ddescB.mat = dague_data_allocate((size_t)ddescB.super.nb_local_tiles * (size_t)ddescB.super.bsiz * (size_t)ddescB.super.mtype);

        sprintf(fname , "zposv_r%d", rank );
        printf("reading matrix from file\n");
        data_read((tiled_matrix_desc_t *) &ddescB, fname);
        
        matrix_zcompare_dist_data((tiled_matrix_desc_t *) &ddescA, (tiled_matrix_desc_t *) &ddescB);

        dague_data_free(ddescB.mat);
        dague_ddesc_destroy( (dague_ddesc_t*)&ddescB);
#endif

 
