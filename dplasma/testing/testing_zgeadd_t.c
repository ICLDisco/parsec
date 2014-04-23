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
#if defined(HAVE_CUDA)
#include "dplasma/cores/cuda_zgemm.h"
#endif

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaLower;
    int info = 0;
    int ret = 0;

    int Aseed = 100;
    int Bseed = 100;
    int Cseed = 2873;
    
    dague_complex64_t alpha =  1;
    dague_complex64_t beta  = 1;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
    beta  += I * 0.21;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 0, 180, 180);
#if defined(HAVE_CUDA)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    double flops = M*N;
    double gflops = M*N/10e9;

    /* initializing matrix structure */
    LDA = dplasma_imax( LDA, N );
    LDB = dplasma_imax( LDB, M );

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, matrix_Tile, 
                                   nodes, rank, MB, NB, LDA, M, 0, 0,
                                   N, M, SMB, SNB, P));
 
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        two_dim_block_cyclic, (&ddescB, matrix_ComplexDouble, matrix_Tile, 
                                   nodes, rank, MB, NB, LDB, N, 0, 0,
                                   M, N, SMB, SNB, P));
   
    PASTE_CODE_ALLOCATE_MATRIX(ddescB2, 1,
        two_dim_block_cyclic, (&ddescB2, matrix_ComplexDouble, matrix_Tile, 
                                   nodes, rank, MB, NB, LDB, N, 0, 0,
                                   M, N, SMB, SNB, P));
    /* matrix generation */
    if(loud > 3) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescA, Aseed);
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB, Bseed);
    dplasma_zplrnt( dague, 0, (tiled_matrix_desc_t *)&ddescB2, Bseed);
    dplasma_zprint( dague, 
                    PlasmaUpperLower,
                    (tiled_matrix_desc_t*)&ddescA );
    dplasma_zprint( dague, 
                    PlasmaUpperLower,
                    (tiled_matrix_desc_t*)&ddescB );
    /* 
    dplasma_zlacpy( dague, PlasmaLower,
                    (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescA2 );
    */
    
    if(loud > 3) printf("Done\n");

    PASTE_CODE_ENQUEUE_KERNEL(dague, zgeadd_t,
                              (PlasmaUpperLower, alpha, (tiled_matrix_desc_t*)&ddescA, (tiled_matrix_desc_t*)&ddescB));
    PASTE_CODE_PROGRESS_KERNEL(dague, zgeadd_t);


    dplasma_zgeadd_t_Destruct( DAGUE_zgeadd_t );
    dague_handle_sync_ids(); /* recursive DAGs are not synchronous on ids */
    dplasma_zprint( dague, 
                    PlasmaUpperLower,
                    (tiled_matrix_desc_t*)&ddescB );


    dague_complex64_t *matrix_i = (dague_complex64_t*)malloc(N*N*sizeof(dague_complex64_t));
    for(int ii=0;ii<N;ii++){
        for(int jj=0;jj<N;jj++){
            if(ii==jj){
                matrix_i[ii+jj*N] = 1;
            }else{
                matrix_i[ii+jj*N] = 0;
            }
        }
    }

    dague_complex64_t *matrix_A = (dague_complex64_t*)malloc(N*M*sizeof(dague_complex64_t));
    dague_complex64_t *matrix_B = (dague_complex64_t*)malloc(M*N*sizeof(dague_complex64_t));
    dague_complex64_t *matrix_B2 = (dague_complex64_t*)malloc(M*N*sizeof(dague_complex64_t));
    
    twoDBC_ztolapack((two_dim_block_cyclic_t*)(&ddescA), matrix_A, N);
    twoDBC_ztolapack((two_dim_block_cyclic_t*)(&ddescB), matrix_B, M);
    twoDBC_ztolapack((two_dim_block_cyclic_t*)(&ddescB2), matrix_B2, M);

    CORE_zgemm(PlasmaTrans, PlasmaNoTrans,
               M, N, N,
                alpha, matrix_A, N,
                matrix_i, N, 
                beta, matrix_B2, M);

     for(int ii=0;ii<M;ii++){
        for(int jj=0;jj<N;jj++){
            printf("%f\t", matrix_B[ii+jj*M] - matrix_B2[ii+jj*M]);
        }
        printf("\n");
    }


    


    dague_data_free(ddescA.mat); ddescA.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescA);

    dague_data_free(ddescB.mat); ddescB.mat = NULL;
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&ddescB);
    
    cleanup_dague(dague, iparam);
    return ret;
}

