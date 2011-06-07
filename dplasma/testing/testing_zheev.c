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
#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/generated/diag_band_to_rect.h"
#include "dplasmatypes.h"

/* Including the bulge chassing */
#define FADDS_ZHEEV(__n) (((__n) * (-8.0 / 3.0 + (__n) * (1.0 + 2.0 / 3.0 * (__n)))) - 4.0)
#define FMULS_ZHEEV(__n) (((__n) * (-1.0 / 6.0 + (__n) * (5.0 / 2.0 + 2.0 / 3.0 * (__n)))) - 15.0)


int main(int argc, char *argv[])
{
    dague_context_t *dague;
    int iparam[IPARAM_SIZEOF];
    PLASMA_enum uplo = PlasmaLower;
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
    
    PASTE_CODE_FLOPS_COUNT(FADDS_ZHEEV, FMULS_ZHEEV, ((DagDouble_t)N))


    PLASMA_Init(1);

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
         sym_two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
         nodes, cores, rank, MB, NB, M, N, 0, 0, 
         N, N, P, uplo))
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1, 
         sym_two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, 
         nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
         MT*IB, N, P, uplo))
    PASTE_CODE_ALLOCATE_MATRIX(ddescBAND, 1, 
        two_dim_block_cyclic, (&ddescBAND, matrix_ComplexDouble,
        nodes, cores, rank, MB+1, NB+2, MB+1, (NB+2)*NT, 0, 0, 
        MB+1, (NB+2)*NT, 1, SNB, 1 /* 1D cyclic */ ));
/*
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
         two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
         nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
         N, N, 1, 1, P))
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1, 
         two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, 
         nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
         MT*IB, N, 1, 1, P))
*/
    PLASMA_Desc_Create(&plasmaDescA, ddescA.mat, PlasmaComplexDouble, 
         ddescA.super.mb, ddescA.super.nb, ddescA.super.bsiz, 
         ddescA.super.lm, ddescA.super.ln, ddescA.super.i, ddescA.super.j, 
         ddescA.super.m, ddescA.super.n);
    PLASMA_Desc_Create(&plasmaDescT, ddescT.mat, PlasmaComplexDouble, 
         ddescT.super.mb, ddescT.super.nb, ddescT.super.bsiz, 
         ddescT.super.lm, ddescT.super.ln, ddescT.super.i, ddescT.super.j, 
         ddescT.super.m, ddescT.super.n);

    generate_tiled_random_sym_pos_mat((tiled_matrix_desc_t *) &ddescA, 100);

    PASTE_CODE_ENQUEUE_KERNEL(dague, zherbt, 
         (uplo, IB, *plasmaDescA, (tiled_matrix_desc_t*)&ddescA, *plasmaDescT, (tiled_matrix_desc_t*)&ddescT));
    PASTE_CODE_PROGRESS_KERNEL(dague, zherbt);
    
    SYNC_TIME_START();
    dague_object_t* DAGUE_diag_band_to_rect = (dague_object_t*) dague_diag_band_to_rect_new((sym_two_dim_block_cyclic_t*)&ddescA, &ddescBAND, 
            MT, NT, MB, NB, sizeof(matrix_ComplexDouble));
    dague_arena_t* arena = ((dague_diag_band_to_rect_object_t*)DAGUE_diag_band_to_rect)->arenas[DAGUE_diag_band_to_rect_DEFAULT_ARENA]; 
    dplasma_add2arena_tile(arena,
        MB*NB*sizeof(Dague_Complex64_t),
        DAGUE_ARENA_ALIGNMENT_SSE,
        MPI_DOUBLE_COMPLEX, MB);
    dague_enqueue(dague, DAGUE_diag_band_to_rect);
    dague_progress(dague);
    SYNC_TIME_PRINT(rank, ( "diag_band_to_rect N= %d NB = %d : %f s\n", N, NB, sync_time_elapsed));

    PASTE_CODE_ENQUEUE_KERNEL(dague, zhbrdt, ((tiled_matrix_desc_t*)&ddescBAND));
    PASTE_CODE_PROGRESS_KERNEL(dague, zhbrdt)



    dplasma_zherbt_Destruct( DAGUE_zherbt );
    dague_diag_band_to_rect_destroy( DAGUE_diag_band_to_rect );
    dplasma_zhbrdt_Destruct( DAGUE_zhbrdt );

    
    dague_data_free(ddescA.mat);
    dague_data_free(ddescT.mat);
    dague_data_free(ddescBAND.mat);

    cleanup_dague(dague, &iparam);
   
    dague_ddesc_destroy((dague_ddesc_t*)&ddescBAND);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescT);
        
    return EXIT_SUCCESS;
}

