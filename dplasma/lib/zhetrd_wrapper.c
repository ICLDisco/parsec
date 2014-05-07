/*
 * Copyright (c) 2014      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dplasma/lib/memory_pool.h"
#include "data_dist/matrix/diag_band_to_rect.h"
#include "dplasma/lib/zhetrd_h2b_L.h"
#include "dplasma/lib/zhetrd_b2s.h"


/* SUBROUTINE ZHETRD( UPLO, N, A, LDA, D, E, TAU, WORK, LWORK, INFO )
 */
int
dplasma_zhetrd( dague_context_t* dague,
                PLASMA_enum uplo,
                int ib,
                tiled_matrix_desc_t* A,
                tiled_matrix_desc_t* DE,
                tiled_matrix_desc_t* T,
                int* info )
{
    dague_zhetrd_h2b_L_object_t * h2b_obj=NULL;
    dague_diag_band_to_rect_object_t* band2rect_obj=NULL;
    dague_zhetrd_b2s_object_t * b2s_obj=NULL;
    dague_memory_pool_t pool[4];

    if( uplo != PlasmaLower && uplo != PlasmaUpper ) {
        dplasma_error("DPLASMA_zhetrd", "illegal value of uplo");
        *info = -1;
        return *info;
    }
    
    dague_private_memory_init( &pool[0], (sizeof(dague_complex64_t)*T->nb) ); /* tau */
    dague_private_memory_init( &pool[1], (sizeof(dague_complex64_t)*T->nb*ib) ); /* work */
    dague_private_memory_init( &pool[2], (sizeof(dague_complex64_t)*T->nb*2 *T->nb) ); /* work for HERFB1 */
    dague_private_memory_init( &pool[3], (sizeof(dague_complex64_t)*T->nb*4 *T->nb) ); /* work for the TSMQRLR */
    
    if( PlasmaLower == uplo ) {
        h2b_obj = dague_zhetrd_h2b_L_new( ib, A, *A, T, *T, &pool[3], &pool[2], &pool[1], &pool[0] );
        dplasma_add2arena_rectangle( h2b_obj->arenas[DAGUE_zhetrd_h2b_L_DEFAULT_ARENA],
                                 A->mb*A->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, A->mb, A->nb, -1);
        dplasma_add2arena_rectangle( h2b_obj->arenas[DAGUE_zhetrd_h2b_L_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1);
#if 0
    } else {
        h2b_obj = dague_zhetrd_h2b_U_new( ib, A, *A, T, *T, pool[3], pool[2], pool[1], pool[0] );
        dplasma_add2arena_rectangle( h2b_obj->arenas[DAGUE_zhetrd_h2b_U_DEFAULT_ARENA],
                                 A->mb*A->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, A->mb, A->nb, -1);
        dplasma_add2arena_rectangle( h2b_obj->arenas[DAGUE_zhetrd_h2b_U_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1);
#endif
    }
    if( NULL == h2b_obj ) { 
        *info=-101; goto cleanup;
    }

    band2rect_obj = dague_diag_band_to_rect_new((sym_two_dim_block_cyclic_t*)A, (two_dim_block_cyclic_t*)DE,
                                                A->mt, A->nt, A->mb, A->nb, sizeof(dague_complex64_t));
    if( NULL == band2rect_obj ) goto cleanup;
    dplasma_add2arena_tile(band2rect_obj->arenas[DAGUE_diag_band_to_rect_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    b2s_obj = dague_zhetrd_b2s_new( DE, DE->mb-1 );
    if( NULL == b2s_obj ) goto cleanup;
    dplasma_add2arena_rectangle(b2s_obj->arenas[DAGUE_zhetrd_b2s_DEFAULT_ARENA], 
                                DE->mb*DE->nb*sizeof(dague_complex64_t),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_DOUBLE_COMPLEX, DE->mb, DE->nb, -1);
        
    dague_enqueue( dague, (dague_object_t*)h2b_obj );
    dague_enqueue( dague, (dague_object_t*)band2rect_obj );
    dague_enqueue( dague, (dague_object_t*)b2s_obj );
    dplasma_progress(dague);

cleanup:
    if( h2b_obj ) DAGUE_INTERNAL_OBJECT_DESTRUCT( h2b_obj );
    if( band2rect_obj ) DAGUE_INTERNAL_OBJECT_DESTRUCT( band2rect_obj );
    if( b2s_obj ) DAGUE_INTERNAL_OBJECT_DESTRUCT( b2s_obj );
    dague_private_memory_fini( &pool[0] );
    dague_private_memory_fini( &pool[1] );
    dague_private_memory_fini( &pool[2] );
    dague_private_memory_fini( &pool[3] );
    return *info; 
}


