/*
 * Copyright (c) 2010-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 */
#ifndef _DPLASMA_ZRECURSIVE_H_
#define _DPLASMA_ZRECURSIVE_H_

#include "data_dist/matrix/subtile.h"
#include "dplasma/lib/memory_pool.h"

dague_handle_t* dplasma_zgeqrfr_geqrt_New(tiled_matrix_desc_t *A,  tiled_matrix_desc_t *T,  dague_memory_pool_t *work);
dague_handle_t* dplasma_zgeqrfr_tsqrt_New(tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2, tiled_matrix_desc_t *T, dague_memory_pool_t *tau, dague_memory_pool_t *work);
dague_handle_t* dplasma_zgeqrfr_unmqr_New(tiled_matrix_desc_t *A,  tiled_matrix_desc_t *T,  tiled_matrix_desc_t *B, dague_memory_pool_t *work);
dague_handle_t* dplasma_zgeqrfr_tsmqr_New(tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2, tiled_matrix_desc_t *V, tiled_matrix_desc_t *T, dague_memory_pool_t *work);

void dplasma_zgeqrfr_geqrt_Destruct( dague_handle_t *o );
void dplasma_zgeqrfr_tsqrt_Destruct( dague_handle_t *o );
void dplasma_zgeqrfr_unmqr_Destruct( dague_handle_t *o );
void dplasma_zgeqrfr_tsmqr_Destruct( dague_handle_t *o );

#endif /* _DPLASMA_ZRECURSIVE_H_ */
