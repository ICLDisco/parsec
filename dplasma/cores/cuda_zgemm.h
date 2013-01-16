/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */

#ifndef _cuda_zgemm_h_
#define _cuda_zgemm_h_

#include "dague_config.h"
#include <dague/devices/cuda/dev_cuda.h>
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "fifo.h"
#include "data_dist/matrix/matrix.h"

#define GEMM_KEY( _desc_, _M_, _N_) (uint32_t)(NULL == (_desc_) ? \
                                               0 : (_M_) * (((tiled_matrix_desc_t*)(_desc_))->lmt) + (_N_))

int gpu_kernel_init_zgemm( dague_context_t* dague_context );

int gpu_zgemm( dague_execution_unit_t* eu_context,
               dague_execution_context_t* this_task,
               int pushout,
               PLASMA_enum transA, PLASMA_enum transB,
               int M, int N, int K, 
               dague_complex64_t alpha, int Am, int An, const tiled_matrix_desc_t *descA, int lda,
                                        int Bm, int Bn, const tiled_matrix_desc_t *descB, int ldb,
               dague_complex64_t beta,  int Cm, int Cn, const tiled_matrix_desc_t *descC, int ldc );

#endif /* _cuda_zgemm_h_ */
