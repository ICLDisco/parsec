/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */

#ifndef _cuda_ztsmqr_h_
#define _cuda_ztsmqr_h_

#include "dague_config.h"
#include <dague/devices/cuda/dev_cuda.h>
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "dague/class/fifo.h"
#include "data_dist/matrix/matrix.h"

int gpu_kernel_init_ztsmqr( dague_context_t* dague_context );

int gpu_ztsmqr( dague_execution_unit_t* eu_context,
                dague_execution_context_t* this_task,
                int pushout_A1, int pushout_A2,
                PLASMA_enum side, PLASMA_enum trans,
                int M1, int N1, int M2, int N2, int K, int IB,
                int A1m, int A1n, const tiled_matrix_desc_t *descA1, int LDA1,
                int A2m, int A2n, const tiled_matrix_desc_t *descA2, int LDA2,
                int Vm,  int Vn,  const tiled_matrix_desc_t *descV,  int LDV,
                int Tm,  int Tn,  const tiled_matrix_desc_t *descT,  int LDT);

#endif /* _cuda_ztsmqr_h_ */
