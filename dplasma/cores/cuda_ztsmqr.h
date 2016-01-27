/*
 * Copyright (c) 2010-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */

#ifndef _cuda_ztsmqr_h_
#define _cuda_ztsmqr_h_

#include "dague_config.h"
#include "dague/devices/cuda/dev_cuda.h"
#include "dague.h"
#include "dague/execution_unit.h"
#include "dague/class/fifo.h"
#include "data_dist/matrix/matrix.h"

int gpu_ztsmqr( dague_execution_unit_t* eu_context,
                dague_execution_context_t* this_task,
                int pushout_A1, int pushout_A2, int m, int n, int k,
                PLASMA_enum side, PLASMA_enum trans,
                int M1, int N1, int M2, int N2, int K, int IB,
                int LDA1, int LDA2, int LDV, int LDT );

#endif /* _cuda_ztsmqr_h_ */
