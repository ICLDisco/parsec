/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#ifndef _DPLASMA_Z_CORES_H
#define _DPLASMA_Z_CORES_H

#include "dague.h"
#include "data_dist/matrix/precision.h"
#include "data_dist/matrix/matrix.h"
#include <core_blas.h>

int blgchase_ztrdv2(int NT, int N, int NB,
                   dague_complex64_t *A1, dague_complex64_t *A2,
                   dague_complex64_t *V1, dague_complex64_t *TAU1,
                   dague_complex64_t *V2, dague_complex64_t *TAU2,
                   int sweep, int id, int blktile);

int CORE_zamax(PLASMA_enum storev, PLASMA_enum uplo, int M, int N,
               const PLASMA_Complex64_t *A, int lda, double *work);
int CORE_zamax_tile( PLASMA_enum storev, PLASMA_enum uplo, const PLASMA_desc descA, double *work);

int dplasma_core_ztradd(PLASMA_enum uplo, PLASMA_enum trans, int M, int N,
                              dague_complex64_t  alpha,
                        const dague_complex64_t *A, int LDA,
                              dague_complex64_t  beta,
                              dague_complex64_t *B, int LDB);

int dplasma_core_zgeadd(PLASMA_enum trans, int M, int N,
                              dague_complex64_t  alpha,
                        const dague_complex64_t *A, int LDA,
                              dague_complex64_t  beta,
                              dague_complex64_t *B, int LDB);

#if defined(DAGUE_HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime_api.h>

int dplasma_cuda_zparfb(PLASMA_enum side, PLASMA_enum trans,
                        PLASMA_enum direct, PLASMA_enum storev,
                        int M1, int N1,
                        int M2, int N2,
                        int K, int L,
                        dague_complex64_t *A1, int LDA1,
                        dague_complex64_t *A2, int LDA2,
                        const dague_complex64_t *V, int LDV,
                        const dague_complex64_t *T, int LDT,
                        dague_complex64_t *WORK, int LDWORK,
                        dague_complex64_t *WORKC, int LDWORKC,
                        cudaStream_t stream);

int dplasma_cuda_ztsmqr( PLASMA_enum side, PLASMA_enum trans,
                         int M1, int N1,
                         int M2, int N2,
                         int K, int IB,
                         dague_complex64_t *A1, int LDA1,
                         dague_complex64_t *A2, int LDA2,
                         const dague_complex64_t *V, int LDV,
                         const dague_complex64_t *T, int LDT,
                         dague_complex64_t *WORK, int LDWORK,
                         dague_complex64_t *WORKC, int LDWORKC,
                         cudaStream_t stream);
#endif /* defined(DAGUE_HAVE_CUDA) */

#endif /* _DPLASMA_Z_CORES_ */
