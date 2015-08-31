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
#include <plasma.h>

int blgchase_ztrdv2(int NT, int N, int NB,
                   dague_complex64_t *A1, dague_complex64_t *A2,
                   dague_complex64_t *V1, dague_complex64_t *TAU1,
                   dague_complex64_t *V2, dague_complex64_t *TAU2,
                   int sweep, int id, int blktile);

int CORE_zgetrf_rectil_1thrd(const PLASMA_desc A, int *IPIV);
void dplasmacore_zgetrf_rectil_init(void);
int dplasmacore_zgetrf_rectil(volatile dague_complex64_t *amax1buf,
                              const tiled_matrix_desc_t *A, int *IPIV, int *info);

int CORE_zplssq(int M, int N,
                dague_complex64_t *A, int LDA,
                double *scale, double *sumsq);

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

 int CORE_ztsmqr_wei(PLASMA_enum side, PLASMA_enum trans,
                    int M1, int N1, int M2, int N2, int K, int IB,
                    PLASMA_Complex64_t *A1, int LDA1,
                    PLASMA_Complex64_t *A2, int LDA2,
                    const PLASMA_Complex64_t *V, int LDV,
                    const PLASMA_Complex64_t *T, int LDT,
                    PLASMA_Complex64_t *WORK, int LDWORK,
                    PLASMA_Complex64_t *WORKC, int LDWORKC);

int CORE_zparfb_wei(PLASMA_enum side, PLASMA_enum trans, PLASMA_enum direct, PLASMA_enum storev,
                    int M1, int N1, int M2, int N2, int K, int L,
                    PLASMA_Complex64_t *A1, int LDA1,
                    PLASMA_Complex64_t *A2, int LDA2,
                    const PLASMA_Complex64_t *V, int LDV,
                    const PLASMA_Complex64_t *T, int LDT,
                    PLASMA_Complex64_t *WORK, int LDWORK,
                    PLASMA_Complex64_t *WORKC, int LDWORKC);

#endif /* _DPLASMA_Z_CORES_ */
