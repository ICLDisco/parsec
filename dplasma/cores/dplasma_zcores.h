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

int blgchase_ztrdv2(int NT, int N, int NB,
                   Dague_Complex64_t *A1, Dague_Complex64_t *A2,
                   Dague_Complex64_t *V1, Dague_Complex64_t *TAU1,
                   Dague_Complex64_t *V2, Dague_Complex64_t *TAU2,
                   int sweep, int id, int blktile);

int CORE_zgetrf_rectil_1thrd(const PLASMA_desc A, int *IPIV);
void dplasmacore_zgetrf_rectil_init(void);
int dplasmacore_zgetrf_rectil(volatile Dague_Complex64_t *amax1buf,
                              const tiled_matrix_desc_t *A, int *IPIV, int *info);

int CORE_zplssq(int M, int N,
                Dague_Complex64_t *A, int LDA,
                double *scale, double *sumsq);

#endif /* _DPLASMA_Z_CORES_ */
