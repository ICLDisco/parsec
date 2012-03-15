/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "dague.h"

int blgchase_ztrdv2(int NT, int N, int NB,
                   Dague_Complex64_t *A1, Dague_Complex64_t *A2,
                   Dague_Complex64_t *V1, Dague_Complex64_t *TAU1,
                   Dague_Complex64_t *V2, Dague_Complex64_t *TAU2,
                   int sweep, int id, int blktile);

void CORE_zgetrf_sp(int m, int  n,
                    Dague_Complex64_t *A,
                    int  stride,
                    double  criteria,
                    int *nbpivot);
void CORE_zgetrf_sp_rec(int m, int  n,
                        Dague_Complex64_t *A,
                        int  stride,
                        double  criteria,
                        int *nbpivot);

