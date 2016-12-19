/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#ifndef _DPLASMA_CORES_H_
#define _DPLASMA_CORES_H_

#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include "parsec.h"
#include "data_dist/matrix/precision.h"
#include "data_dist/matrix/matrix.h"
#include <core_blas.h>

#define PLASMA_BLKADDR(A, type, m, n)  (type *)plasma_getaddr(A, m, n)
#define PLASMA_BLKLDD(A, k) ( ( (k) + (A).i/(A).mb) < (A).lm1 ? (A).mb : (A).lm%(A).mb )

#endif /* _DPLASMA_CORES_H_ */
