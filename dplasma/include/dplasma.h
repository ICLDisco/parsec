/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _DPLASMA_H_
#define _DPLASMA_H_

#include "dague_config.h"

#define dplasma_error(__func, __msg) fprintf(stderr, "%s: %s\n", (__func), (__msg))

#include "data_dist/matrix/matrix.h"

/* Functions specific to QR */
#include "dplasma_qr_pivgen.h"

#define DPLASMA_FLAT_TREE       0
#define DPLASMA_GREEDY_TREE     1
#define DPLASMA_FIBONACCI_TREE  2
#define DPLASMA_BINARY_TREE     3
#define DPLASMA_GREEDY1P_TREE   4

/*
 * Enum criteria for LU/QR algorithm
 */
enum criteria_e {
    DEFAULT_CRITERIUM    = 0,
    HIGHAM_CRITERIUM     = 1,
    MUMPS_CRITERIUM      = 2,
    LU_ONLY_CRITERIUM    = 3,
    QR_ONLY_CRITERIUM    = 4,
    RANDOM_CRITERIUM     = 5,
    HIGHAM_SUM_CRITERIUM = 6,
    HIGHAM_MAX_CRITERIUM = 7,
    HIGHAM_MOY_CRITERIUM = 8
};

/**
 * No macro with the name max or min is acceptable as there is
 * no way to correctly define them without borderline effects.
 */
static inline int dplasma_imax(int a, int b) { return (a > b) ? a : b; };
static inline int dplasma_imin(int a, int b) { return (a < b) ? a : b; };

/* sqrt function */
#define dplasma_zsqrt csqrt
#define dplasma_csqrt csqrtf
#define dplasma_dsqrt sqrt
#define dplasma_ssqrt sqrtf

#include <core_blas.h>
#include "dplasma/include/dplasma_s.h"
#include "dplasma/include/dplasma_d.h"
#include "dplasma/include/dplasma_c.h"
#include "dplasma/include/dplasma_z.h"

/*
 * Map operations
 */
int dplasma_map(  dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_unary_op_t operator, void *op_args);
int dplasma_map2( dague_context_t *dague, PLASMA_enum uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_binary_op_t operator, void *op_args);

dague_object_t *dplasma_map_New( PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_unary_op_t operator, void *op_args);
dague_object_t *dplasma_map2_New( PLASMA_enum uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_binary_op_t operator, void *op_args);

void dplasma_map_Destruct( dague_object_t *o );
void dplasma_map2_Destruct( dague_object_t *o );

#endif /* _DPLASMA_H_ */
