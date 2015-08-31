/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _DPLASMA_H_
#define _DPLASMA_H_

#include "dague.h"

BEGIN_C_DECLS

#define DPLASMA_DEBUG
#if defined(DPLASMA_DEBUG)
#define dplasma_error(__func, __msg) do { fprintf(stderr, "%s: %s\n", (__func), (__msg)); *((int*)0) = 42; } while(0)
#else
#define dplasma_error(__func, __msg) do { fprintf(stderr, "%s: %s\n", (__func), (__msg)); } while(0)
#endif /* defined(DPLASMA_DEBUG) */


#include "data_dist/matrix/matrix.h"

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

/* Functions specific to QR */
#include "dplasma_qr_param.h"

#include "dplasma_s.h"
#include "dplasma_d.h"
#include "dplasma_c.h"
#include "dplasma_z.h"

/*
 * Map operations
 */
int dplasma_map(  dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_unary_op_t op, void *op_args);
int dplasma_map2( dague_context_t *dague, PLASMA_enum uplo, PLASMA_enum trans, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_binary_op_t op, void *op_args);
int dplasma_map2t( dague_context_t *dague, PLASMA_enum uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_binary_op_t op, void *op_args);

dague_handle_t *dplasma_map_New( PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_unary_op_t op, void *op_args);
dague_handle_t *dplasma_map2_New( PLASMA_enum uplo, PLASMA_enum trans, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_binary_op_t op, void *op_args);
dague_handle_t *dplasma_map2t_New( PLASMA_enum uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_binary_op_t op, void *op_args);

void dplasma_map_Destruct( dague_handle_t *o );
void dplasma_map2_Destruct( dague_handle_t *o );
void dplasma_map2t_Destruct( dague_handle_t *o );

END_C_DECLS

#endif /* _DPLASMA_H_ */
