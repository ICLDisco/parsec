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
    DEFAULT_CRITERIUM = 0,
    HIGHAM_CRITERIUM  = 1,
    MUMPS_CRITERIUM   = 2,
    LU_ONLY_CRITERIUM = 3,
    QR_ONLY_CRITERIUM = 4,
    RANDOM_CRITERIUM = 5
};

/*
 * Type of matrices that can be generated with zplrnt_perso
 */
enum matrix_init_e {
    MATRIX_RANDOM    = 0,
    MATRIX_HADAMARD  = 1,
    MATRIX_HOUSE     = 2,
    MATRIX_PARTER    = 3,
    MATRIX_RIS       = 4,
    MATRIX_KMS       = 5,
    MATRIX_TOEPPEN   = 6,   /* Unavailable */
    MATRIX_CONDEX    = 7,
    MATRIX_MOLER     = 8,   /* Unavailable */
    MATRIX_CIRCUL    = 9,
    MATRIX_RANDCORR  = 10,  /* Unavailable */
    MATRIX_POISSON   = 11,  /* Unavailable */
    MATRIX_HANKEL    = 12,
    MATRIX_JORDBLOC  = 13,  /* Unavailable */
    MATRIX_COMPAN    = 14,
    MATRIX_PEI       = 15,  /* Unavailable */
    MATRIX_RANDCOLU  = 16,  /* Unavailable */
    MATRIX_SPRANDN   = 17,  /* Unavailable */
    MATRIX_RIEMANN   = 18,
    MATRIX_COMPAR    = 19,  /* Unavailable */
    MATRIX_TRIDIAG   = 20,  /* Unavailable */
    MATRIX_CHEBSPEC  = 21,  /* Unavailable */
    MATRIX_LEHMER    = 22,
    MATRIX_TOEPPD    = 23,
    MATRIX_MINIJ     = 24,
    MATRIX_RANDSVD   = 25,  /* Unavailable */
    MATRIX_FORSYTHE  = 26,  /* Unavailable */
    MATRIX_FIEDLER   = 27,
    MATRIX_DORR      = 28,
    MATRIX_DEMMEL    = 29,
    MATRIX_CHEBVAND  = 30,
    MATRIX_INVHESS   = 31,
    MATRIX_PROLATE   = 32,
    MATRIX_FRANK     = 33,  /* Unavailable */
    MATRIX_CAUCHY    = 34,
    MATRIX_HILB      = 35,
    MATRIX_LOTKIN    = 36,
    MATRIX_KAHAN     = 37,
    MATRIX_ORTHOGO   = 38,
    MATRIX_WILKINSON = 39,
    MATRIX_FOSTER    = 40,
    MATRIX_WRIGHT    = 41,
};

/*
 * Map operations
 */
void dplasma_map2( dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, dague_operator_t operator, void *op_args);
dague_object_t* dplasma_map2_New( PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, dague_operator_t operator, void *op_args);
void dplasma_map2_Destruct( dague_object_t *o );

/**
 * No macro with the name max or min is acceptable as there is
 * no way to correctly define them without borderline effects.
 */
#undef imax
#undef imin
static inline int imax(int32_t a, int32_t b) { return a > b ? a : b; }
static inline int imin(int32_t a, int32_t b) { return a < b ? a : b; }

/* sqrt function */
#define dplasma_zsqrt csqrt
#define dplasma_csqrt csqrtf
#define dplasma_dsqrt sqrt
#define dplasma_ssqrt sqrtf

#include "dplasma/include/dplasma_s.h"
#include "dplasma/include/dplasma_d.h"
#include "dplasma/include/dplasma_c.h"
#include "dplasma/include/dplasma_z.h"

#endif /* _DPLASMA_H_ */
