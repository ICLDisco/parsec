/*
 * Copyright (c) 2019 The Universiy of Tennessee and The Universiy
 *                    of Tennessee Research Foundation. All rights
 *                    reserved.
 */
/* includes parsec headers */
#include "parsec.h"
#include <parsec/data_dist/matrix/two_dim_rectangle_cyclic.h>
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/parsec_config.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/mca_param.h"

/* system and io */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

/* Flops */
#define FLOPS_STENCIL_1D(n) ( (DTYPE)(iter) * (2*(2*R+1)) * (DTYPE)(n) )

/* Type of Kernel in CORE_stencil,
 * If LOOPGEN = 1, then should execute ./loop_gen_1D R before make */ 
#define LOOPGEN 0

/* Datatype */
#define DTYPE double 
#define MY_TYPE parsec_datatype_double_t

/* Define shorthand for indexing a multi-dimensional array */
#define WEIGHT_1D(jj) weight_1D[jj+R]
#define IN(i,j) IN[(j)*lda+i]
#define OUT(i,j) OUT[(j)*lda+i]

/** @brief Macro, defined copy submatrix of S (Source) to submatrix of D (Destination) */
#define MOVE_SUBMATRIX(m, n, S, S_i, S_j, S_lda, D, D_i, D_j, D_lda) \
        for(j = 0; j < n; j++)                                       \
            for(i = 0; i < m; i++)                                   \
                D[(D_j+j)*(D_lda)+D_i+i] = S[(S_j+j)*(S_lda)+S_i+i]; 

/* Global array of weight */
DTYPE * restrict weight_1D;

/* Print matrix: size mb * nb from displacement (disi, disj) */
static inline void PRINT_MATRIX(DTYPE *A, int mb, int nb, int disi, int disj, int lda){
        for(int i = 0; i < mb; i++){
            for(int j = 0; j < nb; j++)
                printf("%lf ", A[(disj+j)*lda+disi+i]);
            printf("\n");
        }
        printf("\n");
}

/* get the rank of its neighbors */
static inline int rank_neighbor(parsec_tiled_matrix_dc_t* descA,
                                int m, int n, int m_max, int n_max){
    if( (m >= 0) && (n >= 0) && (m <= m_max) && (n <= n_max) )
        return descA->super.rank_of(&descA->super, m, n);
    else
        return -999;
}

/**
 * @brief Stencil 1D
 * 
 * @param [inout] dcA: the data, already distributed and allocated
 * @param [in] iter: iterations
 * @param [in] R: radius
 */
int parsec_stencil_1D(parsec_context_t *parsec,
                      parsec_tiled_matrix_dc_t *A,
                      int iter, int R);

/**
 * @brief Init dcA
 * 
 * @param [inout] dcA: the data, already distributed and allocated
 * @param [in] R: radius of ghost region
 */
int parsec_stencil_init_1D(parsec_context_t *parsec,
                        parsec_tiled_matrix_dc_t *dcA, int R);

/**
 * @brief CORE Kernel of Stencil 1D 
 *    
 * @param [out] OUT: output data
 * @param [in] IN: input data
 * @param [in] weight_1D: weight 
 * @param [in] mb: row tile size 
 * @param [in] nb: column tile size 
 * @param [in] lda: lda 
 * @param [in] R: radius of ghost region
 */
void CORE_stencil_1D(DTYPE *restrict OUT, const DTYPE *restrict IN,
                     const DTYPE *restrict weight_1D, const int mb,
                     const int nb, const int lda, const int R);

/**
 * @brief stencil_1D init operator
 *
 * @param [in] es: execution stream
 * @param [in] descA: tiled matrix date descriptor
 * @param [inout] A:  inout data
 * @param [in] uplo: matrix shape
 * @param [in] m: tile row index
 * @param [in] n: tile column index
 * @param [in] args: R 
 */
int stencil_1D_init_ops(parsec_execution_stream_t *es,
                        const parsec_tiled_matrix_dc_t *descA,
                        void *_A, enum matrix_uplo uplo,
                        int m, int n, void *args);
