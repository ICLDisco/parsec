/*
 * Copyright (c) 2017-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/data_dist/matrix/redistribute/redistribute_internal.h"

/* Define whether run PTG or DTD */
#define RUN_PTG 1
#define RUN_DTD 1

/* Print more info */
#define PRINT_MORE 1

/* Copy to a node to check */
#define COPY_TO_1NODE 0

/**
 * @brief Calculate time bound of communication
 *
 * @param [in] dcY: source distribution, already distributed and allocated
 * @param [in] dcT: target distribution, redistributed and allocated
 * @param [in] size_row: row size of submatrix
 * @param [in] size_col: column size of submatrix
 * @param [in] disi_Y: row displacement in dcY
 * @param [in] disj_Y: column displacement in dcY
 * @param [out]: Time Bound
 */
double* parsec_redistribute_bound(parsec_context_t *parsec,
                               parsec_tiled_matrix_t *dcY,
                               parsec_tiled_matrix_t *dcT,
                               int size_row, int size_col,
                               int disi_Y, int disj_Y,
                               int disi_T, int disj_T);

/**
 * @brief Check result: copy to a single node
 *
 * @details
 * Check whether value of submatrix in dcY and in dcT are
 * the same, not including the ghost region.
 *
 * @param [in] dcY: the data, already distributed and allocated
 * @param [in] dcT: the data, already distributed and allocated
 * @param [in] size_row: row size of submatrix to be checked
 * @param [in] size_col: column size of submatrix to be checked
 * @param [in] disi_Y: row displacement of submatrix in Y
 * @param [in] disj_Y: column displacement of submatrix in Y
 * @param [in] disi_T: row displacement of submatrix in T
 * @param [in] disj_T: column displacement of submatrix in T
 * @return 0, if they are the same, and print "Redistribute
 * Result is CORRECT", otherwise print the first detected
 * location and values where values are different.
 */
int parsec_redistribute_check(parsec_context_t *parsec,
                              parsec_tiled_matrix_t *dcY,
                              parsec_tiled_matrix_t *dcT,
                              int size_row, int size_col,
                              int disi_Y, int disj_Y,
                              int disi_T, int disj_T);

/**
 * @brief Check result 2: init matrix to specail value
 * @details
 * Check whether value of submatrix in dcY correct
 *
 * @param [in] dcY: the data, already distributed and allocated
 * @param [in] size_row: row size of submatrix to be checked
 * @param [in] size_col: column size of submatrix to be checked
 * @param [in] disi_Y: row displacement of submatrix in Y
 * @param [in] disj_Y: column displacement of submatrix in Y
 * @return 0, if CORRECT, and print "Redistribute Result is CORRECT",
 * otherwise Number of data not correct, and print "Redistribute Result
 * "is NOT correct"
 */
int parsec_redistribute_check2(parsec_context_t *parsec,
                              parsec_tiled_matrix_t *dcY,
                              int size_row, int size_col,
                              int disi_Y, int disj_Y);

/**
 * @brief Redistribute source to target of PTG: no optimization of variable size
 *
 * @details
 * Source and target could be ANY distribuiton with ANY displacement
 * in both source and target.
 *
 * @param [in] source: source distribution, already distributed and allocated
 * @param [out] target: target distribution, redistributed and allocated
 * @param [in] size_row: row size to be redistributed
 * @param [in] size_col: column size to be redistributed
 * @param [in] disi_source: row displacement in source
 * @param [in] disj_source: column displacement in source
 * @param [in] disi_target: row displacement in target
 * @param [in] disj_target: column displacement in target
 */
int parsec_redistribute_no_optimization(parsec_context_t *parsec,
                        parsec_tiled_matrix_t *source,
                        parsec_tiled_matrix_t *target,
                        int size_row, int size_col,
                        int disi_source, int disj_source,
                        int disi_target, int disj_target);

/**
 * @brief redistribute init operator
 *
 * @param [in] es: execution stream
 * @param [in] descA: tiled matrix date descriptor
 * @param [inout] A:  inout data
 * @param [in] uplo: matrix shape
 * @param [in] m: tile row index
 * @param [in] n: tile column index
 * @param [in] args: NULL
 */
static inline int redistribute_init_ops(parsec_execution_stream_t *es,
                                const parsec_tiled_matrix_t *descA,
                                void *_A, parsec_matrix_uplo_t uplo,
                                int m, int n, void *args){
    DTYPE *A = (DTYPE *)_A;
    int initvalue = ((int *)args)[0];

    for(int j = 0; j < descA->nb; j++) {
        for(int i = 0; i < descA->mb; i++) {
            if( initvalue )
                A[j*descA->mb+i] = (DTYPE)(10*i+j+1)/(i*j+10);
            else
                A[j*descA->mb+i] = (DTYPE)0.0;
        }
    }

    (void)es; (void)uplo; (void)m; (void)n;
    return 0;
}
