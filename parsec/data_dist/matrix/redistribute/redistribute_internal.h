/*
 * Copyright (c) 2017-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/two_dim_tabular.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/datatype.h"
#include "parsec/arena.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

/** @brief Define datatype, should be the same as dc source */
#define DTYPE           double
#define MY_TYPE parsec_datatype_double_t

/** @brief Macro, defined copy submatrix of S (Source) to submatrix of D (Destination) */
#define MOVE_SUBMATRIX(m, n, S, S_i, S_j, S_lda, D, D_i, D_j, D_lda)      \
    do {                                                                  \
        for(int j = 0; j < n; j++) {                                      \
            memcpy((void *)&D[(D_j+j)*(D_lda)+D_i], (void *)&S[(S_j+j)*(S_lda)+S_i], (m)*sizeof(DTYPE));     \
        }                                                                 \
    } while(0)

#define MOVE_SUBMATRIX_SEND(m, n, S, S_i, S_j, S_lda, D, D_i, D_j, D_lda) \
    do {                                                                  \
        if( m == S_lda && m == D_lda ) {                                  \
            memcpy((void *)&D[(D_j)*(D_lda)+D_i], (void *)&S[(S_j)*(S_lda)+S_i], (m)*(n)*sizeof(DTYPE));     \
        } else {                                                          \
            for(int j = 0; j < n; j++)                                    \
                memcpy((void *)&D[(D_j+j)*(D_lda)+D_i], (void *)&S[(S_j+j)*(S_lda)+S_i], (m)*sizeof(DTYPE)); \
        }                                                                 \
    } while(0)

#define MOVE_SUBMATRIX_RECEIVE(m, n, S, S_i, S_j, S_lda, D, D_i, D_j, D_lda) \
    do {                                                                     \
        if( m == S_lda && m == D_lda ) {                                     \
            memcpy((void *)&D[(D_j)*(D_lda)+D_i], (void *)&S[(S_j)*(S_lda)+S_i], (m)*(n)*sizeof(DTYPE));     \
        } else {                                                             \
            for(int j = 0; j < n; j++)                                       \
                memcpy((void *)&D[(D_j+j)*(D_lda)+D_i], (void *)&S[(S_j+j)*(S_lda)+S_i], (m)*sizeof(DTYPE)); \
        }                                                                    \
    } while(0)

/**
 * @brief Get size
 *
 * @param [in] index: tile index
 * @param [in] index_start: tile index start
 * @param [in] index_end: tile index end
 * @param [in] mb: tile size
 * @param [in] size: total size of submatrix
 * @param [in] dis: displacement
 * @return size in different case
 */
static inline int
getsize(const int index, const int index_start, const int index_end,
        const int mb, const int size, const int dis)
{
    if( index_start == index_end )
        return size;
    if( index == index_start )
        return mb - dis;
    if( index == index_end )
        return size + dis - (index_end-index_start) * mb;

    return mb;
}

/**
 * @brief CORE function, used in parsec_redistribute_dtd
 *
 * @details
 * Copy data for Y to T
 *
 * @param [out] T: data, already distributed and allocated
 * @param [in] Y: data, already distributed and allocated
 * @param [in] mb_Y: row tile size
 * @param [in] nb_Y: column tile size
 * @param [in] m_Y: row tile index
 * @param [in] n_Y: column tile index
 * @param [in] m_Y_start: row tile index start
 * @param [in] m_Y_end: row tile index end
 * @param [in] n_Y_start: column tile index start
 * @param [in] n_Y_end: column tile index end
 * @param [in] i_start: row start index of submatrix
 * @param [in] i_end: row end index of submatrix
 * @param [in] j_start: column start index of submatrix
 * @param [in] j_end: column end index of submatrix
 * @param [in] mb_T_inner: row size, not including ghost region
 * @param [in] R: radius of ghost region
 * @param [in] i_start_T: row displacememnt in T
 * @param [in] j_start_T: column displacememnt in T
 */
void CORE_redistribute_dtd(DTYPE* T, DTYPE* Y, int mb_Y, int nb_Y, int m_Y, int n_Y,
                           int m_Y_start, int m_Y_end, int n_Y_start, int n_Y_end,
                           int i_start, int i_end, int j_start, int j_end, int mb_T_inner,
                           int size_row, int size_col, int R, int i_start_T, int j_start_T);

/**
 * @brief Copy from Y to T
 *
 * @param [out] T: target
 * @param [in] Y: source
 * @param [in] mb: row size to be copied
 * @param [in] nb: column size to be copied
 * @param [in] T_LDA: LDA of T
 * @param [in] Y_LDA: LDA of Y
 */
static inline void
CORE_redistribute_reshuffle_copy(DTYPE *T, DTYPE *Y, const int mb,
                                 const int nb, const int T_LDA, const int Y_LDA)
{
    MOVE_SUBMATRIX(mb, nb, Y, 0, 0, Y_LDA, T, 0, 0, T_LDA);
}

