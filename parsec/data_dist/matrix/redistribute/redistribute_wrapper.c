/*
 * Copyright (c) 2017-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "redistribute_internal.h"
#include "redistribute.h"
#include "redistribute_reshuffle.h"

static inline int parsec_imin(int a, int b)
{
    return (a <= b) ? a : b;
};

static inline int parsec_imax(int a, int b)
{
    return (a >= b) ? a : b;
};

/**
 * @brief New function for redistribute
 *
 * @param [in] dcY: the data, already distributed and allocated
 * @param [out] dcT: the data, redistributed and allocated
 * @param [in] size_row: row size to be redistributed
 * @param [in] size_col: column size to be redistributed
 * @param [in] disi_Y: row displacement in dcY
 * @param [in] disj_Y: column displacement in dcY
 * @param [in] disi_T: row displacement in dcT
 * @param [in] disj_T: column displacement in dcT
 * @return the parsec object to schedule.
 */
parsec_taskpool_t*
parsec_redistribute_New(parsec_tiled_matrix_t *dcY,
                        parsec_tiled_matrix_t *dcT,
                        int size_row, int size_col,
                        int disi_Y, int disj_Y,
                        int disi_T, int disj_T)
{
    parsec_taskpool_t* redistribute_taskpool;
    int num_cols;

    if( size_row < 1 || size_col < 1 ) {
        if( 0 == dcY->super.myrank )
            parsec_warning("ERROR: Submatrix size should be bigger than 1\n");
        return NULL;
    }

    if( disi_Y < 0 || disj_Y < 0 ||
        disi_T < 0 || disj_T < 0 ) {
        if( 0 == dcY->super.myrank )
            parsec_warning("ERROR: Submatrix displacement should not be negative\n");
        return NULL;
    }

    if( (disi_Y+size_row > dcY->lmt*dcY->mb) ||
        (disj_Y+size_col > dcY->lnt*dcY->nb) ){
        if( 0 == dcY->super.myrank )
            parsec_warning("ERROR: Submatrix exceed SOURCE size\n");
        return NULL;
    }

    if( (disi_T+size_row > dcT->lmt*dcT->mb)
        || (disj_T+size_col > dcT->lnt*dcT->nb) ){
        if( 0 == dcY->super.myrank )
            parsec_warning("ERROR: Submatrix exceed TARGET size\n");
        return NULL;
    }

    /* Check distribution, and determine batch size: num_col */
    if( (dcY->dtype & parsec_matrix_tabular_type) && (dcT->dtype & parsec_matrix_tabular_type) ) {
        num_cols = parsec_imin( ceil(size_col/dcY->nb), dcY->super.nodes );
    } else if( (dcY->dtype & parsec_matrix_tabular_type) && (dcT->dtype & parsec_matrix_block_cyclic_type) ) {
        num_cols = ((parsec_matrix_block_cyclic_t *)dcT)->grid.cols * ((parsec_matrix_block_cyclic_t *)dcT)->grid.kcols;
    } else if( (dcY->dtype & parsec_matrix_block_cyclic_type) && (dcT->dtype & parsec_matrix_tabular_type) ) {
        num_cols = ((parsec_matrix_block_cyclic_t *)dcY)->grid.cols * ((parsec_matrix_block_cyclic_t *)dcY)->grid.kcols;
    } else if( (dcY->dtype & parsec_matrix_block_cyclic_type) && (dcT->dtype & parsec_matrix_block_cyclic_type) ) {
        num_cols = parsec_imax( ((parsec_matrix_block_cyclic_t *)dcY)->grid.cols * ((parsec_matrix_block_cyclic_t *)dcY)->grid.kcols,
                                ((parsec_matrix_block_cyclic_t *)dcT)->grid.cols * ((parsec_matrix_block_cyclic_t *)dcT)->grid.kcols );
    } else {
        parsec_warning("This version of data redistribution only supports parsec_matrix_block_cyclic_type and parsec_matrix_tabular_type");
        return NULL;
    }

    /* Optimized version: tile sizes of source and target ar the same,
     * displacements in both source and target are at the start of tiles */
    if( (dcY->mb == dcT->mb) && (dcY->nb == dcT->nb) && (disi_Y % dcY->mb == 0)
        && (disj_Y % dcY->nb == 0) && (disi_T % dcT->mb == 0) && (disj_T % dcT->nb == 0) ) {
        parsec_redistribute_reshuffle_taskpool_t* taskpool = NULL;

        taskpool = parsec_redistribute_reshuffle_new(dcY, dcT, size_row, size_col, disi_Y, disj_Y, disi_T, disj_T);
        taskpool->_g_num_col = num_cols;
        redistribute_taskpool = (parsec_taskpool_t*)taskpool;

        /* Calculate NT, need to update !!! */
        int n_T_START = disj_T / dcT->nb;
        int n_T_END = (size_col+disj_T-1) / dcT->nb;
        taskpool->_g_NT = (n_T_END-n_T_START)/taskpool->_g_num_col;

        parsec_add2arena(&taskpool->arenas_datatypes[PARSEC_redistribute_reshuffle_DEFAULT_ADT_IDX],
                         MY_TYPE, PARSEC_MATRIX_FULL,
                         1, dcY->mb, dcY->nb, dcY->mb,
                         PARSEC_ARENA_ALIGNMENT_SSE, -1 );
        /* General version */
    } else {
        parsec_redistribute_taskpool_t* taskpool = NULL;
        /* R will be used for padding tiles like in AMR,
         * here for a normal redistribution problem, R is set to 0.
         */
        int R = 0;

        taskpool = parsec_redistribute_new(dcY, dcT, size_row, size_col, disi_Y, disj_Y, disi_T, disj_T, R);
        taskpool->_g_num_col = num_cols;
        redistribute_taskpool = (parsec_taskpool_t*)taskpool;

        /* Calculate NT, need to update !!! */
        int n_T_START = disj_T / (dcT->nb-2*R);
        int n_T_END = (size_col+disj_T-1) / (dcT->nb-2*R);
        taskpool->_g_NT = (n_T_END-n_T_START)/taskpool->_g_num_col;

        parsec_add2arena(&taskpool->arenas_datatypes[PARSEC_redistribute_DEFAULT_ADT_IDX],
                         MY_TYPE, PARSEC_MATRIX_FULL,
                         1, 1, 1, 1,
                         PARSEC_ARENA_ALIGNMENT_SSE, -1 );

        int Y_LDA = dcY->storage == PARSEC_MATRIX_LAPACK ? dcY->llm : dcY->mb;
        int T_LDA = dcT->storage == PARSEC_MATRIX_LAPACK ? dcT->llm : dcT->mb;

        parsec_add2arena(&taskpool->arenas_datatypes[PARSEC_redistribute_TARGET_ADT_IDX],
                         MY_TYPE, PARSEC_MATRIX_FULL,
                         1, dcT->mb, dcT->nb, T_LDA,
                         PARSEC_ARENA_ALIGNMENT_SSE, -1 );

        parsec_add2arena(&taskpool->arenas_datatypes[PARSEC_redistribute_INNER_ADT_IDX],
                         MY_TYPE, PARSEC_MATRIX_FULL,
                         1, dcY->mb-2*R, dcY->nb-2*R, Y_LDA,
                         PARSEC_ARENA_ALIGNMENT_SSE, -1 );
    }

    return redistribute_taskpool;
}

/**
 * @param [inout] the parsec object to destroy
 */
static void
__parsec_redistribute_destructor(parsec_redistribute_taskpool_t *redistribute_taskpool)
{
    /* Optimized version: tile sizes of source and target ar the same,
     * displacements in both source and target are at the start of tiles */
    if( (redistribute_taskpool->_g_descY->mb == redistribute_taskpool->_g_descT->mb)
        && (redistribute_taskpool->_g_descY->nb == redistribute_taskpool->_g_descT->nb)
        && (redistribute_taskpool->_g_disi_Y % redistribute_taskpool->_g_descY->mb == 0)
        && (redistribute_taskpool->_g_disj_Y % redistribute_taskpool->_g_descY->nb == 0)
        && (redistribute_taskpool->_g_disi_T % redistribute_taskpool->_g_descT->mb == 0)
        && (redistribute_taskpool->_g_disj_T % redistribute_taskpool->_g_descT->nb == 0) )
    {
        parsec_redistribute_reshuffle_taskpool_t *redistribute_reshuffle_taskpool = (parsec_redistribute_reshuffle_taskpool_t *)redistribute_taskpool;
        parsec_del2arena(&redistribute_reshuffle_taskpool->arenas_datatypes[PARSEC_redistribute_reshuffle_DEFAULT_ADT_IDX]);
    } else {
        parsec_del2arena(&redistribute_taskpool->arenas_datatypes[PARSEC_redistribute_DEFAULT_ADT_IDX]);
        parsec_del2arena(&redistribute_taskpool->arenas_datatypes[PARSEC_redistribute_TARGET_ADT_IDX]);
        // parsec_del2arena(&redistribute_taskpool->arenas_datatypes[PARSEC_redistribute_SOURCE_ADT_IDX]);
        parsec_del2arena(&redistribute_taskpool->arenas_datatypes[PARSEC_redistribute_INNER_ADT_IDX]);
    }
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_redistribute_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_redistribute_destructor);
PARSEC_OBJ_CLASS_INSTANCE(parsec_redistribute_reshuffle_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_redistribute_destructor);

/**
 * @brief Redistribute dcY to dcT in PTG
 *
 * @param [in] dcY: source distribution, already distributed and allocated
 * @param [out] dcT: target distribution, redistributed and allocated
 * @param [in] size_row: row size to be redistributed
 * @param [in] size_col: column size to be redistributed
 * @param [in] disi_Y: row displacement in dcY
 * @param [in] disj_Y: column displacement in dcY
 * @param [in] disi_T: row displacement in dcT
 * @param [in] disj_T: column displacement in dcT
 */
int parsec_redistribute(parsec_context_t *parsec,
                        parsec_tiled_matrix_t *dcY,
                        parsec_tiled_matrix_t *dcT,
                        int size_row, int size_col,
                        int disi_Y, int disj_Y,
                        int disi_T, int disj_T)
{
    parsec_taskpool_t *parsec_redistribute_ptg = NULL;

    parsec_redistribute_ptg = parsec_redistribute_New(
                              dcY, dcT, size_row, size_col, disi_Y,
                              disj_Y, disi_T, disj_T);

    if( NULL != parsec_redistribute_ptg ){
        parsec_context_add_taskpool(parsec, parsec_redistribute_ptg);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        return PARSEC_SUCCESS;
    }

    return PARSEC_ERR_NOT_SUPPORTED;
}

