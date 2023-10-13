/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "redistribute_internal.h"
#include "parsec/interfaces/dtd/insert_function_internal.h"

#define TILE_OF_INSERT(DC, I, J) \
    parsec_dtd_tile_of(&(dc##DC->super.super), (&(dc##DC->super.super))->data_key(&(dc##DC->super.super), I, J))

static inline int parsec_imin(int a, int b)
{
    return (a <= b) ? a : b;
};

/* IDs for the Arena Datatypes */
static int TARGET;
static int SOURCE;

/**
 * @brief CORE function
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
 * @param [in] mb_T: row tile size, including ghost region
 * @param [in] mb_T_inner: row tile size, not including ghost region
 * @param [in] nb_T_inner: column tile size, not including ghost region
 * @param [in] R: radius of ghost region
 * @param [in] i_start_T: row displacement of T
 * @param [in] j_start_T: column displacement of T
 */
void
CORE_redistribute_dtd(DTYPE *T, DTYPE *Y, int mb_Y, int nb_Y, int m_Y, int n_Y,
                      int m_Y_start, int m_Y_end, int n_Y_start, int n_Y_end, int i_start,
                      int i_end, int j_start, int j_end, int mb_T, int mb_T_inner,
                      int nb_T_inner, int R, int i_start_T, int j_start_T)
{
    int mb_Y_inner = mb_Y - 2 * R;
    int nb_Y_inner = nb_Y - 2 * R;
    int TL_row = parsec_imin(mb_Y_inner-i_start, mb_T_inner);
    int TL_col = parsec_imin(nb_Y_inner-j_start, nb_T_inner);

    /* Check start point of tiles located in T */
    if( m_Y != m_Y_start )
        i_start_T = (m_Y - m_Y_start) * mb_Y_inner - i_start + i_start_T;

    if( n_Y != n_Y_start )
        j_start_T = (n_Y - n_Y_start) * nb_Y_inner - j_start + j_start_T;

    /* Copy from Y to T*/
    if( m_Y == m_Y_start ){
        /* North west corner */
        if( n_Y == n_Y_start ){
            MOVE_SUBMATRIX(TL_row, TL_col, Y, i_start+R, j_start+R, mb_Y, T, i_start_T, j_start_T, mb_T);
        }
        /* North bar*/
        else if( (n_Y > n_Y_start) && (n_Y < n_Y_end) ){
            MOVE_SUBMATRIX(TL_row, nb_Y_inner, Y, i_start+R, R, mb_Y, T, i_start_T, j_start_T, mb_T);
        }
        /* North east corner */
        else if( (n_Y == n_Y_end) && (n_Y_start != n_Y_end) ){
            MOVE_SUBMATRIX(TL_row, j_end+1, Y, i_start+R, R, mb_Y, T, i_start_T, j_start_T, mb_T);
        }
    }

    else if( (m_Y > m_Y_start) && (m_Y < m_Y_end) ){
        /* West bar*/
        if( n_Y == n_Y_start ){
            MOVE_SUBMATRIX(mb_Y_inner, TL_col, Y, R, j_start+R, mb_Y, T, i_start_T, j_start_T, mb_T);
        }
        /* Inner tile*/
        else if( (n_Y > n_Y_start) && (n_Y < n_Y_end) ){
            MOVE_SUBMATRIX(mb_Y_inner, nb_Y_inner, Y, R, R, mb_Y, T, i_start_T, j_start_T, mb_T);
        }
        /* East bar*/
        else if( (n_Y == n_Y_end) && (n_Y_start != n_Y_end) ){
            MOVE_SUBMATRIX(mb_Y_inner, j_end+1, Y, R, R, mb_Y, T, i_start_T, j_start_T, mb_T);
        }
    }

    else if( (m_Y == m_Y_end) && (m_Y_start != m_Y_end) ){
        /* South west corner */
        if( n_Y == n_Y_start ){
            MOVE_SUBMATRIX(i_end+1, TL_col, Y, R, j_start+R, mb_Y, T, i_start_T, j_start_T, mb_T);
        }
        /* South bar*/
        else if( (n_Y > n_Y_start) && (n_Y < n_Y_end) ){
            MOVE_SUBMATRIX(i_end+1, nb_Y_inner, Y, R, R, mb_Y, T, i_start_T, j_start_T, mb_T);
        }
        /* South east corner */
        else if( (n_Y == n_Y_end) && (n_Y_start != n_Y_end) ){
            MOVE_SUBMATRIX(i_end+1, j_end+1, Y, R, R, mb_Y, T, i_start_T, j_start_T, mb_T);
        }
    }
}

/*
 * @brief redistribute DTD: unpack parameters
 */
static int parsec_core_redistribute_dtd(parsec_execution_stream_t *es, parsec_task_t *this_task){

    (void)es;
    void *T;
    void *Y;
    int mb_Y;
    int nb_Y;
    int m_Y;
    int n_Y;
    int m_Y_start;
    int m_Y_end;
    int n_Y_start;
    int n_Y_end;
    int i_start;
    int i_end;
    int j_start;
    int j_end;
    int mb_T;
    int mb_T_inner;
    int nb_T_inner;
    int i_start_T;
    int j_start_T;

    parsec_dtd_unpack_args(this_task, &T, &Y, &mb_Y, &nb_Y, &m_Y, &n_Y, &m_Y_start, &m_Y_end,
                                      &n_Y_start, &n_Y_end, &i_start, &i_end, &j_start, &j_end,
                                      &mb_T, &mb_T_inner, &nb_T_inner, &i_start_T, &j_start_T);

    CORE_redistribute_dtd(T, Y, mb_Y, nb_Y, m_Y, n_Y, m_Y_start, m_Y_end, n_Y_start,
                            n_Y_end, i_start, i_end, j_start, j_end, mb_T, mb_T_inner,
                            nb_T_inner, 0, i_start_T, j_start_T);

    return PARSEC_HOOK_RETURN_DONE;
}

/*
 * @brief Insert Task
 */
static int
insert_task(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    parsec_matrix_block_cyclic_t *dcY;
    parsec_matrix_block_cyclic_t *dcT;
    int size_row, size_col, disi_Y, disj_Y, disi_T, disj_T;

    parsec_taskpool_t *dtd_tp = (parsec_taskpool_t *)this_task->taskpool;
    parsec_dtd_unpack_args(this_task, &dcY, &dcT, &size_row, &size_col,
                           &disi_Y, &disj_Y, &disi_T, &disj_T);

    /* Parameters */
    int m_T, n_T, m_Y, n_Y;
    int i_start, j_start, i_end, j_end;
    int sizei_T, sizej_T, i_start_T, j_start_T;
    int m_Y_start, n_Y_start, m_Y_end, n_Y_end;
    int mb_T_inner, nb_T_inner;

    /* Global parameters */
    int mb_Y_INNER = dcY->super.mb;
    int nb_Y_INNER = dcY->super.nb;
    int mb_T_INNER = dcT->super.mb;
    int nb_T_INNER = dcT->super.nb;
    int m_T_START = disi_T / mb_T_INNER;
    int n_T_START = disj_T / nb_T_INNER;
    int m_T_END = (size_row + disi_T - 1) / mb_T_INNER;
    int n_T_END = (size_col + disj_T - 1) / nb_T_INNER;

    /* Insert task */
    for(m_T = m_T_START; m_T <= m_T_END; m_T++){
        for(n_T = n_T_START; n_T <= n_T_END; n_T++){
            mb_T_inner = getsize(m_T, m_T_START, m_T_END, mb_T_INNER, size_row, disi_T%mb_T_INNER);
            nb_T_inner = getsize(n_T, n_T_START, n_T_END, nb_T_INNER, size_col, disj_T%nb_T_INNER);

            sizei_T = (m_T - m_T_START) * mb_T_INNER - disi_T % mb_T_INNER;
            sizej_T = (n_T - n_T_START) * nb_T_INNER - disj_T % nb_T_INNER;

            i_start = (m_T == m_T_START)? disi_Y % mb_Y_INNER: (sizei_T + disi_Y) % mb_Y_INNER;
            j_start = (n_T == n_T_START)? disj_Y % nb_Y_INNER: (sizej_T + disj_Y) % nb_Y_INNER;
            i_end = (i_start + mb_T_inner - 1) % mb_Y_INNER;
            j_end = (j_start + nb_T_inner - 1) % nb_Y_INNER;

            m_Y_start = (m_T == m_T_START)? disi_Y / mb_Y_INNER: (sizei_T + disi_Y) / mb_Y_INNER;
            n_Y_start = (n_T == n_T_START)? disj_Y / nb_Y_INNER: (sizej_T + disj_Y) / nb_Y_INNER;
            m_Y_end = (m_T == m_T_START)? (disi_Y + mb_T_inner - 1) / mb_Y_INNER: (sizei_T + disi_Y + mb_T_inner - 1) / mb_Y_INNER;
            n_Y_end = (n_T == n_T_START)? (disj_Y + nb_T_inner - 1) / nb_Y_INNER: (sizej_T + disj_Y + nb_T_inner - 1) / nb_Y_INNER;

            i_start_T = (m_T == m_T_START)? disi_T % mb_T_INNER: 0;
            j_start_T = (n_T == n_T_START)? disj_T % nb_T_INNER: 0;

            for(m_Y = m_Y_start; m_Y <= m_Y_end; m_Y++){
                for(n_Y = n_Y_start; n_Y <= n_Y_end; n_Y++){
                      parsec_dtd_insert_task(dtd_tp,
                                             &parsec_core_redistribute_dtd, 0, PARSEC_DEV_CPU,"redistribute_dtd",
                                             PASSED_BY_REF, TILE_OF_INSERT(T, m_T, n_T),   PARSEC_OUTPUT | TARGET | PARSEC_AFFINITY,
                                             PASSED_BY_REF, TILE_OF_INSERT(Y, m_Y, n_Y),   PARSEC_INPUT | SOURCE,
                                             sizeof(int), &dcY->super.mb, PARSEC_VALUE,
                                             sizeof(int), &dcY->super.nb, PARSEC_VALUE,
                                             sizeof(int), &m_Y, PARSEC_VALUE,
                                             sizeof(int), &n_Y, PARSEC_VALUE,
                                             sizeof(int), &m_Y_start, PARSEC_VALUE,
                                             sizeof(int), &m_Y_end, PARSEC_VALUE,
                                             sizeof(int), &n_Y_start, PARSEC_VALUE,
                                             sizeof(int), &n_Y_end, PARSEC_VALUE,
                                             sizeof(int), &i_start, PARSEC_VALUE,
                                             sizeof(int), &i_end, PARSEC_VALUE,
                                             sizeof(int), &j_start, PARSEC_VALUE,
                                             sizeof(int), &j_end, PARSEC_VALUE,
                                             sizeof(int), &dcT->super.mb, PARSEC_VALUE,
                                             sizeof(int), &mb_T_inner, PARSEC_VALUE,
                                             sizeof(int), &nb_T_inner, PARSEC_VALUE,
                                             sizeof(int), &i_start_T, PARSEC_VALUE,
                                             sizeof(int), &j_start_T, PARSEC_VALUE,
                                             PARSEC_DTD_ARG_END );
                }
            }
        }
    }

    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)dcY );
    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)dcT );

    return PARSEC_HOOK_RETURN_DONE;
}

/*
 * @brief redistribute DTD: unpack parameters
 */
static int parsec_core_redistribute_reshuffle_dtd(parsec_execution_stream_t *es, parsec_task_t *this_task){

    (void)es;
    void *T;
    void *Y;
    int mb;
    int nb;
    int lda;
    int m_T;
    int m_T_END;

    parsec_dtd_unpack_args(this_task, &T, &Y, &mb, &nb, &lda, &m_T, &m_T_END);

    if( m_T == m_T_END ) {
        CORE_redistribute_reshuffle_copy(T, Y, mb, nb, lda, lda);
    } else {
        memcpy((void *)T, (void *)Y, mb*nb*sizeof(DTYPE));
    }

    return PARSEC_HOOK_RETURN_DONE;
}

/*
 * @brief Insert Task Same Tile Size
 */
static int
insert_task_reshuffle(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    (void)es;
    parsec_matrix_block_cyclic_t *dcY;
    parsec_matrix_block_cyclic_t *dcT;
    int size_row, size_col, disi_Y, disj_Y, disi_T, disj_T;

    parsec_taskpool_t *dtd_tp = (parsec_taskpool_t *)this_task->taskpool;
    parsec_dtd_unpack_args(this_task, &dcY, &dcT, &size_row, &size_col,
                           &disi_Y, &disj_Y, &disi_T, &disj_T);

    /* Parameters */
    int m_Y, n_Y, m_T, n_T, mb, nb, count = 0;
    int m_Y_START = disi_Y / dcY->super.mb;
    int n_Y_START = disj_Y / dcY->super.nb;
    int m_T_START = disi_T / dcT->super.mb;
    int n_T_START = disj_T / dcT->super.nb;
    int m_T_END = (disi_T+size_row-1) / dcT->super.mb;
    int n_T_END = (disj_T+size_col-1) / dcT->super.nb;

    /* Insert task */
    for(m_T = m_T_START; m_T <= m_T_END; m_T++, count++) {
        for(n_T = n_T_START; n_T <= n_T_END ; n_T++) {
            m_Y = m_T - m_T_START + m_Y_START;
            n_Y = n_T - n_T_START + n_Y_START;
            mb = (m_T == m_T_END)? parsec_imin(dcT->super.mb,
                 size_row-(m_T_END-m_T_START)*dcT->super.mb): dcT->super.mb;
            nb = (n_T == n_T_END)? parsec_imin(dcT->super.nb,
                 size_col-(n_T_END-n_T_START)*dcT->super.nb): dcT->super.nb;
            parsec_dtd_insert_task(dtd_tp,
                                   &parsec_core_redistribute_reshuffle_dtd, 0,
                                   PARSEC_DEV_CPU, "redistribute_reshuffle_dtd",
                                   PASSED_BY_REF, TILE_OF_INSERT(T, m_T, n_T),   PARSEC_OUTPUT | TARGET | PARSEC_AFFINITY,
                                   PASSED_BY_REF, TILE_OF_INSERT(Y, m_Y, n_Y),   PARSEC_INPUT | SOURCE,
                                   sizeof(int), &mb, PARSEC_VALUE,
                                   sizeof(int), &nb, PARSEC_VALUE,
                                   sizeof(int), &dcT->super.mb, PARSEC_VALUE,
                                   sizeof(int), &m_T, PARSEC_VALUE,
                                   sizeof(int), &m_T_END, PARSEC_VALUE,
                                   PARSEC_DTD_ARG_END );
        }
    }

    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)dcY );
    parsec_dtd_data_flush_all( dtd_tp, (parsec_data_collection_t *)dcT );

    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief Redistribute dcY to dcT, only deal with taskpool,
 *        not parsec context
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
static int
parsec_redistribute_New_dtd(parsec_context_t *parsec,
                            parsec_tiled_matrix_t *dcY,
                            parsec_tiled_matrix_t *dcT,
                            int size_row, int size_col,
                            int disi_Y, int disj_Y,
                            int disi_T, int disj_T)
{
    parsec_arena_datatype_t *adt;
    if( size_row < 1 || size_col < 1 ) {
        if( 0 == dcY->super.myrank )
            parsec_warning("ERROR: Submatrix size should be bigger than 1\n");
        return PARSEC_ERROR;
    }

    if( disi_Y < 0 || disj_Y < 0 ||
        disi_T < 0 || disj_T < 0 ) {
        if( 0 == dcY->super.myrank )
            parsec_warning("ERROR: Submatrix displacement should not be negative\n");
        return PARSEC_ERROR;
    }

    if( (disi_Y+size_row > dcY->lmt*dcY->mb)
        || (disj_Y+size_col > dcY->lnt*dcY->nb) ){
        if( 0 == dcY->super.myrank )
            parsec_warning("ERROR: Submatrix exceed SOURCE size\n");
        return PARSEC_ERROR;
    }

    if( (disi_T+size_row > dcT->lmt*dcT->mb)
        || (disj_T + size_col > dcT->lnt*dcT->nb) ){
        if( 0 == dcY->super.myrank )
            parsec_warning("ERROR: Submatrix exceed TARGET size\n");
        return PARSEC_ERROR;
    }

    /* Initializing dc for dtd */
    parsec_dtd_data_collection_init((parsec_data_collection_t *)dcY);

    /* Initializing dc for dtd */
    parsec_dtd_data_collection_init((parsec_data_collection_t *)dcT);

    /* Getting new parsec handle of dtd type */
    parsec_taskpool_t *dtd_tp = parsec_dtd_taskpool_new();

    /* Allocating data arrays to be used by comm engine */
    adt = parsec_dtd_create_arena_datatype(parsec, &TARGET);
    parsec_add2arena(adt,
                     MY_TYPE, PARSEC_MATRIX_FULL,
                     1, dcT->mb, dcT->nb, dcT->mb,
                     PARSEC_ARENA_ALIGNMENT_SSE, -1);

    adt = parsec_dtd_create_arena_datatype(parsec, &SOURCE);
    parsec_add2arena(adt,
                     MY_TYPE, PARSEC_MATRIX_FULL,
                     1, dcY->mb, dcY->nb, dcY->mb,
                     PARSEC_ARENA_ALIGNMENT_SSE, -1);

    /* Registering the handle with parsec context */
    parsec_context_add_taskpool(parsec, dtd_tp);

    /* Insert task */
    if( (dcY->mb == dcT->mb) && (dcY->nb == dcT->nb) && (disi_Y % dcY->mb == 0)
        && (disj_Y % dcY->nb == 0) && (disi_T % dcT->mb == 0) && (disj_T % dcT->nb == 0) ) {
        /* When tile sizes are the same and displacements are at start of tiles */
        parsec_dtd_insert_task( dtd_tp,       insert_task_reshuffle, 0, PARSEC_DEV_CPU, "insert_task_reshuffle",
                       sizeof(parsec_matrix_block_cyclic_t *), (parsec_matrix_block_cyclic_t *)dcY,  PARSEC_REF,
                       sizeof(parsec_matrix_block_cyclic_t *), (parsec_matrix_block_cyclic_t *)dcT,  PARSEC_REF,
                       sizeof(int),                &size_row,           PARSEC_VALUE,
                       sizeof(int),                &size_col,           PARSEC_VALUE,
                       sizeof(int),                &disi_Y,             PARSEC_VALUE,
                       sizeof(int),                &disj_Y,             PARSEC_VALUE,
                       sizeof(int),                &disi_T,             PARSEC_VALUE,
                       sizeof(int),                &disj_T,             PARSEC_VALUE,
                       PARSEC_DTD_ARG_END );
    } else {
        parsec_dtd_insert_task( dtd_tp,       insert_task, 0, PARSEC_DEV_CPU, "insert_task",
                       sizeof(parsec_matrix_block_cyclic_t *), (parsec_matrix_block_cyclic_t *)dcY,  PARSEC_REF,
                       sizeof(parsec_matrix_block_cyclic_t *), (parsec_matrix_block_cyclic_t *)dcT,  PARSEC_REF,
                       sizeof(int),                      &size_row,                      PARSEC_VALUE,
                       sizeof(int),                      &size_col,                      PARSEC_VALUE,
                       sizeof(int),                      &disi_Y,                        PARSEC_VALUE,
                       sizeof(int),                      &disj_Y,                        PARSEC_VALUE,
                       sizeof(int),                      &disi_T,                        PARSEC_VALUE,
                       sizeof(int),                      &disj_T,                        PARSEC_VALUE,
                       PARSEC_DTD_ARG_END );
    }

    /* Finishing all the tasks inserted, but not finishing the handle */
    int rc = parsec_taskpool_wait( dtd_tp );
    PARSEC_CHECK_ERROR(rc, "parsec_taskpool_wait");

    /* Cleaning up the parsec handle */
    parsec_taskpool_free( dtd_tp );

    /* Cleaning data arrays we allocated for communication */
    adt = parsec_dtd_get_arena_datatype(parsec, SOURCE);
    parsec_type_free(&adt->opaque_dtt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec, SOURCE);
    adt = parsec_dtd_get_arena_datatype(parsec, TARGET);
    parsec_type_free(&adt->opaque_dtt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec, TARGET);

    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)dcY );
    parsec_dtd_data_collection_fini( (parsec_data_collection_t *)dcT );

    return PARSEC_SUCCESS;
}

/**
 * @brief Redistribute dcY to dcT
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
int parsec_redistribute_dtd(parsec_context_t *parsec,
                            parsec_tiled_matrix_t *dcY,
                            parsec_tiled_matrix_t *dcT,
                            int size_row, int size_col,
                            int disi_Y, int disj_Y,
                            int disi_T, int disj_T)
{
    /* start parsec context */
    parsec_context_start(parsec);

    /* New function, only deal with taskpool, not parsec context */
    parsec_redistribute_New_dtd(parsec, dcY, dcT, size_row, size_col,
                                disi_Y, disj_Y, disi_T, disj_T);

    /* Waiting on all handle and turning everything off for this context */
    int rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    return PARSEC_SUCCESS;
}
