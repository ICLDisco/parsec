/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "zlanm2.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlanm2_New - Generates the handle that computes an estimate of the
 *  matrix 2-norm:
 *
 *     ||A||_2 = sqrt( \lambda_{max} A* A ) = \sigma_{max}(A)
 *
 *  WARNING: The computations are not done by this call
 *
 *******************************************************************************
 *
 * @param[in] A
 *          The descriptor of the matrix A.
 *          Must be a two_dim_rectangle_cyclic matrix.
 *
 * @param[out] result
 *          The estimate of the norm described above.
 *
 * @param[out] info
 *          Returns the number of iterations performed by the algorithm. If no
 *          convergence, then returns -1. Can be NULL.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zlanm2_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlanm2
 * @sa dplasma_zlanm2_Destruct
 * @sa dplasma_clanm2_New
 * @sa dplasma_dlanm2_New
 * @sa dplasma_slanm2_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zlanm2_New( const tiled_matrix_desc_t *A,
                    double *result, int *info )
{
    int P, Q, m, n, mb, nb, elt;
    two_dim_block_cyclic_t *Tdist;
    parsec_handle_t *parsec_zlanm2 = NULL;

    if ( !(A->dtype & two_dim_block_cyclic_type) ) {
        dplasma_error("dplasma_zlanm2", "illegal type of descriptor for A");
        return NULL;
    }

    P = ((two_dim_block_cyclic_t*)A)->grid.rows;
    Q = ((two_dim_block_cyclic_t*)A)->grid.cols;

    /* Warning: Pb with smb/snb when mt/nt lower than P/Q */
    mb = A->mb;
    nb = A->nb;
    m  = dplasma_imax(A->mt, P);
    n  = dplasma_imax(A->nt, Q);
    elt = 2;

    /* Create a copy of the A matrix to be used as a data distribution metric.
     * As it is used as a NULL value we must have a data_copy and a data associated
     * with it, so we can create them here.
     * Create the task distribution */
    Tdist = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));

    two_dim_block_cyclic_init(
        Tdist, matrix_RealDouble, matrix_Tile,
        A->super.nodes, A->super.myrank,
        1, 1, /* Dimensions of the tiles              */
        m, n, /* Dimensions of the matrix             */
        0, 0, /* Starting points (not important here) */
        m, n, /* Dimensions of the submatrix          */
        1, 1, P);
    Tdist->super.super.data_of = NULL;
    Tdist->super.super.data_of_key = NULL;

    /* Create the DAG */
    if (NULL != info) {
        *info = -1;
    }
    parsec_zlanm2 = (parsec_handle_t*)parsec_zlanm2_new(
        P, Q, (parsec_ddesc_t*)Tdist, A, result, info);

    /* Set the datatypes */
    dplasma_add2arena_tile(((parsec_zlanm2_handle_t*)parsec_zlanm2)->arenas[PARSEC_zlanm2_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_double_complex_t, A->mb);
    dplasma_add2arena_rectangle(((parsec_zlanm2_handle_t*)parsec_zlanm2)->arenas[PARSEC_zlanm2_ZCOL_ARENA],
                                mb * sizeof(parsec_complex64_t), PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_complex_t, mb, 1, -1);
    dplasma_add2arena_rectangle(((parsec_zlanm2_handle_t*)parsec_zlanm2)->arenas[PARSEC_zlanm2_ZROW_ARENA],
                                nb * sizeof(parsec_complex64_t), PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_complex_t, 1, nb, -1);
    dplasma_add2arena_rectangle(((parsec_zlanm2_handle_t*)parsec_zlanm2)->arenas[PARSEC_zlanm2_DROW_ARENA],
                                nb * sizeof(double), PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_t, 1, nb, -1);
    dplasma_add2arena_rectangle(((parsec_zlanm2_handle_t*)parsec_zlanm2)->arenas[PARSEC_zlanm2_ELT_ARENA],
                                elt * sizeof(double), PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_t, elt, 1, -1);

    return (parsec_handle_t*)parsec_zlanm2;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlanm2_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zlanm2_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlanm2_New
 * @sa dplasma_zlanm2
 *
 ******************************************************************************/
void
dplasma_zlanm2_Destruct( parsec_handle_t *handle )
{
    parsec_zlanm2_handle_t *parsec_zlanm2 = (parsec_zlanm2_handle_t *)handle;

    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)(parsec_zlanm2->_g_Tdist) );
    free( parsec_zlanm2->_g_Tdist );

    parsec_matrix_del2arena( parsec_zlanm2->arenas[PARSEC_zlanm2_DEFAULT_ARENA] );
    parsec_matrix_del2arena( parsec_zlanm2->arenas[PARSEC_zlanm2_ZCOL_ARENA] );
    parsec_matrix_del2arena( parsec_zlanm2->arenas[PARSEC_zlanm2_ZROW_ARENA] );
    parsec_matrix_del2arena( parsec_zlanm2->arenas[PARSEC_zlanm2_DROW_ARENA] );
    parsec_matrix_del2arena( parsec_zlanm2->arenas[PARSEC_zlanm2_ELT_ARENA] );

    parsec_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlanm2 - Computes an estimate of the matrix 2-norm:
 *
 *     ||A||_2 = sqrt( \lambda_{max} A* A ) = \sigma_{max}(A)
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] A
 *          The descriptor of the matrix A.
 *          Must be a two_dim_rectangle_cyclic matrix.
 *
 * @param[out] info
 *          Returns the number of iterations performed by the algorithm. If no
 *          convergence, then returns -1. Can be NULL.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The computed estimate of the norm described above.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlanm2_New
 * @sa dplasma_zlanm2_Destruct
 * @sa dplasma_clanm2
 * @sa dplasma_dlanm2
 * @sa dplasma_slanm2
 *
 ******************************************************************************/
double
dplasma_zlanm2( parsec_context_t *parsec,
                const tiled_matrix_desc_t *A,
                int *info )
{
    double result = 0.;
    parsec_handle_t *parsec_zlanm2 = NULL;

    if ( !(A->dtype & two_dim_block_cyclic_type) ) {
        dplasma_error("dplasma_zlanm2", "illegal type of descriptor for A");
        return -3.;
    }

    parsec_zlanm2 = dplasma_zlanm2_New(A, &result, info);

    if ( parsec_zlanm2 != NULL )
    {
        parsec_enqueue( parsec, (parsec_handle_t*)parsec_zlanm2);
        dplasma_wait_until_completion(parsec);
        dplasma_zlanm2_Destruct( parsec_zlanm2 );
    }

    return result;
}

