/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> s d c
 *
 */

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
 *          \retval The dague handle describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
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
dague_handle_t*
dplasma_zlanm2_New( const tiled_matrix_desc_t *A,
                    double *result, int *info )
{
    int P, Q, m, n, mb, nb, elt;
    two_dim_block_cyclic_t *Tdist;
    dague_handle_t *dague_zlanm2 = NULL;

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
    dague_zlanm2 = (dague_handle_t*)dague_zlanm2_new(
        P, Q, (dague_ddesc_t*)Tdist, A, result, info);

    /* Set the datatypes */
    dplasma_add2arena_tile(((dague_zlanm2_handle_t*)dague_zlanm2)->arenas[DAGUE_zlanm2_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           dague_datatype_double_complex_t, A->mb);
    dplasma_add2arena_rectangle(((dague_zlanm2_handle_t*)dague_zlanm2)->arenas[DAGUE_zlanm2_ZCOL_ARENA],
                                mb * sizeof(dague_complex64_t), DAGUE_ARENA_ALIGNMENT_SSE,
                                dague_datatype_double_complex_t, mb, 1, -1);
    dplasma_add2arena_rectangle(((dague_zlanm2_handle_t*)dague_zlanm2)->arenas[DAGUE_zlanm2_ZROW_ARENA],
                                nb * sizeof(dague_complex64_t), DAGUE_ARENA_ALIGNMENT_SSE,
                                dague_datatype_double_complex_t, 1, nb, -1);
    dplasma_add2arena_rectangle(((dague_zlanm2_handle_t*)dague_zlanm2)->arenas[DAGUE_zlanm2_DROW_ARENA],
                                nb * sizeof(double), DAGUE_ARENA_ALIGNMENT_SSE,
                                dague_datatype_double_t, 1, nb, -1);
    dplasma_add2arena_rectangle(((dague_zlanm2_handle_t*)dague_zlanm2)->arenas[DAGUE_zlanm2_ELT_ARENA],
                                elt * sizeof(double), DAGUE_ARENA_ALIGNMENT_SSE,
                                dague_datatype_double_t, elt, 1, -1);

    return (dague_handle_t*)dague_zlanm2;
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
dplasma_zlanm2_Destruct( dague_handle_t *handle )
{
    dague_zlanm2_handle_t *dague_zlanm2 = (dague_zlanm2_handle_t *)handle;

    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)(dague_zlanm2->_g_Tdist) );
    free( dague_zlanm2->_g_Tdist );

    dague_matrix_del2arena( dague_zlanm2->arenas[DAGUE_zlanm2_DEFAULT_ARENA] );
    dague_matrix_del2arena( dague_zlanm2->arenas[DAGUE_zlanm2_ZCOL_ARENA] );
    dague_matrix_del2arena( dague_zlanm2->arenas[DAGUE_zlanm2_ZROW_ARENA] );
    dague_matrix_del2arena( dague_zlanm2->arenas[DAGUE_zlanm2_DROW_ARENA] );
    dague_matrix_del2arena( dague_zlanm2->arenas[DAGUE_zlanm2_ELT_ARENA] );

    dague_handle_free(handle);
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
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
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
dplasma_zlanm2( dague_context_t *dague,
                const tiled_matrix_desc_t *A,
                int *info )
{
    double result = 0.;
    dague_handle_t *dague_zlanm2 = NULL;

    if ( !(A->dtype & two_dim_block_cyclic_type) ) {
        dplasma_error("dplasma_zlanm2", "illegal type of descriptor for A");
        return -3.;
    }

    dague_zlanm2 = dplasma_zlanm2_New(A, &result, info);

    if ( dague_zlanm2 != NULL )
    {
        dague_enqueue( dague, (dague_handle_t*)dague_zlanm2);
        dplasma_progress(dague);
        dplasma_zlanm2_Destruct( dague_zlanm2 );
    }

    return result;
}

