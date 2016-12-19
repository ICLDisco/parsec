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

#include "zlange_frb_cyclic.h"
#include "zlange_one_cyclic.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlantr_New - Generates the handle that computes the value
 *
 *     zlantr = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
 *              (
 *              ( norm1(A),         NORM = PlasmaOneNorm
 *              (
 *              ( normI(A),         NORM = PlasmaInfNorm
 *              (
 *              ( normF(A),         NORM = PlasmaFrobeniusNorm
 *
 *  where norm1 denotes the one norm of a matrix (maximum column sum),
 *  normI denotes the infinity norm of a matrix (maximum row sum) and
 *  normF denotes the Frobenius norm of a matrix (square root of sum
 *  of squares). Note that max(abs(A(i,j))) is not a consistent matrix
 *  norm.
 *
 *  WARNING: The computations are not done by this call
 *
 *******************************************************************************
 *
 * @param[in] norm
 *          = PlasmaMaxNorm: Max norm
 *          = PlasmaOneNorm: One norm
 *          = PlasmaInfNorm: Infinity norm
 *          = PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] diag
 *          = PlasmaNonUnit: Non-unit diagonal
 *          = PlasmaUnit: Unit diagonal
 *
 * @param[in] A
 *          The descriptor of the matrix A.
 *          Must be a two_dim_rectangle_cyclic matrix.
 *
 * @param[out] result
 *          The norm described above. Might not be set when the function returns.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zlantr_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlantr
 * @sa dplasma_zlantr_Destruct
 * @sa dplasma_clantr_New
 * @sa dplasma_dlantr_New
 * @sa dplasma_slantr_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zlantr_New( PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag,
                    const tiled_matrix_desc_t *A,
                    double *result )
{
    int P, Q, m, n, mb, nb, elt;
    two_dim_block_cyclic_t *Tdist;
    parsec_handle_t *parsec_zlantr = NULL;

    if ( (norm != PlasmaMaxNorm) && (norm != PlasmaOneNorm)
        && (norm != PlasmaInfNorm) && (norm != PlasmaFrobeniusNorm) ) {
        dplasma_error("dplasma_zlantr", "illegal value of norm");
        return NULL;
    }
    if ( !(A->dtype & two_dim_block_cyclic_type) ) {
        dplasma_error("dplasma_zlantr", "illegal type of descriptor for A");
        return NULL;
    }

    P = ((two_dim_block_cyclic_t*)A)->grid.rows;
    Q = ((two_dim_block_cyclic_t*)A)->grid.cols;

    /* Warning: Pb with smb/snb when mt/nt lower than P/Q */
    switch( norm ) {
    case PlasmaFrobeniusNorm:
        mb = 2;
        nb = 1;
        m  = dplasma_imax(A->mt, P);
        n  = Q;
        elt = 2;
        break;
    case PlasmaInfNorm:
        mb = A->mb;
        nb = 1;
        m  = dplasma_imax(A->mt, P);
        n  = Q;
        elt = 1;
        break;
    case PlasmaOneNorm:
        mb = 1;
        nb = A->nb;
        m  = P;
        n  = dplasma_imax(A->nt, Q);
        elt = 1;
        break;
    case PlasmaMaxNorm:
    default:
        mb = 1;
        nb = 1;
        m  = dplasma_imax(A->mt, P);
        n  = Q;
        elt = 1;
    }

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
    switch( norm ) {
    case PlasmaOneNorm:
        parsec_zlantr = (parsec_handle_t*)parsec_zlange_one_cyclic_new(
            P, Q, norm, uplo, diag, A, (parsec_ddesc_t*)Tdist, result);
        break;

    case PlasmaMaxNorm:
    case PlasmaInfNorm:
    case PlasmaFrobeniusNorm:
    default:
        parsec_zlantr = (parsec_handle_t*)parsec_zlange_frb_cyclic_new(
            P, Q, norm, uplo, diag, A, (parsec_ddesc_t*)Tdist, result);
    }

    /* Set the datatypes */
    dplasma_add2arena_tile(((parsec_zlange_frb_cyclic_handle_t*)parsec_zlantr)->arenas[PARSEC_zlange_frb_cyclic_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_double_complex_t, A->mb);
    dplasma_add2arena_rectangle(((parsec_zlange_frb_cyclic_handle_t*)parsec_zlantr)->arenas[PARSEC_zlange_frb_cyclic_COL_ARENA],
                                mb * nb * sizeof(double), PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_t, mb, nb, -1);
    dplasma_add2arena_rectangle(((parsec_zlange_frb_cyclic_handle_t*)parsec_zlantr)->arenas[PARSEC_zlange_frb_cyclic_ELT_ARENA],
                                elt * sizeof(double), PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_t, elt, 1, -1);

    return (parsec_handle_t*)parsec_zlantr;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlantr_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zlantr_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlantr_New
 * @sa dplasma_zlantr
 *
 ******************************************************************************/
void
dplasma_zlantr_Destruct( parsec_handle_t *handle )
{
    parsec_zlange_frb_cyclic_handle_t *parsec_zlantr = (parsec_zlange_frb_cyclic_handle_t *)handle;

    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)(parsec_zlantr->_g_Tdist) );
    free( parsec_zlantr->_g_Tdist );

    parsec_matrix_del2arena( parsec_zlantr->arenas[PARSEC_zlange_frb_cyclic_DEFAULT_ARENA] );
    parsec_matrix_del2arena( parsec_zlantr->arenas[PARSEC_zlange_frb_cyclic_COL_ARENA] );
    parsec_matrix_del2arena( parsec_zlantr->arenas[PARSEC_zlange_frb_cyclic_ELT_ARENA] );

    parsec_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlantr - Computes the value
 *
 *     zlantr = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
 *              (
 *              ( norm1(A),         NORM = PlasmaOneNorm
 *              (
 *              ( normI(A),         NORM = PlasmaInfNorm
 *              (
 *              ( normF(A),         NORM = PlasmaFrobeniusNorm
 *
 *  where norm1 denotes the one norm of a matrix (maximum column sum),
 *  normI denotes the infinity norm of a matrix (maximum row sum) and
 *  normF denotes the Frobenius norm of a matrix (square root of sum
 *  of squares). Note that max(abs(A(i,j))) is not a consistent matrix
 *  norm.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] norm
 *          = PlasmaMaxNorm: Max norm
 *          = PlasmaOneNorm: One norm
 *          = PlasmaInfNorm: Infinity norm
 *          = PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] diag
 *          = PlasmaNonUnit: Non-unit diagonal
 *          = PlasmaUnit: Unit diagonal
 *
 * @param[in] A
 *          The descriptor of the matrix A.
 *          Must be a two_dim_rectangle_cyclic matrix.
 *
*******************************************************************************
 *
 * @return
 *          \retval the computed norm described above.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlantr_New
 * @sa dplasma_zlantr_Destruct
 * @sa dplasma_clantr
 * @sa dplasma_dlantr
 * @sa dplasma_slantr
 *
 ******************************************************************************/
double
dplasma_zlantr( parsec_context_t *parsec,
                PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag,
                const tiled_matrix_desc_t *A)
{
    double result = 0.;
    parsec_handle_t *parsec_zlantr = NULL;

    if ( (norm != PlasmaMaxNorm) && (norm != PlasmaOneNorm)
        && (norm != PlasmaInfNorm) && (norm != PlasmaFrobeniusNorm) ) {
        dplasma_error("dplasma_zlantr", "illegal value of norm");
        return -2.;
    }
    if ( !(A->dtype & two_dim_block_cyclic_type) ) {
        dplasma_error("dplasma_zlantr", "illegal type of descriptor for A");
        return -3.;
    }

    parsec_zlantr = dplasma_zlantr_New(norm, uplo, diag, A, &result);

    if ( parsec_zlantr != NULL )
    {
        parsec_enqueue( parsec, (parsec_handle_t*)parsec_zlantr);
        dplasma_progress(parsec);
        dplasma_zlantr_Destruct( parsec_zlantr );
    }

    return result;
}

