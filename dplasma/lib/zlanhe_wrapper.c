/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> z c
 *
 */

#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h"

#include "zlansy.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlanhe_New - Generates the taskpool that computes the value
 *
 *     zlanhe = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
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
 * @param[in] A
 *          The descriptor of the hermitian matrix A.
 *          Must be a two_dim_rectangle_cyclic or sym_two_dim_rectangle_cyclic
 *          matrix
 *
 * @param[out] result
 *          The norm described above. Might not be set when the function returns.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zlanhe_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlanhe
 * @sa dplasma_zlanhe_Destruct
 * @sa dplasma_clanhe_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zlanhe_New( PLASMA_enum norm,
                    PLASMA_enum uplo,
                    const parsec_tiled_matrix_dc_t *A,
                    double *result )
{
    int P, Q, mb, nb, elt, m;
    two_dim_block_cyclic_t *Tdist;
    parsec_taskpool_t *parsec_zlanhe = NULL;

    if ( (norm != PlasmaMaxNorm) && (norm != PlasmaOneNorm)
        && (norm != PlasmaInfNorm) && (norm != PlasmaFrobeniusNorm) ) {
        dplasma_error("dplasma_zlanhe", "illegal value of norm");
        return NULL;
    }
    if ( (uplo != PlasmaUpper) && (uplo != PlasmaLower) ) {
        dplasma_error("dplasma_zlanhe", "illegal value of uplo");
        return NULL;
    }
    if ( !(A->dtype & ( two_dim_block_cyclic_type | sym_two_dim_block_cyclic_type)) ) {
        dplasma_error("dplasma_zlanhe", "illegal type of descriptor for A");
        return NULL;
    }

    P = ((sym_two_dim_block_cyclic_t*)A)->grid.rows;
    Q = ((sym_two_dim_block_cyclic_t*)A)->grid.cols;

    /* Warning: Pb with smb/snb when mt/nt lower than P/Q */
    switch( norm ) {
    case PlasmaFrobeniusNorm:
        mb = 2;
        nb = 1;
        elt = 2;
        break;
    case PlasmaInfNorm:
    case PlasmaOneNorm:
        mb = A->mb;
        nb = 1;
        elt = 1;
        break;
    case PlasmaMaxNorm:
    default:
        mb = 1;
        nb = 1;
        elt = 1;
    }
    m = dplasma_imax(A->mt, P);

    /* Create a copy of the A matrix to be used as a data distribution metric.
     * As it is used as a NULL value we must have a data_copy and a data associated
     * with it, so we can create them here.
     * Create the task distribution */
    Tdist = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));

    two_dim_block_cyclic_init(
        Tdist, matrix_RealDouble, matrix_Tile,
        A->super.nodes, A->super.myrank,
        1, 1,   /* Dimensions of the tiles              */
        m, P*Q, /* Dimensions of the matrix             */
        0, 0,   /* Starting points (not important here) */
        m, P*Q, /* Dimensions of the submatrix          */
        1, 1, P);
    Tdist->super.super.data_of = fake_data_of;

    /* Create the DAG */
    parsec_zlanhe = (parsec_taskpool_t*)parsec_zlansy_new(
        P, Q, norm, uplo, PlasmaConjTrans,
        A, (parsec_data_collection_t*)Tdist,
        result);

    /* Set the datatypes */
    dplasma_add2arena_tile(((parsec_zlansy_taskpool_t*)parsec_zlanhe)->arenas[PARSEC_zlansy_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_double_complex_t, A->mb);
    dplasma_add2arena_rectangle(((parsec_zlansy_taskpool_t*)parsec_zlanhe)->arenas[PARSEC_zlansy_COL_ARENA],
                                mb * nb * sizeof(double), PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_t, mb, nb, -1);
    dplasma_add2arena_rectangle(((parsec_zlansy_taskpool_t*)parsec_zlanhe)->arenas[PARSEC_zlansy_ELT_ARENA],
                                elt * sizeof(double), PARSEC_ARENA_ALIGNMENT_SSE,
                                parsec_datatype_double_t, elt, 1, -1);

    return (parsec_taskpool_t*)parsec_zlanhe;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlanhe_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zlanhe_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlanhe_New
 * @sa dplasma_zlanhe
 *
 ******************************************************************************/
void
dplasma_zlanhe_Destruct( parsec_taskpool_t *tp )
{
    parsec_zlansy_taskpool_t *parsec_zlanhe = (parsec_zlansy_taskpool_t *)tp;

    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)(parsec_zlanhe->_g_Tdist) );
    free( parsec_zlanhe->_g_Tdist );

    parsec_matrix_del2arena( parsec_zlanhe->arenas[PARSEC_zlansy_DEFAULT_ARENA] );
    parsec_matrix_del2arena( parsec_zlanhe->arenas[PARSEC_zlansy_COL_ARENA] );
    parsec_matrix_del2arena( parsec_zlanhe->arenas[PARSEC_zlansy_ELT_ARENA] );

    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlanhe - Computes the value
 *
 *     zlanhe = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
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
 * @param[in] A
 *          The descriptor of the hermitian matrix A.
 *          Must be a two_dim_rectangle_cyclic or sym_two_dim_rectangle_cyclic
 *          matrix
 *
*******************************************************************************
 *
 * @return
 *          \retval the computed norm described above.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlanhe_New
 * @sa dplasma_zlanhe_Destruct
 * @sa dplasma_clanhe
 *
 ******************************************************************************/
double
dplasma_zlanhe( parsec_context_t *parsec,
                PLASMA_enum norm,
                PLASMA_enum uplo,
                const parsec_tiled_matrix_dc_t *A)
{
    double result = 0.;
    parsec_taskpool_t *parsec_zlanhe = NULL;

    if ( (norm != PlasmaMaxNorm) && (norm != PlasmaOneNorm)
        && (norm != PlasmaInfNorm) && (norm != PlasmaFrobeniusNorm) ) {
        dplasma_error("dplasma_zlanhe", "illegal value of norm");
        return -2.;
    }
    if ( (uplo != PlasmaUpper) && (uplo != PlasmaLower) ) {
        dplasma_error("dplasma_zlanhe", "illegal value of uplo");
        return -3.;
    }
    if ( !(A->dtype & ( two_dim_block_cyclic_type | sym_two_dim_block_cyclic_type)) ) {
        dplasma_error("dplasma_zlanhe", "illegal type of descriptor for A");
        return -4.;
    }
    if ( A->m != A->n ) {
        dplasma_error("dplasma_zlanhe", "illegal matrix A (not square)");
        return -5.;
    }

    parsec_zlanhe = dplasma_zlanhe_New(norm, uplo, A, &result);

    if ( parsec_zlanhe != NULL )
    {
        parsec_enqueue( parsec, (parsec_taskpool_t*)parsec_zlanhe);
        dplasma_wait_until_completion(parsec);
        dplasma_zlanhe_Destruct( parsec_zlanhe );
    }

    return result;
}

