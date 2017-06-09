/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */
#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zlauum_L.h"
#include "zlauum_U.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlauum_New - Generates parsec handle to compute the product U * U' or
 *  L' * L, where the triangular factor U or L is stored in the upper or lower
 *  triangular part of the array A.
 *
 *  If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
 *  overwriting the factor U in A.
 *  If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
 *  overwriting the factor L in A.
 *
 *  WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower
 *          triangular:
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in,out] A
 *          Descriptor of the triangular matrix A of size N-by-N.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of
 *          the array A contains the upper triangular matrix, and the strictly
 *          lower triangular part of A is not referenced. If uplo = PlasmaLower,
 *          the leading N-by-N lower triangular part of the array A contains the
 *          lower triangular matrix, and the strictly upper triangular part of A
 *          is not referenced.
 *          On exit, contains the result of the computation described above.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zlauum_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlauum
 * @sa dplasma_zlauum_Destruct
 * @sa dplasma_clauum_New
 * @sa dplasma_dlauum_New
 * @sa dplasma_slauum_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zlauum_New( PLASMA_enum uplo,
                    tiled_matrix_desc_t *A )
{
    parsec_handle_t *parsec_lauum = NULL;

    if ( uplo == PlasmaLower ) {
        parsec_lauum = (parsec_handle_t*)parsec_zlauum_L_new(
            uplo, A );

        /* Lower part of A with diagonal part */
        dplasma_add2arena_lower( ((parsec_zlauum_L_handle_t*)parsec_lauum)->arenas[PARSEC_zlauum_L_LOWER_TILE_ARENA],
                                 A->mb*A->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, A->mb, 1 );
    } else {
        parsec_lauum = (parsec_handle_t*)parsec_zlauum_U_new(
            uplo, A );

        /* Upper part of A with diagonal part */
        dplasma_add2arena_upper( ((parsec_zlauum_U_handle_t*)parsec_lauum)->arenas[PARSEC_zlauum_U_UPPER_TILE_ARENA],
                                 A->mb*A->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, A->mb, 1 );
    }

    dplasma_add2arena_tile(((parsec_zlauum_L_handle_t*)parsec_lauum)->arenas[PARSEC_zlauum_L_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_double_complex_t, A->mb);

    return parsec_lauum;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlauum_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zlauum_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlauum_New
 * @sa dplasma_zlauum
 *
 ******************************************************************************/
void
dplasma_zlauum_Destruct( parsec_handle_t *handle )
{
    parsec_zlauum_L_handle_t *olauum = (parsec_zlauum_L_handle_t *)handle;

    parsec_matrix_del2arena( olauum->arenas[PARSEC_zlauum_L_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( olauum->arenas[PARSEC_zlauum_L_LOWER_TILE_ARENA] );
    parsec_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlauum - Computes the product U * U' or L' * L, where the triangular
 *  factor U or L is stored in the upper or lower triangular part of the array
 *  A.
 *
 *  If uplo = PlasmaUpper then the upper triangle of the result is stored,
 *  overwriting the factor U in A.
 *  If uplo = PlasmaLower then the lower triangle of the result is stored,
 *  overwriting the factor L in A.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower
 *          triangular:
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in,out] A
 *          Descriptor of the triangular matrix A of size N-by-N.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of
 *          the array A contains the upper triangular matrix, and the strictly
 *          lower triangular part of A is not referenced. If uplo = PlasmaLower,
 *          the leading N-by-N lower triangular part of the array A contains the
 *          lower triangular matrix, and the strictly upper triangular part of A
 *          is not referenced.
 *          On exit, contains the result of the computation described above.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 0 on success.
 *          \retval -i if the ith parameters is incorrect.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlauum_New
 * @sa dplasma_zlauum_Destruct
 * @sa dplasma_clauum
 * @sa dplasma_dlauum
 * @sa dplasma_slauum
 *
 ******************************************************************************/
int
dplasma_zlauum( parsec_context_t *parsec,
                PLASMA_enum uplo,
                tiled_matrix_desc_t *A )
{
    parsec_handle_t *parsec_zlauum = NULL;

    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_zlauum", "illegal value of uplo");
        return -1;
    }

    if ( (A->m != A->n) ) {
        dplasma_error("dplasma_zlauum", "illegal matrix A");
        return -6;
    }

    parsec_zlauum = dplasma_zlauum_New(uplo, A);

    if ( parsec_zlauum != NULL )
    {
        parsec_enqueue( parsec, parsec_zlauum );
        dplasma_wait_until_completion( parsec );
        dplasma_zlauum_Destruct( parsec_zlauum );
        return 0;
    }
    else {
        return -101;
    }
}
