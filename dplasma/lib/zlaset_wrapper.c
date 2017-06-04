/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */

#include <lapacke.h>
#include "parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map.h"

static int
dplasma_zlaset_operator( parsec_execution_unit_t *eu,
                         const tiled_matrix_desc_t *descA,
                         void *_A,
                         PLASMA_enum uplo, int m, int n,
                         void *args )
{
    int tempmm, tempnn, ldam;
    parsec_complex64_t *alpha = (parsec_complex64_t*)args;
    parsec_complex64_t *A = (parsec_complex64_t*)_A;
    (void)eu;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( descA, m );

    if (m == n) {
        LAPACKE_zlaset_work(
            LAPACK_COL_MAJOR, lapack_const( uplo ), tempmm, tempnn,
            alpha[0], alpha[1], A, ldam);
    } else {
        LAPACKE_zlaset_work(
            LAPACK_COL_MAJOR, 'A', tempmm, tempnn,
            alpha[0], alpha[0], A, ldam);
    }
    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zlaset_New - Generates the handle that set the elements of the matrix
 * A on the diagonal to beta and the off-diagonals eklements to alpha.
 *
 * See dplasma_map_New() for further information.
 *
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is set:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is referenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *
 * @param[in] alpha
 *         The constant to which the off-diagonal elements are to be set.
 *
 * @param[in] beta
 *         The constant to which the diagonal elements are to be set.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A. Any tiled matrix
 *          descriptor can be used.
 *          On exit, A has been set accordingly.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zlaset_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlaset
 * @sa dplasma_zlaset_Destruct
 * @sa dplasma_claset_New
 * @sa dplasma_dlaset_New
 * @sa dplasma_slaset_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zlaset_New( PLASMA_enum uplo,
                    parsec_complex64_t alpha,
                    parsec_complex64_t beta,
                    tiled_matrix_desc_t *A )
{
    parsec_complex64_t *params = (parsec_complex64_t*)malloc(2 * sizeof(parsec_complex64_t));

    params[0] = alpha;
    params[1] = beta;

    return dplasma_map_New( uplo, A, dplasma_zlaset_operator, params );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlaset_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zlaset_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlaset_New
 * @sa dplasma_zlaset
 *
 ******************************************************************************/
void
dplasma_zlaset_Destruct( parsec_handle_t *handle )
{
    dplasma_map_Destruct(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zlaset - Set the elements of the matrix
 * A on the diagonal to beta and the off-diagonals eklements to alpha.
 *
 * See dplasma_map() for further information.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is set:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is refrenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *
 * @param[in] alpha
 *         The constant to which the off-diagonal elements are to be set.
 *
 * @param[in] beta
 *         The constant to which the diagonal elements are to be set.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A. Any tiled matrix
 *          descriptor can be used.
 *          On exit, A has been set accordingly.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlaset_New
 * @sa dplasma_zlaset_Destruct
 * @sa dplasma_claset
 * @sa dplasma_dlaset
 * @sa dplasma_slaset
 *
 ******************************************************************************/
int
dplasma_zlaset( parsec_context_t *parsec,
                PLASMA_enum uplo,
                parsec_complex64_t alpha,
                parsec_complex64_t beta,
                tiled_matrix_desc_t *A )
{
    parsec_handle_t *parsec_zlaset = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        dplasma_error("dplasma_zlaset", "illegal value of type");
        return -1;
    }

    parsec_zlaset = dplasma_zlaset_New(uplo, alpha, beta, A);

    if ( parsec_zlaset != NULL ) {
        parsec_enqueue(parsec, (parsec_handle_t*)parsec_zlaset);
        dplasma_progress(parsec);
        dplasma_zlaset_Destruct( parsec_zlaset );
    }
    return 0;
}
