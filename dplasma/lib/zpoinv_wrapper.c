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
#include "dplasma/lib/dplasmaaux.h"

#include "zpoinv_U.h"
#include "zpoinv_L.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpoinv_New - Generates the handle that computes the inverse of an
 * hermitian matrix through Cholesky factorization and inversion.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten by the inverse of A,
 *          A^(-1)
 *
 * @param[out] info
 *          Address where to store the output information of the factorization,
 *          this is not synchronized between the nodes, and might not be set
 *          when function exists.
 *          On DAG completion:
 *              - info = 0 on all nodes if successful.
 *              - info > 0 if the leading minor of order i of A is not positive
 *                definite, so the factorization could not be completed, and the
 *                solution has not been computed. Info will be equal to i on the
 *                node that owns the diagonal element (i,i), and 0 on all other
 *                nodes.
 *         TODO: support this correctly as it is made in Cholesky
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zpoinv_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zpoinv
 * @sa dplasma_zpoinv_Destruct
 * @sa dplasma_cpoinv_New
 * @sa dplasma_dpoinv_New
 * @sa dplasma_spoinv_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zpoinv_New( PLASMA_enum uplo,
                    tiled_matrix_desc_t *A,
                    int *info )
{
    parsec_zpoinv_L_handle_t *parsec_zpoinv = NULL;
    parsec_handle_t *handle = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaUpper) && (uplo != PlasmaLower)) {
        dplasma_error("dplasma_zpoinv_New", "illegal value of uplo");
        return NULL /*-1*/;
    }

    *info = 0;
    if ( uplo == PlasmaUpper ) {
        handle = (parsec_handle_t*)parsec_zpoinv_U_new( A /*, info */);

        /* Upper part of A with diagonal part */
        /* dplasma_add2arena_upper( ((parsec_zpoinv_U_handle_t*)parsec_poinv)->arenas[PARSEC_zpoinv_U_UPPER_TILE_ARENA], */
        /*                          A->mb*A->nb*sizeof(parsec_complex64_t), */
        /*                          PARSEC_ARENA_ALIGNMENT_SSE, */
        /*                          parsec_datatype_double_complex_t, A->mb, 1 ); */
    } else {
        handle = (parsec_handle_t*)parsec_zpoinv_L_new( A /*, info */);

        /* Lower part of A with diagonal part */
        /* dplasma_add2arena_lower( ((parsec_zpoinv_L_handle_t*)parsec_poinv)->arenas[PARSEC_zpoinv_L_LOWER_TILE_ARENA], */
        /*                          A->mb*A->nb*sizeof(parsec_complex64_t), */
        /*                          PARSEC_ARENA_ALIGNMENT_SSE, */
        /*                          parsec_datatype_double_complex_t, A->mb, 1 ); */
    }

    parsec_zpoinv = (parsec_zpoinv_L_handle_t*)handle;

    dplasma_add2arena_tile( parsec_zpoinv->arenas[PARSEC_zpoinv_L_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    return handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zpoinv_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zpoinv_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpoinv_New
 * @sa dplasma_zpoinv
 *
 ******************************************************************************/
void
dplasma_zpoinv_Destruct( parsec_handle_t *handle )
{
    parsec_zpoinv_L_handle_t *parsec_zpoinv = (parsec_zpoinv_L_handle_t *)handle;

    parsec_matrix_del2arena( parsec_zpoinv->arenas[PARSEC_zpoinv_L_DEFAULT_ARENA   ] );
    /* parsec_matrix_del2arena( parsec_zpoinv->arenas[PARSEC_zpoinv_L_LOWER_TILE_ARENA] ); */
    parsec_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpoinv - Computes the matrix inverse of an hermitian matrix through
 * Cholesky factorization and inversion.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with inverse of A.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval > 0 if the leading minor of order i of A is not positive
 *          definite, so the factorization could not be completed, and the
 *          solution has not been computed. Info will be equal to i on the node
 *          that owns the diagonal element (i,i), and 0 on all other nodes.
 *         TODO: support this correctly as it is made in Cholesky
 *
 *******************************************************************************
 *
 * @sa dplasma_zpoinv_New
 * @sa dplasma_zpoinv_Destruct
 * @sa dplasma_cpoinv
 * @sa dplasma_dpoinv
 * @sa dplasma_spoinv
 *
 ******************************************************************************/
int
dplasma_zpoinv( parsec_context_t *parsec,
                PLASMA_enum uplo,
                tiled_matrix_desc_t *A )
{
    parsec_handle_t *parsec_zpoinv = NULL;
    int info = 0, ginfo = 0 ;

    parsec_zpoinv = dplasma_zpoinv_New( uplo, A, &info );

    if ( parsec_zpoinv != NULL )
    {
        parsec_enqueue( parsec, (parsec_handle_t*)parsec_zpoinv);
        dplasma_progress(parsec);
        dplasma_zpoinv_Destruct( parsec_zpoinv );
    }

#if defined(PARSEC_HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpoinv_sync - Computes the matrix inverse of an hermitian matrix
 * through Cholesky factorization and inversion as in dplasma_zpoinv. The
 * difference is in the fact that it calls successively three different DAGs
 * with intermediate synchronizations.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with inverse of A.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval > 0 if the leading minor of order i of A is not positive
 *          definite, so the factorization could not be completed, and the
 *          solution has not been computed. Info will be equal to i on the node
 *          that owns the diagonal element (i,i), and 0 on all other nodes.
 *         TODO: support this correctly as it is made in Cholesky
 *
 *******************************************************************************
 *
 * @sa dplasma_zpoinv_New
 * @sa dplasma_zpoinv_Destruct
 * @sa dplasma_cpoinv
 * @sa dplasma_dpoinv
 * @sa dplasma_spoinv
 *
 ******************************************************************************/
int
dplasma_zpoinv_sync( parsec_context_t *parsec,
                     PLASMA_enum uplo,
                     tiled_matrix_desc_t* A )
{
    int info = 0;
    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_zpoinv_sync", "illegal value of uplo");
        return -1;
    }

    info = dplasma_zpotrf( parsec, uplo, A );
    info = dplasma_ztrtri( parsec, uplo, PlasmaNonUnit, A );
    dplasma_zlauum( parsec, uplo, A );

    return info;
}

