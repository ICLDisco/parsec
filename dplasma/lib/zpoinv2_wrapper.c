/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zpoinv2_U.h"
//#include "zpoinv2_L.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpoinv2_New - Generates the object that computes the inverse of an
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
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zpoinv2_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zpoinv2
 * @sa dplasma_zpoinv2_Destruct
 * @sa dplasma_cpoinv2_New
 * @sa dplasma_dpoinv2_New
 * @sa dplasma_spoinv2_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zpoinv2_New( PLASMA_enum uplo,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *B,
                    tiled_matrix_desc_t *C,
                    int *info )
{
    dague_zpoinv2_U_handle_t *dague_zpoinv2 = NULL;
    dague_handle_t *o = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaUpper) && (uplo != PlasmaLower)) {
        dplasma_error("dplasma_zpoinv2_New", "illegal value of uplo");
        return NULL /*-1*/;
    }

    *info = 0;
    if ( uplo == PlasmaUpper ) {
        o = (dague_handle_t*)dague_zpoinv2_U_new( uplo, (dague_ddesc_t*)A, (dague_ddesc_t*)B, (dague_ddesc_t*)C /*, info */);

        /* Upper part of A with diagonal part */
        /* dplasma_add2arena_upper( ((dague_zpoinv2_U_handle_t*)dague_poinv2)->arenas[DAGUE_zpoinv2_U_UPPER_TILE_ARENA], */
        /*                          A->mb*A->nb*sizeof(dague_complex64_t), */
        /*                          DAGUE_ARENA_ALIGNMENT_SSE, */
        /*                          MPI_DOUBLE_COMPLEX, A->mb, 1 ); */
    } else {
        //o = (dague_handle_t*)dague_zpoinv2_L_new( uplo, (dague_ddesc_t*)A /*, info */);

        /* Lower part of A with diagonal part */
        /* dplasma_add2arena_lower( ((dague_zpoinv2_L_handle_t*)dague_poinv2)->arenas[DAGUE_zpoinv2_L_LOWER_TILE_ARENA], */
        /*                          A->mb*A->nb*sizeof(dague_complex64_t), */
        /*                          DAGUE_ARENA_ALIGNMENT_SSE, */
        /*                          MPI_DOUBLE_COMPLEX, A->mb, 1 ); */
    }

    dague_zpoinv2 = (dague_zpoinv2_U_handle_t*)o;

    dplasma_add2arena_tile( dague_zpoinv2->arenas[DAGUE_zpoinv2_U_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    return o;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zpoinv2_Destruct - Free the data structure associated to an object
 *  created with dplasma_zpoinv2_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpoinv2_New
 * @sa dplasma_zpoinv2
 *
 ******************************************************************************/
void
dplasma_zpoinv2_Destruct( dague_handle_t *o )
{
    dague_zpoinv2_U_handle_t *dague_zpoinv2 = (dague_zpoinv2_U_handle_t *)o;

    dplasma_datatype_undefine_type( &(dague_zpoinv2->arenas[DAGUE_zpoinv2_U_DEFAULT_ARENA   ]->opaque_dtt) );
    /* dplasma_datatype_undefine_type( &(dague_zpoinv2->arenas[DAGUE_zpoinv2_U_LOWER_TILE_ARENA]->opaque_dtt) ); */
    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpoinv2 - Computes the matrix inverse of an hermitian matrix through
 * Cholesky factorization and inversion.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
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
 * @sa dplasma_zpoinv2_New
 * @sa dplasma_zpoinv2_Destruct
 * @sa dplasma_cpoinv2
 * @sa dplasma_dpoinv2
 * @sa dplasma_spoinv2
 *
 ******************************************************************************/
int
dplasma_zpoinv2( dague_context_t *dague,
                PLASMA_enum uplo,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *B,
                tiled_matrix_desc_t *C )
{
    dague_handle_t *dague_zpoinv2 = NULL;
    int info = 0, ginfo = 0 ;

    dague_zpoinv2 = dplasma_zpoinv2_New( uplo, A, B, C, &info );

    if ( dague_zpoinv2 != NULL )
    {
        dague_enqueue( dague, (dague_handle_t*)dague_zpoinv2);
        dplasma_progress(dague);
        dplasma_zpoinv2_Destruct( dague_zpoinv2 );
    }

#if defined(HAVE_MPI)
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
 * dplasma_zpoinv2_sync - Computes the matrix inverse of an hermitian matrix
 * through Cholesky factorization and inversion as in dplasma_zpoinv2. The
 * difference is in the fact that it calls successively three different DAGs
 * with intermediate synchronizations.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
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
 * @sa dplasma_zpoinv2_New
 * @sa dplasma_zpoinv2_Destruct
 * @sa dplasma_cpoinv2
 * @sa dplasma_dpoinv2
 * @sa dplasma_spoinv2
 *
 ******************************************************************************/
int
dplasma_zpoinv2_sync( dague_context_t *dague,
                      PLASMA_enum uplo,
                      tiled_matrix_desc_t* A,
                      tiled_matrix_desc_t* B,
                      tiled_matrix_desc_t* C )
{
    int info = 0;
    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_zpoinv2_sync", "illegal value of uplo");
        return -1;
    }

    info = dplasma_zpotrf( dague, uplo, A );
    info = dplasma_ztrtri( dague, uplo, PlasmaNonUnit, A );
    dplasma_zlauum( dague, uplo, A );
    dplasma_zhemm( dague, PlasmaRight, uplo, 1.0, A, B, 0., C );

    return info;
}

