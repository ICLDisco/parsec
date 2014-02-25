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

#include "ztrtri_L.h"
#include "ztrtri_U.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrtri_New - Generates dague object to compute the inverse of an
 *  upper or lower triangular matrix A.
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
 * @param[in] diag
 *          Specifies whether or not A is unit triangular:
 *          = PlasmaNonUnit: A is non unit;
 *          = PlasmaUnit:    A us unit.
 *
 * @param[in,out] A
 *          Descriptor of the triangular matrix A of size N-by-N.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of
 *          the array A contains the upper triangular matrix, and the strictly
 *          lower triangular part of A is not referenced. If uplo = PlasmaLower,
 *          the leading N-by-N lower triangular part of the array A contains the
 *          lower triangular matrix, and the strictly upper triangular part of A
 *          is not referenced. If diag = PlasmaUnit, the diagonal elements of A
 *          are also not referenced and are assumed to be 1.
 *          On exit, the (triangular) inverse of the original matrix, in the
 *          same storage format.
 *
 * @param[out] INFO
 *          On algorithm completion if INFO is > 0, A(i,i) is exactly zero.  The
 *          triangular matrix is singular and its inverse can not be computed.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_ztrtri_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrtri
 * @sa dplasma_ztrtri_Destruct
 * @sa dplasma_ctrtri_New
 * @sa dplasma_dtrtri_New
 * @sa dplasma_strtri_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_ztrtri_New( PLASMA_enum uplo,
                    PLASMA_enum diag,
                    tiled_matrix_desc_t *A,
                    int *INFO )
{
    dague_handle_t *dague_trtri = NULL;

    if ( uplo == PlasmaLower ) {
        dague_trtri = (dague_handle_t*)dague_ztrtri_L_new(
            uplo, diag, (dague_ddesc_t*)A, INFO );

        /* Lower part of A with diagonal part */
        dplasma_add2arena_lower( ((dague_ztrtri_L_handle_t*)dague_trtri)->arenas[DAGUE_ztrtri_L_LOWER_TILE_ARENA],
                                 A->mb*A->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, A->mb, 1 );
    } else {
        dague_trtri = (dague_handle_t*)dague_ztrtri_U_new(
            uplo, diag, (dague_ddesc_t*)A, INFO );

        /* Lower part of A with diagonal part */
        dplasma_add2arena_upper( ((dague_ztrtri_U_handle_t*)dague_trtri)->arenas[DAGUE_ztrtri_U_UPPER_TILE_ARENA],
                                 A->mb*A->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, A->mb, 1 );
    }

    dplasma_add2arena_tile(((dague_ztrtri_L_handle_t*)dague_trtri)->arenas[DAGUE_ztrtri_L_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_trtri;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrtri_Destruct - Free the data structure associated to an object
 *  created with dplasma_ztrtri_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrtri_New
 * @sa dplasma_ztrtri
 *
 ******************************************************************************/
void
dplasma_ztrtri_Destruct( dague_handle_t *o )
{
    dague_ztrtri_L_handle_t *otrtri = (dague_ztrtri_L_handle_t *)o;

    dplasma_datatype_undefine_type( &(otrtri->arenas[DAGUE_ztrtri_L_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(otrtri->arenas[DAGUE_ztrtri_L_LOWER_TILE_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrtri - Computes the inverse of an upper or lower triangular matrix
 *  A.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower
 *          triangular:
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] diag
 *          Specifies whether or not A is unit triangular:
 *          = PlasmaNonUnit: A is non unit;
 *          = PlasmaUnit:    A us unit.
 *
 * @param[in,out] A
 *          Descriptor of the triangular matrix A of size N-by-N.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of
 *          the array A contains the upper triangular matrix, and the strictly
 *          lower triangular part of A is not referenced. If uplo = PlasmaLower,
 *          the leading N-by-N lower triangular part of the array A contains the
 *          lower triangular matrix, and the strictly upper triangular part of A
 *          is not referenced. If diag = PlasmaUnit, the diagonal elements of A
 *          are also not referenced and are assumed to be 1.
 *          On exit, the (triangular) inverse of the original matrix, in the
 *          same storage format.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval i if A(i,i) is exactly zero.  The triangular matrix is
 *          singular and its inverse can not be computed.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrtri_New
 * @sa dplasma_ztrtri_Destruct
 * @sa dplasma_ctrtri
 * @sa dplasma_dtrtri
 * @sa dplasma_strtri
 *
 ******************************************************************************/
int
dplasma_ztrtri( dague_context_t *dague,
                PLASMA_enum uplo,
                PLASMA_enum diag,
                tiled_matrix_desc_t *A )
{
    dague_handle_t *dague_ztrtri = NULL;
    int info = 0;

    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_ztrtri", "illegal value of uplo");
        return -1;
    }
    if (diag != PlasmaUnit && diag != PlasmaNonUnit) {
        dplasma_error("dplasma_ztrtri", "illegal value of diag");
        return -2;
    }

    if ( (A->m != A->n) ) {
        dplasma_error("dplasma_ztrtri", "illegal matrix A");
        return -6;
    }

    dague_ztrtri = dplasma_ztrtri_New(uplo, diag, A, &info);

    if ( dague_ztrtri != NULL )
    {
        dague_enqueue( dague, dague_ztrtri );
        dplasma_progress( dague );
        dplasma_ztrtri_Destruct( dague_ztrtri );
        return info;
    }
    else {
        return -101;
    }
}
