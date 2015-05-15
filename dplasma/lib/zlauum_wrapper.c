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

#include "zlauum_L.h"
#include "zlauum_U.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlauum_New - Generates dague object to compute the product U * U' or
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
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
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
dague_handle_t*
dplasma_zlauum_New( PLASMA_enum uplo,
                    tiled_matrix_desc_t *A )
{
    dague_handle_t *dague_lauum = NULL;

    if ( uplo == PlasmaLower ) {
        dague_lauum = (dague_handle_t*)dague_zlauum_L_new(
            uplo, (dague_ddesc_t*)A );

        /* Lower part of A with diagonal part */
        dplasma_add2arena_lower( ((dague_zlauum_L_handle_t*)dague_lauum)->arenas[DAGUE_zlauum_L_LOWER_TILE_ARENA],
                                 A->mb*A->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, A->mb, 1 );
    } else {
        dague_lauum = (dague_handle_t*)dague_zlauum_U_new(
            uplo, (dague_ddesc_t*)A );

        /* Lower part of A with diagonal part */
        dplasma_add2arena_upper( ((dague_zlauum_U_handle_t*)dague_lauum)->arenas[DAGUE_zlauum_U_UPPER_TILE_ARENA],
                                 A->mb*A->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, A->mb, 1 );
    }

    dplasma_add2arena_tile(((dague_zlauum_L_handle_t*)dague_lauum)->arenas[DAGUE_zlauum_L_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_lauum;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlauum_Destruct - Free the data structure associated to an object
 *  created with dplasma_zlauum_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlauum_New
 * @sa dplasma_zlauum
 *
 ******************************************************************************/
void
dplasma_zlauum_Destruct( dague_handle_t *o )
{
    dague_zlauum_L_handle_t *olauum = (dague_zlauum_L_handle_t *)o;

    dplasma_datatype_undefine_type( &(olauum->arenas[DAGUE_zlauum_L_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(olauum->arenas[DAGUE_zlauum_L_LOWER_TILE_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
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
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
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
dplasma_zlauum( dague_context_t *dague,
                PLASMA_enum uplo,
                tiled_matrix_desc_t *A )
{
    dague_handle_t *dague_zlauum = NULL;

    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_zlauum", "illegal value of uplo");
        return -1;
    }

    if ( (A->m != A->n) ) {
        dplasma_error("dplasma_zlauum", "illegal matrix A");
        return -6;
    }

    dague_zlauum = dplasma_zlauum_New(uplo, A);

    if ( dague_zlauum != NULL )
    {
        dague_enqueue( dague, dague_zlauum );
        dplasma_progress( dague );
        dplasma_zlauum_Destruct( dague_zlauum );
        return 0;
    }
    else {
        return -101;
    }
}
