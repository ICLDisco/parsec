/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> c d s
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zlascal.h"

/***************************************************************************//**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zlascal_New - Scale a matrix by a given scalar.
 *  WARNING: This routine is equivalment to the pzlascal routine from ScaLapack
 *  and not to the zlascl/pzlascl routines from Lapack/ScaLapack.
 *
 *******************************************************************************
 *
 * @param[in] Type
 *          Specifies the type of the input matrix as follows:
 *          = PlasmaUpperLower: A is a full matrix.
 *          = PlasmaUpper: A is an upper triangular matrix.
 *          = PlasmaLower: A is a lower triangular matrix.
 *          = PlasmaUpperHessenberg: A is an upper Hessenberg matrix (Not
 *          supported)
 *
 * @param[in] alpha
 *          The scalatr to use to scale the matrix.
 *
 * @param[in,out] A
 *          The descriptor of the matrix to scale.
 *
 ******************************************************************************/
dague_handle_t* dplasma_zlascal_New( PLASMA_enum type,
                                     dague_complex64_t alpha,
                                     tiled_matrix_desc_t *A )
{
    dague_zlascal_handle_t* object;

    object = dague_zlascal_new( type, alpha, (dague_ddesc_t*)A);

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zlascal_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    return (dague_handle_t*)object;
}

int dplasma_zlascal( dague_context_t *dague,
                     PLASMA_enum type, dague_complex64_t alpha,
                     tiled_matrix_desc_t *A)
{
    dague_handle_t *dague_zlascal = NULL;

    /* Check input arguments */
    if ((type != PlasmaLower) &&
        (type != PlasmaUpper) &&
        (type != PlasmaUpperLower))
    {
        dplasma_error("dplasma_zlascal", "illegal value of type");
        return -1;
    }

    dague_zlascal = dplasma_zlascal_New(type, alpha, A);

    dague_enqueue(dague, (dague_handle_t*)dague_zlascal);
    dplasma_progress(dague);

    dplasma_zlascal_Destruct( dague_zlascal );
    return 0;
}

void
dplasma_zlascal_Destruct( dague_handle_t *o )
{
    dague_zlascal_handle_t *dague_zlascal = (dague_zlascal_handle_t *)o;
    dplasma_datatype_undefine_type( &(dague_zlascal->arenas[DAGUE_zlascal_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_HANDLE_DESTRUCT(dague_zlascal);
}
