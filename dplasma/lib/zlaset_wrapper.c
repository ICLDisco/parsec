/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zlaset.h"

/***************************************************************************/
/**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zlaset_New - Sets the elements of the matrix A on the diagonal
 *  to beta and on the off-diagonals to alpha
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies which elements of the matrix are to be set 
 *          = PlasmaUpper: Upper part of A is set;
 *          = PlasmaLower: Lower part of A is set;
 *          = PlasmaUpperLower: ALL elements of A are set.
 *
 * @param[in] alpha
 *         The constant to which the off-diagonal elements are to be set.
 *
 * @param[in] beta
 *         The constant to which the diagonal elements are to be set.
 *
 * @param[in,out] A
 *         On entry, the M-by-N tile A.
 *         On exit, A has been set accordingly.
 *
 **/
dague_handle_t* dplasma_zlaset_New( PLASMA_enum uplo, dague_complex64_t alpha, dague_complex64_t beta,
                                    tiled_matrix_desc_t *A )
{
    dague_zlaset_handle_t* object;
    
    object = dague_zlaset_new( uplo, alpha, beta, *A, (dague_ddesc_t*)A);

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zlaset_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    return (dague_handle_t*)object;
}

int dplasma_zlaset( dague_context_t *dague, 
                    PLASMA_enum uplo, dague_complex64_t alpha, dague_complex64_t beta,
                    tiled_matrix_desc_t *A) 
{
    dague_handle_t *dague_zlaset = NULL;

    dague_zlaset = dplasma_zlaset_New(uplo, alpha, beta, A);

    dague_enqueue(dague, (dague_handle_t*)dague_zlaset);
    dplasma_progress(dague);

    dplasma_zlaset_Destruct( dague_zlaset );
    return 0;
}

void
dplasma_zlaset_Destruct( dague_handle_t *o )
{
    dague_zlaset_handle_t *dague_zlaset = (dague_zlaset_handle_t *)o;
    dplasma_datatype_undefine_type( &(dague_zlaset->arenas[DAGUE_zlaset_DEFAULT_ARENA   ]->opaque_dtt) );
    DAGUE_INTERNAL_HANDLE_DESTRUCT(dague_zlaset);
}

