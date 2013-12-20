/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "dague_internal.h"
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/dplasmatypes.h"

#include "ztrdsm.h"

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_ztrdsm_New -
 *
 *******************************************************************************
 *
 * @param[in] A
 * @param[in,out] B
 *
 *******************************************************************************
 *
 * @return
 *          \retval 0 successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
dague_object_t*
dplasma_ztrdsm_New(const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B )
{
    dague_object_t *dague_trdsm = NULL;

    dague_trdsm = (dague_object_t*)dague_ztrdsm_new( *A, (dague_ddesc_t*)A,
                                                     *B, (dague_ddesc_t*)B );

    dplasma_add2arena_tile(((dague_ztrdsm_object_t*)dague_trdsm)->arenas[DAGUE_ztrdsm_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_trdsm;
}

/***************************************************************************//**
 *
 * @param[in] o
 *          Object to destroy.
 ******************************************************************************/
void
dplasma_ztrdsm_Destruct( dague_object_t *o )
{
    dague_ztrdsm_object_t *otrdsm = (dague_ztrdsm_object_t *)o;
    dplasma_datatype_undefine_type( &(otrdsm->arenas[DAGUE_ztrdsm_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

/***************************************************************************//**
 *  dplasma_ztrdsm - Blocking version of dplasma_ztrdsm_New
 *******************************************************************************
 *
 * @param[in] dague
 *          Dague context to which submit the DAG object.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 0 if success
 *          \retval < 0 if one of the parameter had an illegal value.
 *
 ******************************************************************************/
int
dplasma_ztrdsm( dague_context_t *dague,
                const tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *B)
{
    dague_object_t *dague_ztrdsm = NULL;

    dague_ztrdsm = dplasma_ztrdsm_New(A, B);

    if ( dague_ztrdsm != NULL ) {
        dague_enqueue( dague, dague_ztrdsm );
        dplasma_progress( dague );

        dplasma_ztrdsm_Destruct( dague_ztrdsm );
    }

    return 0;
}
