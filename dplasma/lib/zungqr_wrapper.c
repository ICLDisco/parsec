/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "zungqr.h"

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zungqr_New - Generates the dague object to generate an
 *  M-by-N matrix Q with orthonormal columns, which is defined as the
 *  first N columns of a product of the elementary reflectors returned
 *  by dplasma_zgeqrf.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K.
 *
 * @param[in] T
 *          Descriptor of the auxiliary factorization data, computed
 *          by dplasma_zgeqrf.
 *
 * @param[out] Q
 *          Descriptor of the M-by-N matrix Q returned.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The dague object which describes the operation to perform
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_Destruct
 * @sa dplasma_zungqr
 * @sa dplasma_cungqr_New
 * @sa dplasma_dorgqr_New
 * @sa dplasma_sorgqr_New
 * @sa dplasma_zgeqrf_New
 *
 ******************************************************************************/
dague_object_t* 
dplasma_zungqr_New( tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *T,
                    tiled_matrix_desc_t *Q)
{
    dague_zungqr_object_t* object;
    int ib = T->mb;

    /* 
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf 
     */
    object = dague_zungqr_new( *A, (dague_ddesc_t*)A, 
                               *T, (dague_ddesc_t*)T, 
                               *Q, (dague_ddesc_t*)Q, 
                               ib, NULL);

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * T->nb * sizeof(Dague_Complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zungqr_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zungqr_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zungqr_LITTLE_T_ARENA], 
                                 T->mb*T->nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1);

    return (dague_object_t*)object;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zungqr_Destruct - Clean the data structures associated to a
 *  zungqr dague object.
 *
 *******************************************************************************
 *
 * @param[in] object
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_New
 * @sa dplasma_zungqr
 * @sa dplasma_cungqr_Destruct
 * @sa dplasma_dorgqr_Destruct
 * @sa dplasma_sorgqr_Destruct
 *
 ******************************************************************************/
void
dplasma_zungqr_Destruct( dague_object_t *object )
{
    dague_zungqr_object_t *dague_zungqr = (dague_zungqr_object_t *)object;

    dague_private_memory_fini( dague_zungqr->p_work );
    free( dague_zungqr->p_work );
    dague_zungqr_destroy(dague_zungqr);
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zungqr - Synchronous version of dplasma_zungqr_New
 *
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
 *******************************************************************************
 *
 * @sa dplasma_zungqr_Destroy
 * @sa dplasma_zungqr_New
 * @sa dplasma_cungqr
 * @sa dplasma_dorgqr
 * @sa dplasma_sorgqr
 * @sa dplasma_zgeqrf
 *
 ******************************************************************************/
int 
dplasma_zungqr( dague_context_t *dague, 
                tiled_matrix_desc_t *A, 
                tiled_matrix_desc_t *T, 
                tiled_matrix_desc_t *Q ) 
{
    dague_object_t *dague_zungqr = NULL;

    if (dague == NULL) {
        dplasma_error("dplasma_zungqr", "dplasma not initialized");
        return -1;
    }

    if ( Q->n > Q->m) {
        dplasma_error("dplasma_zungqr", "illegal number of columns in Q (N)");
        return -2;
    }
    if ( A->n > Q->n) {
        dplasma_error("dplasma_zungqr", "illegal number of columns in A (K)");
        return -3;
    }
    if ( A->m != Q->m ) {
        dplasma_error("dplasma_zungqr", "illegal number of rows in A");
        return -5;
    }

    if (imin(Q->m, imin(Q->n, A->n)) == 0)
        return 0;

    dague_zungqr = dplasma_zungqr_New(A, T, Q);

    dague_enqueue(dague, (dague_object_t*)dague_zungqr);
    dplasma_progress(dague);

    dplasma_zungqr_Destruct( dague_zungqr );

    return 0;
}
