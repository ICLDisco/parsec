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
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "zunglq.h"

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunglq_New - Generates the dague object to generate an
 *  M-by-N matrix Q with orthonormal rows, which is defined as the
 *  first N rows of a product of the elementary reflectors returned
 *  by dplasma_zgelqf.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of the matrix A of size K-by-N.
 *          On entry, the i-th row must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgelqf in the first k rows of its array
 *          argument A.
 *
 * @param[in] T
 *          Descriptor of the auxiliary factorization data, computed
 *          by dplasma_zgelqf.
 *
 * @param[out] Q
 *          Descriptor of the M-by-N matrix Q returned.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The dague object which describes the operation to perform
 *                  NULL if one of the parameter is incorrect
 *
 *******************************************************************************
 *
 * @sa dplasma_zunglq_Destruct
 * @sa dplasma_zunglq
 * @sa dplasma_cunglq_New
 * @sa dplasma_dorglq_New
 * @sa dplasma_sorglq_New
 * @sa dplasma_zgelqf_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zunglq_New( tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *T,
                    tiled_matrix_desc_t *Q)
{
    dague_zunglq_object_t* object;
    int ib = T->mb;

    /* if ( !dplasma_check_desc(A) ) { */
    /*     dplasma_error("dplasma_zunglq_New", "illegal A descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(T) ) { */
    /*     dplasma_error("dplasma_zunglq_New", "illegal T descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(Q) ) { */
    /*     dplasma_error("dplasma_zunglq_New", "illegal Q descriptor"); */
    /*     return NULL; */
    /* } */
    if ( Q->m > Q->n ) {
        dplasma_error("dplasma_zunglq_New", "illegal size of Q (M should be smaller or equal to N)");
        return NULL;
    }
    if ( A->m > Q->m ) {
        dplasma_error("dplasma_zunglq_New", "illegal size of A (K should be smaller or equal to M)");
        return NULL;
    }
    if ( (T->nt < A->nt) || (T->mt < A->mt) ) {
        dplasma_error("dplasma_zunglq_New", "illegal size of T (T should have as many tiles as A)");
        return NULL;
    }

    /*
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf
     */
    object = dague_zunglq_new( (dague_ddesc_t*)A,
                               (dague_ddesc_t*)T,
                               (dague_ddesc_t*)Q,
                               NULL);

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * T->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zunglq_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Upper triangular part of tile without diagonal */
    dplasma_add2arena_upper( object->arenas[DAGUE_zunglq_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zunglq_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1);

    return (dague_object_t*)object;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunglq_Destruct - Clean the data structures associated to a
 *  zunglq dague object.
 *
 *******************************************************************************
 *
 * @param[in] object
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunglq_New
 * @sa dplasma_zunglq
 * @sa dplasma_cunglq_Destruct
 * @sa dplasma_dorglq_Destruct
 * @sa dplasma_sorglq_Destruct
 *
 ******************************************************************************/
void
dplasma_zunglq_Destruct( dague_object_t *object )
{
    dague_zunglq_object_t *dague_zunglq = (dague_zunglq_object_t *)object;

    dplasma_datatype_undefine_type( &(dague_zunglq->arenas[DAGUE_zunglq_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zunglq->arenas[DAGUE_zunglq_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zunglq->arenas[DAGUE_zunglq_LITTLE_T_ARENA  ]->opaque_dtt) );

    dague_private_memory_fini( dague_zunglq->p_work );
    free( dague_zunglq->p_work );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(dague_zunglq);
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunglq - Synchronous version of dplasma_zunglq_New
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
 * @sa dplasma_zunglq_Destroy
 * @sa dplasma_zunglq_New
 * @sa dplasma_cunglq
 * @sa dplasma_dorglq
 * @sa dplasma_sorglq
 * @sa dplasma_zgelqf
 *
 ******************************************************************************/
int
dplasma_zunglq( dague_context_t *dague,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *T,
                tiled_matrix_desc_t *Q )
{
    dague_object_t *dague_zunglq = NULL;

    if (dague == NULL) {
        dplasma_error("dplasma_zunglq", "dplasma not initialized");
        return -1;
    }
    if ( Q->m > Q->n) {
        dplasma_error("dplasma_zunglq", "illegal number of rows in Q (M)");
        return -2;
    }
    if ( A->m > Q->m) {
        dplasma_error("dplasma_zunglq", "illegal number of rows in A (K)");
        return -3;
    }
    if ( A->n != Q->n ) {
        dplasma_error("dplasma_zunglq", "illegal number of columns in A");
        return -5;
    }

    if (dplasma_imin(Q->m, dplasma_imin(Q->n, A->n)) == 0)
        return 0;

    dague_zunglq = dplasma_zunglq_New(A, T, Q);

    dague_enqueue(dague, (dague_object_t*)dague_zunglq);
    dplasma_progress(dague);

    dplasma_zunglq_Destruct( dague_zunglq );

    return 0;
}
