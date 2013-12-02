/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "zungqr.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zungqr_New - Generates the dague object that computes the generation
 *  of an M-by-N matrix Q with orthonormal columns, which is defined as the
 *  first N columns of a product of K elementary reflectors of order M
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgeqrf_New().
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgeqrf_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_New() in the first k columns of its array
 *          argument A. N >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgeqrf_New().
 *
 * @param[out] Q
 *          Descriptor of the M-by-N matrix Q with orthonormal columns.
 *          M >= N >= 0.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The dague object which describes the operation to perform
 *                  NULL if one of the parameter is incorrect
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
                    tiled_matrix_desc_t *Q )
{
    dague_zungqr_object_t* object;
    int ib = T->mb;

    /* if ( !dplasma_check_desc(A) ) { */
    /*     dplasma_error("dplasma_zungqr_New", "illegal A descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(T) ) { */
    /*     dplasma_error("dplasma_zungqr_New", "illegal T descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(Q) ) { */
    /*     dplasma_error("dplasma_zungqr_New", "illegal Q descriptor"); */
    /*     return NULL; */
    /* } */
    if ( Q->n > Q->m ) {
        dplasma_error("dplasma_zungqr_New", "illegal size of Q (N should be smaller or equal to M)");
        return NULL;
    }
    if ( A->n > Q->n ) {
        dplasma_error("dplasma_zungqr_New", "illegal size of A (K should be smaller or equal to N)");
        return NULL;
    }
    if ( (T->nt < A->nt) || (T->mt < A->mt) ) {
        dplasma_error("dplasma_zungqr_New", "illegal size of T (T should have as many tiles as A)");
        return NULL;
    }

    object = dague_zungqr_new( (dague_ddesc_t*)A,
                               (dague_ddesc_t*)T,
                               (dague_ddesc_t*)Q,
                               NULL);

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * T->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zungqr_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zungqr_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zungqr_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1);

    return (dague_object_t*)object;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zungqr_Destruct - Free the data structure associated to an object
 *  created with dplasma_zungqr_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_New
 * @sa dplasma_zungqr
 *
 ******************************************************************************/
void
dplasma_zungqr_Destruct( dague_object_t *object )
{
    dague_zungqr_object_t *dague_zungqr = (dague_zungqr_object_t *)object;

    dplasma_datatype_undefine_type( &(dague_zungqr->arenas[DAGUE_zungqr_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zungqr->arenas[DAGUE_zungqr_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zungqr->arenas[DAGUE_zungqr_LITTLE_T_ARENA  ]->opaque_dtt) );

    dague_private_memory_fini( dague_zungqr->p_work );
    free( dague_zungqr->p_work );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(dague_zungqr);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zungqr - Generates of an M-by-N matrix Q with orthonormal columns,
 *  which is defined as the first N columns of a product of K elementary
 *  reflectors of order M
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgeqrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgeqrf_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_New() in the first k columns of its array
 *          argument A. N >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgeqrf_New().
 *
 * @param[out] Q
 *          Descriptor of the M-by-N matrix Q with orthonormal columns.
 *          M >= N >= 0.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_New
 * @sa dplasma_zungqr_Destruct
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

    if (dplasma_imin(Q->m, dplasma_imin(Q->n, A->n)) == 0)
        return 0;

    dague_zungqr = dplasma_zungqr_New(A, T, Q);

    if (dague_zungqr != NULL) {
        dague_enqueue(dague, (dague_object_t*)dague_zungqr);
        dplasma_progress(dague);
        dplasma_zungqr_Destruct( dague_zungqr );
    }
    return 0;
}
