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

#include "zgelqf.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zgelqf_New - Generates the object that computes the LQ factorization
 * a complex M-by-N matrix A: A = L * Q.
 *
 * The method used in this algorithm is a tile LQ algorithm with a flat
 * reduction tree. It is recommended to use the super tiling parameter (SNB) to
 * improve the performance of the factorization.
 * A high SNB parameter reduces the communication volume, but also deteriorates
 * the load balancing if too important. A small one increases the communication
 * volume, but improves load balancing.
 * A good SNB value should provide enough work to all available cores on one
 * node. It is then recommended to set it to 4 when creating the matrix
 * descriptor.
 * For tiling, MB=200, and IB=32 usually give good results.
 *
 * This variant is good for square large problems.
 * For other problems, see:
 *   - dplasma_zgelqf_param_New() parameterized with trees for tall and skinny
 *     matrices
 *   - dplasma_zgelqf_param_New() parameterized with systolic tree if
 *     computation load per node is very low.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and below the diagonal of the array contain
 *          the M-by-min(M,N) lower trapezoidal matrix L (L is lower triangular
 *          if (N >= M); the elements above the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile LQ algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          On exit, contains auxiliary information required to compute the Q
 *          matrix, and/or solve the problem.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgelqf_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgelqf
 * @sa dplasma_zgelqf_Destruct
 * @sa dplasma_cgelqf_New
 * @sa dplasma_dgelqf_New
 * @sa dplasma_sgelqf_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zgelqf_New( tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *T )
{
    dague_zgelqf_handle_t* object;
    int ib = T->mb;

    object = dague_zgelqf_new( (dague_ddesc_t*)A,
                               (dague_ddesc_t*)T,
                               ib, NULL, NULL );

    object->p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_tau, T->nb * sizeof(dague_complex64_t) );

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * T->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zgelqf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Lower triangular part of tile with diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zgelqf_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* Upper triangular part of tile without diagonal */
    dplasma_add2arena_upper( object->arenas[DAGUE_zgelqf_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgelqf_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1);

    return (dague_handle_t*)object;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zgelqf_Destruct - Free the data structure associated to an object
 *  created with dplasma_zgelqf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgelqf_New
 * @sa dplasma_zgelqf
 *
 ******************************************************************************/
void
dplasma_zgelqf_Destruct( dague_handle_t *o )
{
    dague_zgelqf_handle_t *dague_zgelqf = (dague_zgelqf_handle_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgelqf->arenas[DAGUE_zgelqf_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgelqf->arenas[DAGUE_zgelqf_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgelqf->arenas[DAGUE_zgelqf_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgelqf->arenas[DAGUE_zgelqf_LITTLE_T_ARENA  ]->opaque_dtt) );

    dague_private_memory_fini( dague_zgelqf->p_work );
    dague_private_memory_fini( dague_zgelqf->p_tau  );
    free( dague_zgelqf->p_work );
    free( dague_zgelqf->p_tau  );

    DAGUE_INTERNAL_HANDLE_DESTRUCT(dague_zgelqf);
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zgelqf - Computes the LQ factorization a M-by-N matrix A:
 * A = L * Q.
 *
 * The method used in this algorithm is a tile LQ algorithm with a flat
 * reduction tree. It is recommended to use the super tiling parameter (SNB) to
 * improve the performance of the factorization.
 * A high SNB parameter reduces the communication volume, but also deteriorates
 * the load balancing if too important. A small one increases the communication
 * volume, but improves load balancing.
 * A good SNB value should provide enough work to all available cores on one
 * node. It is then recommended to set it to 4 when creating the matrix
 * descriptor.
 *
 * This variant is good for square large problems.
 * For other problems, see:
 *   - dplasma_zgelqf_param() parameterized with trees for tall and skinny
 *     matrices
 *   - dplasma_zgelqf_param() parameterized with systolic tree if computation
 *     load per node is very low.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and below the diagonal of the array contain
 *          the M-by-min(M,N) lower trapezoidal matrix L (L is lower triangular
 *          if (N >= M); the elements above the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile LQ algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          On exit, contains auxiliary information required to compute the Q
 *          matrix, and/or solve the problem.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgelqf_New
 * @sa dplasma_zgelqf_Destruct
 * @sa dplasma_cgelqf
 * @sa dplasma_dgelqf
 * @sa dplasma_sgelqf
 *
 ******************************************************************************/
int
dplasma_zgelqf( dague_context_t *dague,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *T )
{
    dague_handle_t *dague_zgelqf = NULL;

    if ( (A->mt != T->mt) || (A->nt != T->nt) ) {
        dplasma_error("dplasma_zgelqf", "T doesn't have the same number of tiles as A");
        return -101;
    }

    dague_zgelqf = dplasma_zgelqf_New(A, T);

    if ( dague_zgelqf != NULL ) {
        dague_enqueue(dague, (dague_handle_t*)dague_zgelqf);
        dplasma_progress(dague);
        dplasma_zgelqf_Destruct( dague_zgelqf );
    }

    return 0;
}
