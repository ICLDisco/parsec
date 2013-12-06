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

#include "zgetrf.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zgetrf_New - Generates the object that computes the LU factorization
 * of a M-by-N matrix A: A = P * L * U by partial pivoting algorithm.
 *
 * This algorithm exploits the multi-threaded recursive kernels of the PLASMA
 * library and by consequence require a column-cyclic data distribution if used
 * in distributed memory.
 * This is not an optimal solution for distributed memory system, and should be
 * used only if no other possibiliies is available. Absolute priority scheduler
 * is known to improve the performance of this algorithm and should be prefered.
 *
 * Other variants of LU decomposition are available in the library wioth the
 * following function:
 *     - dplasma_zgetrf_incpiv_New() that performs tile incremental pivoting
 *       algorithm.
 *     - dplasma_zgetrf_nopiv_New() that performs LU decomposition with no pivoting
 *       if the matrix is known as beeing diagonal dominant.
 *     - dplasma_zgetrf_qrf_New() that performs an hybrid LU-QR decomposition.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the factors L and U from the factorization
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 * @param[out] IPIV
 *          Descriptor of the IPIV matrix. Should be of size 1-by-min(M,N).
 *          On exit, contains the pivot indices; for 1 <= i <= min(M,N), row i
 *          of the matrix was interchanged with row IPIV(i).
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgetrf_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf
 * @sa dplasma_zgetrf_Destruct
 * @sa dplasma_cgetrf_New
 * @sa dplasma_dgetrf_New
 * @sa dplasma_sgetrf_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zgetrf_New( tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *IPIV,
                    int *INFO )
{
    dague_zgetrf_object_t *dague_getrf;

    if ( A->storage == matrix_Tile ) {
        CORE_zgetrf_rectil_init();
    } else {
        CORE_zgetrf_reclap_init();
    }

    dague_getrf = dague_zgetrf_new( (dague_ddesc_t*)A,
                                    (dague_ddesc_t*)IPIV,
                                    INFO );

    /* A */
    dplasma_add2arena_tile( dague_getrf->arenas[DAGUE_zgetrf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* IPIV */
    dplasma_add2arena_rectangle( dague_getrf->arenas[DAGUE_zgetrf_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, 1, A->mb, -1 );

    return (dague_object_t*)dague_getrf;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zgetrf_Destruct - Free the data structure associated to an object
 *  created with dplasma_zgetrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf_New
 * @sa dplasma_zgetrf
 *
 ******************************************************************************/
void
dplasma_zgetrf_Destruct( dague_object_t *o )
{
    dague_zgetrf_object_t *dague_zgetrf = (dague_zgetrf_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_PIVOT_ARENA  ]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(dague_zgetrf);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zgetrf - Computes the LU factorization of a M-by-N matrix A: A = P *
 * L * U by partial pivoting algorithm.
 *
 * This algorithm exploits the multi-threaded recursive kernels of the PLASMA
 * library and by consequence require a column-cyclic data distribution if used
 * in distributed memory.
 * This is not an optimal solution for distributed memory system, and should be
 * used only if no other possibiliies is available. Absolute priority scheduler
 * is known to improve the performance of this algorithm and should be prefered.
 *
 * Other variants of LU decomposition are available in the library wioth the
 * following function:
 *     - dplasma_zgetrf_incpiv() that performs tile incremental pivoting
 *       algorithm.
 *     - dplasma_zgetrf_nopiv() that performs LU decomposition with no pivoting
 *       if the matrix is known as beeing diagonal dominant.
 *     - dplasma_zgetrf_qrf() that performs an hybrid LU-QR decomposition.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the factors L and U from the factorization
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 * @param[out] IPIV
 *          Descriptor of the IPIV matrix. Should be of size 1-by-min(M,N).
 *          On exit, contains the pivot indices; for 1 <= i <= min(M,N), row i
 *          of the matrix was interchanged with row IPIV(i).
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval i if ith value is singular. Result is incoherent.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf
 * @sa dplasma_zgetrf_Destruct
 * @sa dplasma_cgetrf_New
 * @sa dplasma_dgetrf_New
 * @sa dplasma_sgetrf_New
 *
 ******************************************************************************/
int
dplasma_zgetrf( dague_context_t *dague,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *IPIV )
{
    dague_object_t *dague_zgetrf = NULL;

    int info = 0;

    dague_zgetrf = dplasma_zgetrf_New(A, IPIV, &info);

    if ( dague_zgetrf != NULL ) {
        dague_enqueue( dague, dague_zgetrf );
        dplasma_progress(dague);
        dplasma_zgetrf_Destruct( dague_zgetrf );
        return info;
    }
    else
        return -101;
}
