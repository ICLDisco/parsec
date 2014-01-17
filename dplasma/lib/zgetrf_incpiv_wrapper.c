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
#include "dplasma/lib/memory_pool.h"

#include "zgetrf_incpiv.h"
#include "zgetrf_incpiv_sd.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zgetrf_incpiv_New - Generates the object that computes the LU
 * factorization of a M-by-N matrix A using tile algorithm.
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
 *     - dplasma_zgetrf_New() that performs LU decomposition with partial pivoting.
 *       This is limited to matrices with column-cyclic distribution.
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
 *          On exit, elements on and above the diagonal are the elements of
 *          U. Elements belowe the diagonal are NOT the classic L, but the L
 *          factors obtaines by succesive pivoting.
 *
 * @param[out] L
 *          Descriptor of the matrix L distributed exactly as the A matrix.
 *           - If IPIV != NULL, L.mb defines the IB parameter of the tile LU
 *          algorithm. This matrix must be of size A.mt * L.mb - by - A.nt *
 *          L.nb, with L.nb == A.nb.
 *          On exit, contains auxiliary information required to solve the system.
 *           - If IPIV == NULL, pivoting information are stored within
 *          L. (L.mb-1) defines the IB parameter of the tile LU algorithm. This
 *          matrix must be of size A.mt * L.mb - by - A.nt * L.nb, with L.nb =
 *          A.nb, and L.mb = ib+1.
 *          On exit, the first A.mb elements contains the IPIV information, the
 *          leftover contains auxiliary information required to solve the
 *          system.
 *
 * @param[out] IPIV
 *          Descriptor of the IPIV matrix. Should be distributed exactly as the
 *          A matrix. This matrix must be of size A.m - by - A.nt with IPIV.mb =
 *          A.mb and IPIV.nb = 1.
 *          On exit, contains the pivot indices of the successive row
 *          interchanged performed during the factorization.
 *          If IPIV == NULL, rows interchange information is stored within L.
 *
 * @param[out] INFO
 *          On algorithm completion: equal to 0 on success, i if the ith
 *          diagonal value is equal to 0. That implies incoherent result.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgetrf_incpiv_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf_incpiv
 * @sa dplasma_zgetrf_incpiv_Destruct
 * @sa dplasma_cgetrf_incpiv_New
 * @sa dplasma_dgetrf_incpiv_New
 * @sa dplasma_sgetrf_incpiv_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zgetrf_incpiv_New( tiled_matrix_desc_t *A,
                           tiled_matrix_desc_t *L,
                           tiled_matrix_desc_t *IPIV,
                           int *INFO )
{
    dague_zgetrf_incpiv_handle_t *dague_getrf_incpiv;
    int ib;

    if ( (A->mt != L->mt) || (A->nt != L->nt) ) {
        dplasma_error("dplasma_zgetrf_incpiv_New", "L doesn't have the same number of tiles as A");
        return NULL;
    }
    if ( (IPIV != NULL) && ((A->mt != IPIV->mt) || (A->nt != IPIV->nt)) ) {
        dplasma_error("dplasma_zgetrf_incpiv_New", "IPIV doesn't have the same number of tiles as A");
        return NULL;
    }

    if ( IPIV != NULL ) {
        ib = L->mb;
        dague_getrf_incpiv = dague_zgetrf_incpiv_new( (dague_ddesc_t*)A,
                                                      (dague_ddesc_t*)L,
                                                      (dague_ddesc_t*)IPIV,
                                                      INFO, NULL);
    } else {
        ib = L->mb - 1;
        dague_getrf_incpiv = (dague_zgetrf_incpiv_handle_t*)
            dague_zgetrf_incpiv_sd_new( (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)L,
                                        NULL, INFO, NULL);
    }

    dague_getrf_incpiv->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( dague_getrf_incpiv->work_pool, ib * L->nb * sizeof(dague_complex64_t) );

    /* A */
    dplasma_add2arena_tile( dague_getrf_incpiv->arenas[DAGUE_zgetrf_incpiv_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Lower part of A without diagonal part */
    dplasma_add2arena_lower( dague_getrf_incpiv->arenas[DAGUE_zgetrf_incpiv_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Upper part of A with diagonal part */
    dplasma_add2arena_upper( dague_getrf_incpiv->arenas[DAGUE_zgetrf_incpiv_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* IPIV */
    dplasma_add2arena_rectangle( dague_getrf_incpiv->arenas[DAGUE_zgetrf_incpiv_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    /* L */
    dplasma_add2arena_rectangle( dague_getrf_incpiv->arenas[DAGUE_zgetrf_incpiv_SMALL_L_ARENA],
                                 L->mb*L->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, L->mb, L->nb, -1);

    return (dague_handle_t*)dague_getrf_incpiv;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zgetrf_incpiv_Destruct - Free the data structure associated to an object
 *  created with dplasma_zgetrf_incpiv_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf_incpiv_New
 * @sa dplasma_zgetrf_incpiv
 *
 ******************************************************************************/
void
dplasma_zgetrf_incpiv_Destruct( dague_handle_t *o )
{
    dague_zgetrf_incpiv_handle_t *dague_zgetrf_incpiv = (dague_zgetrf_incpiv_handle_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgetrf_incpiv->arenas[DAGUE_zgetrf_incpiv_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_incpiv->arenas[DAGUE_zgetrf_incpiv_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_incpiv->arenas[DAGUE_zgetrf_incpiv_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_incpiv->arenas[DAGUE_zgetrf_incpiv_SMALL_L_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_incpiv->arenas[DAGUE_zgetrf_incpiv_PIVOT_ARENA     ]->opaque_dtt) );

    dague_private_memory_fini( dague_zgetrf_incpiv->work_pool );
    free( dague_zgetrf_incpiv->work_pool );

    DAGUE_INTERNAL_HANDLE_DESTRUCT(dague_zgetrf_incpiv);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zgetrf_incpiv - Computes the LU factorization of a M-by-N matrix A
 * using tile algorithm.
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
 *     - dplasma_zgetrf_New() that performs LU decomposition with partial pivoting.
 *       This is limited to matrices with column-cyclic distribution.
 *     - dplasma_zgetrf_nopiv_New() that performs LU decomposition with no pivoting
 *       if the matrix is known as beeing diagonal dominant.
 *     - dplasma_zgetrf_qrf_New() that performs an hybrid LU-QR decomposition.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, elements on and above the diagonal are the elements of
 *          U. Elements belowe the diagonal are NOT the classic L, but the L
 *          factors obtaines by succesive pivoting.
 *
 * @param[out] L
 *          Descriptor of the matrix L distributed exactly as the A matrix.
 *           - If IPIV != NULL, L.mb defines the IB parameter of the tile LU
 *          algorithm. This matrix must be of size A.mt * L.mb - by - A.nt *
 *          L.nb, with L.nb == A.nb.
 *          On exit, contains auxiliary information required to solve the system.
 *           - If IPIV == NULL, pivoting information are stored within
 *          L. (L.mb-1) defines the IB parameter of the tile LU algorithm. This
 *          matrix must be of size A.mt * L.mb - by - A.nt * L.nb, with L.nb =
 *          A.nb, and L.mb = ib+1.
 *          On exit, the first A.mb elements contains the IPIV information, the
 *          leftover contains auxiliary information required to solve the
 *          system.
 *
 * @param[out] IPIV
 *          Descriptor of the IPIV matrix. Should be distributed exactly as the
 *          A matrix. This matrix must be of size A.m - by - A.nt with IPIV.mb =
 *          A.mb and IPIV.nb = 1.
 *          On exit, contains the pivot indices of the successive row
 *          interchanged performed during the factorization.
 *          If IPIV == NULL, rows interchange information is stored within L.
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
 * @sa dplasma_zgetrf_incpiv
 * @sa dplasma_zgetrf_incpiv_Destruct
 * @sa dplasma_cgetrf_incpiv_New
 * @sa dplasma_dgetrf_incpiv_New
 * @sa dplasma_sgetrf_incpiv_New
 *
 ******************************************************************************/
int
dplasma_zgetrf_incpiv( dague_context_t *dague,
                       tiled_matrix_desc_t *A,
                       tiled_matrix_desc_t *L,
                       tiled_matrix_desc_t *IPIV )
{
    dague_handle_t *dague_zgetrf_incpiv = NULL;
    int info = 0;

    if ( (A->mt != L->mt) || (A->nt != L->nt) ) {
        dplasma_error("dplasma_zgetrf_incpiv", "L doesn't have the same number of tiles as A");
        return -3;
    }
    if ( (IPIV != NULL) && ((A->mt != IPIV->mt) || (A->nt != IPIV->nt)) ) {
        dplasma_error("dplasma_zgetrf_incpiv", "IPIV doesn't have the same number of tiles as A");
        return -4;
    }

    dague_zgetrf_incpiv = dplasma_zgetrf_incpiv_New(A, L, IPIV, &info);

    if ( dague_zgetrf_incpiv != NULL ) {
        dague_enqueue( dague, dague_zgetrf_incpiv );
        dplasma_progress(dague);
        dplasma_zgetrf_incpiv_Destruct( dague_zgetrf_incpiv );
        return info;
    }
    else
        return -101;
}
