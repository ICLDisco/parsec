/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dague/private_mempool.h"

#include "zgebrd_ge2gb.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeqrf_param_New - Generates the handle that computes the
 * hierarchical QR factorization of a M-by-N matrix A: A = Q * R.
 *
 * The method used in this algorithm is a hierachical tile QR algorithm with
 * several level of reduction trees defined by the qrtree structure.
 * Thus it is possible with dplasma_hqr_init() to try different type of tree
 * that fits the machine caracteristics. See dplasma_hqr_init() for further
 * details on what kind of trees are well adapted to your problem.
 *
 * For tiling, MB=200, and IB=32 usually give good results. Super-tiling
 * parameters SMB and SNB must be set to 1 for this algorithm.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          hierarchical QR factorization.
 *          See dplasma_hqr_init() or dplasma_systolic_init().
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array contain
 *          the min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular
 *          if (M >= N); the elements below the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] TS
 *          Descriptor of the matrix TS distributed exactly as the A matrix. TS.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb == A.nb.
 *          On exit, contains auxiliary information computed through TS kernels
 *          at the lowest level and which are required to generate the Q matrix,
 *          and/or solve the problem.
 *
 * @param[out] TT
 *          Descriptor of the matrix TT distributed exactly as the A matrix. TT.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb == A.nb.
 *          On exit, contains auxiliary information computed through TT kernels
 *          at higher levels and which are required to generate the Q matrix,
 *          and/or solve the problem.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague handle describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgeqrf_param_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_param
 * @sa dplasma_zgebrd_ge2gb_Destruct
 * @sa dplasma_cgebrd_ge2gb_New
 * @sa dplasma_dgebrd_ge2gb_New
 * @sa dplasma_sgebrd_ge2gb_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zgebrd_ge2gb_New( dplasma_qrtree_t *qrtre0,
                          dplasma_qrtree_t *qrtree,
                          dplasma_qrtree_t *lqtree,
                          tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *TS0,
                          tiled_matrix_desc_t *TT0,
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT,
                          tiled_matrix_desc_t *Band )
{
    dague_zgebrd_ge2gb_handle_t* handle;
    int ib = TS->mb;

    if ( (A->mt > TS->mt) || (A->nt > TS->nt) ) {
        dplasma_error("dplasma_zgebrd_ge2gb_New", "TS doesn't have the same number of tiles as A");
        return NULL;
    }
    if ( (A->mt > TT->mt) || (A->nt > TT->nt) ) {
        dplasma_error("dplasma_zgebrd_ge2gb_New", "TT doesn't have the same number of tiles as A");
        return NULL;
    }

    handle = dague_zgebrd_ge2gb_new( (dague_ddesc_t*)A,
                                     (dague_ddesc_t*)TS0,
                                     (dague_ddesc_t*)TT0,
                                     (dague_ddesc_t*)TS,
                                     (dague_ddesc_t*)TT,
                                     (dague_ddesc_t*)Band,
                                     *qrtre0, *qrtree, *lqtree,
                                     !(qrtre0 == qrtree),
                                     NULL, NULL);

    handle->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( handle->p_work, ib * TS->nb * sizeof(dague_complex64_t) );

    handle->p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( handle->p_tau, TS->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( handle->arenas[DAGUE_zgebrd_ge2gb_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            dague_datatype_double_complex_t, A->mb );

    /* Upper triangular part Non-Unit (QR) */
    dplasma_add2arena_upper( handle->arenas[DAGUE_zgebrd_ge2gb_UPPER_NON_UNIT_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             dague_datatype_double_complex_t, A->mb, 1 );

    /* Upper triangular part Unit (LQ) */
    dplasma_add2arena_upper( handle->arenas[DAGUE_zgebrd_ge2gb_UPPER_UNIT_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             dague_datatype_double_complex_t, A->mb, 0 );

    /* Lower triangular part Non-Unit (LQ) */
    dplasma_add2arena_lower( handle->arenas[DAGUE_zgebrd_ge2gb_LOWER_NON_UNIT_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             dague_datatype_double_complex_t, A->mb, 1 );

    /* Lower triangular part Unit (QR) */
    dplasma_add2arena_lower( handle->arenas[DAGUE_zgebrd_ge2gb_LOWER_UNIT_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             dague_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( handle->arenas[DAGUE_zgebrd_ge2gb_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_double_complex_t, TS->mb, TS->nb, -1);

    /* Band */
    dplasma_add2arena_rectangle( handle->arenas[DAGUE_zgebrd_ge2gb_BAND_ARENA],
                                 Band->mb*Band->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_double_complex_t, Band->mb, Band->nb, -1);

    return (dague_handle_t*)handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgebrd_ge2gb_Destruct - Free the data structure associated to an
 *  handle created with dplasma_zgebrd_ge2gb_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgebrd_ge2gb_New
 * @sa dplasma_zgebrd_ge2gb
 *
 ******************************************************************************/
void
dplasma_zgebrd_ge2gb_Destruct( dague_handle_t *handle )
{
    dague_zgebrd_ge2gb_handle_t *dague_zgebrd_ge2gb = (dague_zgebrd_ge2gb_handle_t *)handle;

    dague_matrix_del2arena( dague_zgebrd_ge2gb->arenas[DAGUE_zgebrd_ge2gb_DEFAULT_ARENA       ] );
    dague_matrix_del2arena( dague_zgebrd_ge2gb->arenas[DAGUE_zgebrd_ge2gb_LOWER_NON_UNIT_ARENA] );
    dague_matrix_del2arena( dague_zgebrd_ge2gb->arenas[DAGUE_zgebrd_ge2gb_LOWER_UNIT_ARENA    ] );
    dague_matrix_del2arena( dague_zgebrd_ge2gb->arenas[DAGUE_zgebrd_ge2gb_UPPER_NON_UNIT_ARENA] );
    dague_matrix_del2arena( dague_zgebrd_ge2gb->arenas[DAGUE_zgebrd_ge2gb_UPPER_UNIT_ARENA    ] );
    dague_matrix_del2arena( dague_zgebrd_ge2gb->arenas[DAGUE_zgebrd_ge2gb_LITTLE_T_ARENA      ] );
    dague_matrix_del2arena( dague_zgebrd_ge2gb->arenas[DAGUE_zgebrd_ge2gb_BAND_ARENA          ] );

    dague_private_memory_fini( dague_zgebrd_ge2gb->p_work );
    dague_private_memory_fini( dague_zgebrd_ge2gb->p_tau  );
    free( dague_zgebrd_ge2gb->p_work );
    free( dague_zgebrd_ge2gb->p_tau  );

    dague_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgebrd_ge2gb - Computes the hierarchical QR factorization of a M-by-N
 * matrix A: A = Q * R.
 *
 * The method used in this algorithm is a hierachical tile QR algorithm with
 * several level of reduction trees defined by the qrtree structure.
 * Thus it is possible with dplasma_hqr_init() to try different type of tree
 * that fits the machine caracteristics. See dplasma_hqr_init() for further
 * details on what kind of trees are well adapted to your problem.
 *
 * For tiling, MB=200, and IB=32 usually give good results. Super-tiling
 * parameters SMB and SNB must be set to 1 for this algorithm.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          hierarchical QR factorization.
 *          See dplasma_hqr_init() or dplasma_systolic_init().
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array contain
 *          the min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular
 *          if (M >= N); the elements below the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] TS
 *          Descriptor of the matrix TS distributed exactly as the A matrix. TS.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb == A.nb.
 *          On exit, contains auxiliary information computed through TS kernels
 *          at the lowest level and which are required to generate the Q matrix,
 *          and/or solve the problem.
 *
 * @param[out] TT
 *          Descriptor of the matrix TT distributed exactly as the A matrix. TT.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb == A.nb.
 *          On exit, contains auxiliary information computed through TT kernels
 *          at higher levels and which are required to generate the Q matrix,
 *          and/or solve the problem.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgebrd_ge2gb_New
 * @sa dplasma_zgebrd_ge2gb_Destruct
 * @sa dplasma_cgebrd_ge2gb
 * @sa dplasma_dgebrd_ge2gb
 * @sa dplasma_sgebrd_ge2gb
 *
 ******************************************************************************/
int
dplasma_zgebrd_ge2gb( dague_context_t *dague,
                      dplasma_qrtree_t *qrtre0,
                      dplasma_qrtree_t *qrtree,
                      dplasma_qrtree_t *lqtree,
                      tiled_matrix_desc_t *A,
                      tiled_matrix_desc_t *TS0,
                      tiled_matrix_desc_t *TT0,
                      tiled_matrix_desc_t *TS,
                      tiled_matrix_desc_t *TT,
                      tiled_matrix_desc_t *Band)
{
    dague_handle_t *dague_zgebrd_ge2gb = NULL;

    if ( (A->mt > TS->mt) || (A->nt > TS->nt) ) {
        dplasma_error("dplasma_zgebrd_ge2gb", "TS doesn't have the same number of tiles as A");
        return -4;
    }
    if ( (A->mt > TT->mt) || (A->nt > TT->nt) ) {
        dplasma_error("dplasma_zgebrd_ge2gb", "TT doesn't have the same number of tiles as A");
        return -5;
    }

    dague_zgebrd_ge2gb = dplasma_zgebrd_ge2gb_New(qrtre0, qrtree, lqtree,
                                                  A, TS0, TT0, TS, TT, Band);

    dague_enqueue(dague, (dague_handle_t*)dague_zgebrd_ge2gb);
    dplasma_progress(dague);

    dplasma_zgebrd_ge2gb_Destruct( dague_zgebrd_ge2gb );

    return 0;
}

