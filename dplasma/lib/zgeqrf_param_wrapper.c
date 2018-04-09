/*
 * Copyright (c) 2011-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "parsec/private_mempool.h"

#include "zgeqrf_param.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeqrf_param_New - Generates the taskpool that computes the
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
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgeqrf_param_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_param
 * @sa dplasma_zgeqrf_param_Destruct
 * @sa dplasma_cgeqrf_param_New
 * @sa dplasma_dgeqrf_param_New
 * @sa dplasma_sgeqrf_param_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zgeqrf_param_New( dplasma_qrtree_t *qrtree,
                          parsec_tiled_matrix_dc_t *A,
                          parsec_tiled_matrix_dc_t *TS,
                          parsec_tiled_matrix_dc_t *TT )
{
    parsec_zgeqrf_param_taskpool_t* tp;
    int ib = TS->mb;

    if ( (A->mt != TS->mt) || (A->nt != TS->nt) ) {
        dplasma_error("dplasma_zgeqrf_param_New", "TS doesn't have the same number of tiles as A");
        return NULL;
    }
    if ( (A->mt != TT->mt) || (A->nt != TT->nt) ) {
        dplasma_error("dplasma_zgeqrf_param_New", "TT doesn't have the same number of tiles as A");
        return NULL;
    }

    tp = parsec_zgeqrf_param_new( A,
                                  TS,
                                  TT,
                                  *qrtree, ib, NULL, NULL);

    tp->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_work, ib * TS->nb * sizeof(parsec_complex64_t) );

    tp->_g_p_tau = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_tau, TS->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( tp->arenas[PARSEC_zgeqrf_param_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( tp->arenas[PARSEC_zgeqrf_param_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 1 );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( tp->arenas[PARSEC_zgeqrf_param_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_zgeqrf_param_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, TS->mb, TS->nb, -1);

    return (parsec_taskpool_t*)tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeqrf_param_Destruct - Free the data structure associated to an
 *  taskpool created with dplasma_zgeqrf_param_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_param_New
 * @sa dplasma_zgeqrf_param
 *
 ******************************************************************************/
void
dplasma_zgeqrf_param_Destruct( parsec_taskpool_t *tp )
{
    parsec_zgeqrf_param_taskpool_t *parsec_zgeqrf_param = (parsec_zgeqrf_param_taskpool_t *)tp;

    parsec_matrix_del2arena( parsec_zgeqrf_param->arenas[PARSEC_zgeqrf_param_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( parsec_zgeqrf_param->arenas[PARSEC_zgeqrf_param_LOWER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zgeqrf_param->arenas[PARSEC_zgeqrf_param_UPPER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zgeqrf_param->arenas[PARSEC_zgeqrf_param_LITTLE_T_ARENA  ] );

    parsec_private_memory_fini( parsec_zgeqrf_param->_g_p_work );
    parsec_private_memory_fini( parsec_zgeqrf_param->_g_p_tau  );
    free( parsec_zgeqrf_param->_g_p_work );
    free( parsec_zgeqrf_param->_g_p_tau  );

    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeqrf_param - Computes the hierarchical QR factorization of a M-by-N
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
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
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
 * @sa dplasma_zgeqrf_param_New
 * @sa dplasma_zgeqrf_param_Destruct
 * @sa dplasma_cgeqrf_param
 * @sa dplasma_dgeqrf_param
 * @sa dplasma_sgeqrf_param
 *
 ******************************************************************************/
int
dplasma_zgeqrf_param( parsec_context_t *parsec,
                      dplasma_qrtree_t *qrtree,
                      parsec_tiled_matrix_dc_t *A,
                      parsec_tiled_matrix_dc_t *TS,
                      parsec_tiled_matrix_dc_t *TT)
{
    parsec_taskpool_t *parsec_zgeqrf_param = NULL;

    if ( (A->mt != TS->mt) || (A->nt != TS->nt) ) {
        dplasma_error("dplasma_zgeqrf_param", "TS doesn't have the same number of tiles as A");
        return -4;
    }
    if ( (A->mt != TT->mt) || (A->nt != TT->nt) ) {
        dplasma_error("dplasma_zgeqrf_param", "TT doesn't have the same number of tiles as A");
        return -5;
    }

    parsec_zgeqrf_param = dplasma_zgeqrf_param_New(qrtree, A, TS, TT);

    parsec_enqueue(parsec, (parsec_taskpool_t*)parsec_zgeqrf_param);
    dplasma_wait_until_completion(parsec);

    dplasma_zgeqrf_param_Destruct( parsec_zgeqrf_param );

    return 0;
}

