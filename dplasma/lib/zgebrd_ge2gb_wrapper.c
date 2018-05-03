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
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/private_mempool.h"
#include "parsec/vpmap.h"

#include "zgebrd_ge2gb.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgebrd_ge2gbx_New - Generates the taskpool that computes the
 * reduction of general matrix A to a general band stored in Band.
 *
 * This algorithm is a generic algorithm that exploits trees from the
 * Hierarchical QR algorithm, and that is able to compute a direct
 * bidiagonalization or a R-bidiagonalization where a QR factorization is first
 * performed before to reduce to band only the R factors. For a simplifed used,
 * it is recommended to use the dplasma_zgebrd_ge2gb_New() function.
 * Each QR factorization is parameterized by a different reduction tree to
 * optimize the time to solution: qrtre0 defines the reduction tree used in the
 * QR factorization when R-bidiagonalization is performed, qrtree defines the
 * reduction tree for the QR steps of the bidiagonalization algorithm, and
 * lqtree defines the reduction tree for the LQ steps of the bidiagonalization
 * algorithm. Thus it is possible with dplasma_svd_init() and dplasma_hqr_init()
 * functions to try different types of trees that fits both the machine
 * caracteristics and the problem size. See dplasma_svd_init() and
 * dplasma_hqr_init() for further details on what kind of trees are well adapted
 * to your problem.
 *
 * For tiling, MB=200, and IB=32 usually give good results. Super-tiling
 * parameters SMB and SNB must be set to 1 for this algorithm.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] qrtre0
 *          The structure that describes the trees used to perform the
 *          preliminary QR factorization in R-bidiagonalization algorithm. If
 *          qrtre0 = qrtree, a direct bidiagonalization is performed.
 *          See dplasma_hqr_init() or dplasma_systolic_init() for further
 *          details on initializing this tree.
 *
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          QR steps of the bidiagonalization.
 *          See dplasma_svd_init() for further details on initializing this
 *          tree.
 *
 * @param[in] lqtree
 *          The structure that describes the trees used to perform the
 *          LQ steps of the bidiagonalization.
 *          See dplasma_svd_init() for further details on initializing this
 *          tree.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be reduced to the band form.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the matrix contains householder reflectors from the initial
 *          QR factorization, and from the LQ steps of the bidiagonalization
 *          stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] TS0
 *          Descriptor of the matrix TS0 distributed exactly as the A matrix. TS0.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * TS0.mb - by - A.nt * TS0.nb, with TS0.nb == A.nb.
 *          On exit, contains auxiliary information computed through TS kernels
 *          at the lowest level and which are required to generate the singular
 *          vectors when R-bidiagonalization is involved. If R-bidiagonalization
 *          is not performed, it must be equal to TS.
 *          For now, it is recommended to give a NULL pointer.
 *
 * @param[out] TT0
 *          Descriptor of the matrix TT0 distributed exactly as the A
 *          matrix. This matrix must be of size A.mt * TS0.mb - by - A.nt *
 *          TS0.nb, with TS0.nb == A.nb.  On exit, contains auxiliary
 *          information computed through TT kernels in the reductin trees and
 *          which are required to generate the singular vectors when
 *          R-bidiagonalization is involved. If R-bidiagonalization is not
 *          performed, it must be equal to TT.
 *          For now, it is recommended to give a NULL pointer.
 *
 * @param[out] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. This matrix must be of size A.mt * TS0.mb - by - A.nt *
 *          TS0.nb, with TS0.nb == A.nb.
 *          On exit, contains auxiliary information computed through TS kernels
 *          at the lowest level during the QR and LQ steps of the
 *          bidiagonalization algorithm and which are required to generate the
 *          singular vectors.
 *          For now, it is recommended to give a NULL pointer.
 *
 * @param[out] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. This matrix must be of size A.mt * TS0.mb - by - A.nt *
 *          TS0.nb, with TS0.nb == A.nb.
 *          On exit, contains auxiliary information computed through TT kernels
 *          in the reduction trees during the QR and LQ steps of the
 *          bidiagonalization algorithm and which are required to generate the
 *          singular vectors.
 *          For now, it is recommended to give a NULL pointer.
 *
 * @param[out] Band
 *          Descriptor of the matrix Band matrix. The matrix must be one tile
 *          high, and Band.mb must be greater or equal to A.mb+1. The width must
 *          be at least equal to A.n, and the Band.nb must be equal to A.nb.
 *          On exit, contains the band form of A centered in the middle of the tiles.
 *          For example, Band.mb = mb+1 is enough for a Lapack band reduction
 *          algorithm, but 3mb+1 is required for PLASMA reduction algorithm.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_zgebrd_ge2gb_Destruct();
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
parsec_taskpool_t*
dplasma_zgebrd_ge2gbx_New( int ib,
                           dplasma_qrtree_t *qrtre0,
                           dplasma_qrtree_t *qrtree,
                           dplasma_qrtree_t *lqtree,
                           parsec_tiled_matrix_dc_t *A,
                           parsec_tiled_matrix_dc_t *TS0,
                           parsec_tiled_matrix_dc_t *TT0,
                           parsec_tiled_matrix_dc_t *TS,
                           parsec_tiled_matrix_dc_t *TT,
                           parsec_tiled_matrix_dc_t *Band )
{
    parsec_zgebrd_ge2gb_taskpool_t* tp;

    if ( (TS0 != NULL) && ((A->mt > TS0->mt) || (A->nt > TS0->nt) || (TS0->mb < ib)) ) {
        dplasma_error("dplasma_zgebrd_ge2gb_New", "TS0 doesn't have the same number of tiles as A");
        return NULL;
    }
    if ( (TT0 != NULL) && ((A->mt > TT0->mt) || (A->nt > TT0->nt) || (TT0->mb < ib)) ) {
        dplasma_error("dplasma_zgebrd_ge2gb_New", "TT0 doesn't have the same number of tiles as A");
        return NULL;
    }
    if ( (TS != NULL) && ((A->mt > TS->mt) || (A->nt > TS->nt) || (TS->mb < ib)) ) {
        dplasma_error("dplasma_zgebrd_ge2gb_New", "TS doesn't have the same number of tiles as A");
        return NULL;
    }
    if ( (TT != NULL) && ((A->mt > TT->mt) || (A->nt > TT->nt) || (TT->mb < ib)) ) {
        dplasma_error("dplasma_zgebrd_ge2gb_New", "TT doesn't have the same number of tiles as A");
        return NULL;
    }
    if ( (Band->mb < (A->mb+1)) || (Band->nb != A->nb) ) {
        dplasma_error("dplasma_zgebrd_ge2gb_New", "Band doesn't have the correct tile size");
        return NULL;
    }
    if ( Band->n < A->n ) {
        dplasma_error("dplasma_zgebrd_ge2gb_New", "Band doesn't have enough columns");
        return NULL;
    }

    tp = parsec_zgebrd_ge2gb_new( A, TS0, TT0, TS, TT, Band,
                                  qrtre0, qrtree, lqtree,
                                  !(qrtre0 == qrtree), ib,
                                  NULL, NULL);

    tp->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_work, ib * A->nb * sizeof(parsec_complex64_t) );

    tp->_g_p_tau = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_tau, A->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( tp->arenas[PARSEC_zgebrd_ge2gb_DEFAULT_ARENA],
                            A->mb * A->nb * sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Upper triangular part Non-Unit (QR) */
    dplasma_add2arena_upper( tp->arenas[PARSEC_zgebrd_ge2gb_UPPER_NON_UNIT_ARENA],
                             A->mb * A->nb * sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 1 );

    /* Upper triangular part Unit (LQ) */
    dplasma_add2arena_upper( tp->arenas[PARSEC_zgebrd_ge2gb_UPPER_UNIT_ARENA],
                             A->mb * A->nb * sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Lower triangular part Non-Unit (LQ) */
    dplasma_add2arena_lower( tp->arenas[PARSEC_zgebrd_ge2gb_LOWER_NON_UNIT_ARENA],
                             A->mb * A->nb * sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 1 );

    /* Lower triangular part Unit (QR) */
    dplasma_add2arena_lower( tp->arenas[PARSEC_zgebrd_ge2gb_LOWER_UNIT_ARENA],
                             A->mb * A->nb * sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_zgebrd_ge2gb_LITTLE_T_ARENA],
                                 ib * A->nb * sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, ib, A->nb, -1);

    /* Band */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_zgebrd_ge2gb_BAND_ARENA],
                                 Band->mb * Band->nb * sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, Band->mb, Band->nb, -1);

    return (parsec_taskpool_t*)tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgebrd_ge2gb_New - Generates the taskpool that computes the
 * reduction of general matrix A to a general band stored in Band. This is a
 * simplified version of the dplasma_zgebrd_ge2gbx_New which cannot be used to
 * compute singular vectors as it doesn't return the trees and the T matrices.
 *
 * This algorithm is a generic algorithm that exploits trees from the
 * Hierarchical QR algorithm, and that is able to compute a direct
 * bidiagonalization or a R-bidiagonalization where a QR factorization is first
 * performed before to reduce to band only the R factors.
 * Each QR factorization is parameterized by a different reduction tree to
 * optimize the time to solution: qrtre0 defines the reduction tree used in the
 * QR factorization when R-bidiagonalization is performed, qrtree defines the
 * reduction tree for the QR steps of the bidiagonalization algorithm, and
 * lqtree defines the reduction tree for the LQ steps of the bidiagonalization
 * algorithm. Thus it is possible with dplasma_svd_init() and dplasma_hqr_init()
 * functions to try different types of trees that fits both the machine
 * caracteristics and the problem size. See dplasma_svd_init() and
 * dplasma_hqr_init() for further details on what kind of trees are well adapted
 * to your problem.
 *
 * For tiling, MB=200, and IB=32 usually give good results. Super-tiling
 * parameters SMB and SNB must be set to 1 for this algorithm.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be reduced to the band form.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the matrix contains householder reflectors from the initial
 *          QR factorization, and from the LQ steps of the bidiagonalization
 *          stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] Band
 *          Descriptor of the matrix Band matrix. The matrix must be one tile
 *          high, and Band.mb must be greater or equal to A.mb+1. The width must
 *          be at least equal to A.n, and the Band.nb must be equal to A.nb.
 *          On exit, contains the band form of A centered in the middle of the tiles.
 *          For example, Band.mb = mb+1 is enough for a Lapack band reduction
 *          algorithm, but 3mb+1 is required for PLASMA reduction algorithm.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_zgebrd_ge2gb_Destruct();
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
parsec_taskpool_t*
dplasma_zgebrd_ge2gb_New( int ib,
                          parsec_tiled_matrix_dc_t *A,
                          parsec_tiled_matrix_dc_t *Band )
{
    parsec_taskpool_t *tp;
    parsec_tiled_matrix_dc_t *subA = NULL;
    dplasma_qrtree_t *qrtre0, *qrtree, *lqtree;
    int P, Q, cores;

    cores = dplasma_imax( vpmap_get_nb_total_threads(), 1 );
    qrtree = malloc( sizeof(dplasma_qrtree_t) );
    lqtree = malloc( sizeof(dplasma_qrtree_t) );

    if ( !(A->dtype & two_dim_block_cyclic_type) ) {
        P = 1; Q = 1;
    }
    else {
        P = ((two_dim_block_cyclic_t*)A)->grid.rows;
        Q = ((two_dim_block_cyclic_t*)A)->grid.cols;
    }

    /* R-bidiagonalization */
    if ( A->mt >= A->nt || 1) {
        if ( A->mt > 2*A->nt ) {
            qrtre0 = malloc( sizeof(dplasma_qrtree_t) );
            dplasma_hqr_init( qrtre0, PlasmaNoTrans,
                              A, -1, -1, -1, P, -1, 0 );

            subA = tiled_matrix_submatrix( A, 0, 0, A->n, A->n );
        }
        else {
            qrtre0 = qrtree;
            subA   = A;
        }

        dplasma_svd_init( qrtree,
                          PlasmaNoTrans, subA,
                          -1, P, cores, 2 );

        dplasma_svd_init( lqtree,
                          PlasmaTrans, subA,
                          -1, Q, cores, 2 );

        if (subA != A) {
            free(subA);
        }
    }
    else {
        fprintf(stderr, "The case M < N is not handled yet\n" );
        return NULL;
#if 0
        if ( A->nt > 2*A->mt ) {
            qrtre0 = malloc( sizeof(dplasma_qrtree_t) );
            dplasma_hqr_init( qrtre0, PlasmaTrans,
                              A, -1, -1, -1, Q, -1, 0 );

            subA = tiled_matrix_submatrix( A, 0, 0, A->m, A->m );
        }
        else {
            qrtre0 = qrtree;
            subA = A;
        }

        dplasma_svd_init( qrtree,
                          PlasmaTrans, subA,
                          -1, Q, cores, 2 );

        dplasma_svd_init( lqtree,
                          PlasmaTrans, subA,
                          -1, P, cores, 2 );

        if (subA != A) {
            free(subA);
        }
#endif
    }

    tp = dplasma_zgebrd_ge2gbx_New( ib, qrtre0, qrtree, lqtree,
                                    A, NULL, NULL, NULL, NULL, Band);

    return tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgebrd_ge2gbx_Destruct - Free the data structure associated to an
 *  taskpool created with dplasma_zgebrd_ge2gbx_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgebrd_ge2gb_New
 * @sa dplasma_zgebrd_ge2gb
 *
 ******************************************************************************/
void
dplasma_zgebrd_ge2gbx_Destruct( parsec_taskpool_t *tp)
{
    parsec_zgebrd_ge2gb_taskpool_t *parsec_zgebrd_ge2gb = (parsec_zgebrd_ge2gb_taskpool_t *)tp;

    parsec_matrix_del2arena( parsec_zgebrd_ge2gb->arenas[PARSEC_zgebrd_ge2gb_DEFAULT_ARENA       ] );
    parsec_matrix_del2arena( parsec_zgebrd_ge2gb->arenas[PARSEC_zgebrd_ge2gb_LOWER_NON_UNIT_ARENA] );
    parsec_matrix_del2arena( parsec_zgebrd_ge2gb->arenas[PARSEC_zgebrd_ge2gb_LOWER_UNIT_ARENA    ] );
    parsec_matrix_del2arena( parsec_zgebrd_ge2gb->arenas[PARSEC_zgebrd_ge2gb_UPPER_NON_UNIT_ARENA] );
    parsec_matrix_del2arena( parsec_zgebrd_ge2gb->arenas[PARSEC_zgebrd_ge2gb_UPPER_UNIT_ARENA    ] );
    parsec_matrix_del2arena( parsec_zgebrd_ge2gb->arenas[PARSEC_zgebrd_ge2gb_LITTLE_T_ARENA      ] );
    parsec_matrix_del2arena( parsec_zgebrd_ge2gb->arenas[PARSEC_zgebrd_ge2gb_BAND_ARENA          ] );

    parsec_private_memory_fini( parsec_zgebrd_ge2gb->_g_p_work );
    parsec_private_memory_fini( parsec_zgebrd_ge2gb->_g_p_tau  );
    free( parsec_zgebrd_ge2gb->_g_p_work );
    free( parsec_zgebrd_ge2gb->_g_p_tau  );

    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgebrd_ge2gb_Destruct - Free the data structure associated to an
 *  taskpool created with dplasma_zgebrd_ge2gb_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgebrd_ge2gb_New
 * @sa dplasma_zgebrd_ge2gb
 *
 ******************************************************************************/
void
dplasma_zgebrd_ge2gb_Destruct( parsec_taskpool_t *tp )
{
    parsec_zgebrd_ge2gb_taskpool_t *parsec_zgebrd_ge2gb = (parsec_zgebrd_ge2gb_taskpool_t *)tp;

    if ( parsec_zgebrd_ge2gb->_g_qrtre0 != parsec_zgebrd_ge2gb->_g_qrtree ) {
        dplasma_hqr_finalize( parsec_zgebrd_ge2gb->_g_qrtre0 );
        free( parsec_zgebrd_ge2gb->_g_qrtre0 );
    }
    dplasma_hqr_finalize( parsec_zgebrd_ge2gb->_g_qrtree );
    dplasma_hqr_finalize( parsec_zgebrd_ge2gb->_g_lqtree );
    free( parsec_zgebrd_ge2gb->_g_qrtree );
    free( parsec_zgebrd_ge2gb->_g_lqtree );

    dplasma_zgebrd_ge2gbx_Destruct(tp);
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
 * @sa dplasma_zgebrd_ge2gb_New
 * @sa dplasma_zgebrd_ge2gb_Destruct
 * @sa dplasma_cgebrd_ge2gb
 * @sa dplasma_dgebrd_ge2gb
 * @sa dplasma_sgebrd_ge2gb
 *
 ******************************************************************************/
int
dplasma_zgebrd_ge2gbx( parsec_context_t *parsec, int ib,
                       dplasma_qrtree_t *qrtre0,
                       dplasma_qrtree_t *qrtree,
                       dplasma_qrtree_t *lqtree,
                       parsec_tiled_matrix_dc_t *A,
                       parsec_tiled_matrix_dc_t *TS0,
                       parsec_tiled_matrix_dc_t *TT0,
                       parsec_tiled_matrix_dc_t *TS,
                       parsec_tiled_matrix_dc_t *TT,
                       parsec_tiled_matrix_dc_t *Band)
{
    parsec_taskpool_t *parsec_zgebrd_ge2gb = NULL;

    parsec_zgebrd_ge2gb = dplasma_zgebrd_ge2gbx_New(ib, qrtre0, qrtree, lqtree,
                                                   A, TS0, TT0, TS, TT, Band);

    parsec_context_add_taskpool(parsec, (parsec_taskpool_t*)parsec_zgebrd_ge2gb);
    dplasma_wait_until_completion(parsec);

    dplasma_zgebrd_ge2gbx_Destruct( parsec_zgebrd_ge2gb );

    return 0;
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
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] ib Internal Blocking
 * 
 * @param[in,out] A
 *
 * @param[in,out] Band
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
dplasma_zgebrd_ge2gb( parsec_context_t *parsec, int ib,
                      parsec_tiled_matrix_dc_t *A,
                      parsec_tiled_matrix_dc_t *Band)
{
    parsec_taskpool_t *parsec_zgebrd_ge2gb = NULL;

    parsec_zgebrd_ge2gb = dplasma_zgebrd_ge2gb_New(ib, A, Band);

    parsec_context_add_taskpool(parsec, (parsec_taskpool_t*)parsec_zgebrd_ge2gb);
    dplasma_wait_until_completion(parsec);

    dplasma_zgebrd_ge2gb_Destruct( parsec_zgebrd_ge2gb );

    return 0;
}

