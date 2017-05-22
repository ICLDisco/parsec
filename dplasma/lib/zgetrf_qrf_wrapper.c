/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "parsec/parsec_config.h"
#include "dplasma.h"
#include <math.h>
#include "parsec/vpmap.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmajdf.h"
#include "parsec/private_mempool.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "zgetrf_qrf.h"

static inline void
dplasma_genrandom_lutab(int *lutab, int deb, int fin, int nb_lu, int rec_depth)
{
    if (deb == fin)
    {
        lutab[deb] = (nb_lu != 0);
    }
    else
    {
        int new_nb_lu = 0;
        int new_fin = 0;

        if ((fin - deb + 1) % 2 == 0)
            new_fin = deb - 1 + (fin - deb + 1) / 2;
        else
            new_fin = deb - 1 + (fin - deb) / 2 + (rec_depth % 2);

        if ((nb_lu % 2) == 0)
            new_nb_lu = nb_lu/2;
        else
            new_nb_lu = (nb_lu-1)/2 + (rec_depth % 2);

        dplasma_genrandom_lutab(lutab, deb,       new_fin, new_nb_lu,         rec_depth+1);
        dplasma_genrandom_lutab(lutab, new_fin+1, fin,     nb_lu - new_nb_lu, rec_depth+1);
    }
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgetrf_qrf_New - Generates the handle that computes an hybrid LU-QR
 * factorization of a M-by-N matrix A.
 *
 * This algorithm tries to take advantage of the low number of flops of the LU
 * factorization with no pivoting, and of the stability of the QR factorization
 * to compensate the loss due to the no pivoting strategy.
 * See dplasma_hqr_init() and dplasma_zgeqrf_param() for further detail on how
 * configuring the tree of the QR part.
 *
 * See following paper for further details:
 * [1] Designing LU-QR hybrid solvers for performance and stability. M. Faverge,
 * J. Herrmann, J. Langou, B. Lowery, Y. Robert, and J. Dongarra
 *
 * Other variants of LU decomposition are available in the library wioth the
 * following function:
 *     - dplasma_zgetrf_incpiv_New() that performs tile incremental pivoting
 *       algorithm.
 *     - dplasma_zgetrf_nopiv_New() that performs LU decomposition with no
 *       pivoting if the matrix is known as beeing diagonal dominant.
 *     - dplasma_zgetrf_New() that performs an LU decomposition with partial
 *       pivoting.
 *
 * WARNING:
 *    - The computations are not done by this call.
 *    - This algorithm is a prototype and its interface might change in future
 *      release of the library
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
 *          On exit, the factors L and U from the factorization
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 * @param[out] IPIV
 *          Descriptor of the IPIV matrix. Should be of size 1-by-min(M,N).
 *          On exit, contains the pivot indices; for 1 <= i <= min(M,N), row i
 *          of the matrix was interchanged with row IPIV(i).
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
 * @param[in] criteria
 *          Defines the criteria used to switch from LU to QR factorization.
 *          @arg DEFAULT_CRITERIUM: Even steps are LU, odd ones are QR.
 *          @arg HIGHAM_CRITERIUM:
 *          @arg MUMPS_CRITERIUM:
 *          @arg LU_ONLY_CRITERIUM:
 *          @arg QR_ONLY_CRITERIUM:
 *          @arg RANDOM_CRITERIUM:
 *          @arg HIGHAM_SUM_CRITERIUM:
 *          @arg HIGHAM_MAX_CRITERIUM:
 *          @arg HIGHAM_MOY_CRITERIUM:
 *          @arg -1: The default is ...
 *
 * @param[in] alpha
 *
 *
 * @param[out] lu_tab
 *          Integer array of size min(A.mt,A.nt)
 *          On exit, lu_tab[i] = 1, if an LU factorization has been performed at the ith step.
 *          lu_tab[i] = 0, otherwise.
 *
 * @param[out] INFO
 *          On algorithm completion: equal to 0 on success, i if the ith
 *          diagonal value is equal to 0. That implies incoherent result.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgetrf_qrf_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf_qrf
 * @sa dplasma_zgetrf_qrf_Destruct
 * @sa dplasma_cgetrf_qrf_New
 * @sa dplasma_dgetrf_qrf_New
 * @sa dplasma_sgetrf_qrf_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zgetrf_qrf_New( dplasma_qrtree_t *qrtree,
                        tiled_matrix_desc_t *A,
                        tiled_matrix_desc_t *IPIV,
                        tiled_matrix_desc_t *TS,
                        tiled_matrix_desc_t *TT,
                        int criteria, double alpha, int* lu_tab,
                        int* INFO)
{
    parsec_zgetrf_qrf_taskpool_t* tp;
    int ib = TS->mb;
    size_t sizeW = 1;
    size_t sizeReduceVec = 1;
    int nbthreads = dplasma_imax( 1, vpmap_get_nb_threads_in_vp(0) - 1 );

    /*
     * Compute W size according to criteria used.
     */
    if ((criteria == HIGHAM_CRITERIUM)     ||
        (criteria == HIGHAM_SUM_CRITERIUM) ||
        (criteria == HIGHAM_MAX_CRITERIUM) ||
        (criteria == HIGHAM_MOY_CRITERIUM))
    {
        int P = ((two_dim_block_cyclic_t*)A)->grid.rows;
        sizeReduceVec = P;
        sizeW = (A->mt + P - 1) / P;
    }
    else if (criteria == MUMPS_CRITERIUM)
    {
        sizeReduceVec = 2 * A->nb;
        sizeW         = 2 * A->nb;
    }

    if (criteria == RANDOM_CRITERIUM) {
        int minMNT = dplasma_imin( A->mt, A->nt );
        int nb_lu = lround( ((double)(minMNT) * alpha) / 100. );
        dplasma_genrandom_lutab(lu_tab, 0, minMNT-1, nb_lu, 0);
    }

    tp = parsec_zgetrf_qrf_new( A,
                                (parsec_ddesc_t*)IPIV,
                                TS,
                                TT,
                                lu_tab, *qrtree,
                                ib, criteria, alpha,
                                NULL, NULL, NULL,
                                INFO);

#if defined(CORE_GETRF_270)

    if ( A->storage == matrix_Tile ) {
        CORE_zgetrf_rectil_init();
    } else {
        CORE_zgetrf_reclap_init();
    }
    tp->_g_nbmaxthrd = dplasma_imin( nbthreads, 48 );

#else

    if ( A->storage == matrix_Tile ) {
        tp->_g_getrfdata = CORE_zgetrf_rectil_init(nbthreads);
    } else {
        tp->_g_getrfdata = CORE_zgetrf_reclap_init(nbthreads);
    }
    tp->_g_nbmaxthrd = nbthreads;

#endif

    tp->_g_W = (double*)malloc(sizeW * sizeof(double));

    tp->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_work, ib * TS->nb * sizeof(parsec_complex64_t) );

    tp->_g_p_tau = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_tau, TS->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( tp->arenas[PARSEC_zgetrf_qrf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( tp->arenas[PARSEC_zgetrf_qrf_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* IPIV */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_zgetrf_qrf_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_int_t, A->mb, 1, -1 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( tp->arenas[PARSEC_zgetrf_qrf_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_zgetrf_qrf_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, TS->mb, TS->nb, -1);

    /* ReduceVec */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_zgetrf_qrf_ReduceVec_ARENA],
                                 sizeReduceVec * sizeof(double), PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_t, sizeReduceVec, 1, -1);

    /* Choice */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_zgetrf_qrf_CHOICE_ARENA],
                                 sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_int_t, 1, 1, -1);

    return (parsec_taskpool_t*)tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgetrf_qrf_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zgetrf_qrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf_qrf_New
 * @sa dplasma_zgetrf_qrf
 *
 ******************************************************************************/
void
dplasma_zgetrf_qrf_Destruct( parsec_taskpool_t *tp )
{
    parsec_zgetrf_qrf_taskpool_t *parsec_zgetrf_qrf = (parsec_zgetrf_qrf_taskpool_t *)tp;

    parsec_matrix_del2arena( parsec_zgetrf_qrf->arenas[PARSEC_zgetrf_qrf_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( parsec_zgetrf_qrf->arenas[PARSEC_zgetrf_qrf_LOWER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zgetrf_qrf->arenas[PARSEC_zgetrf_qrf_PIVOT_ARENA     ] );
    parsec_matrix_del2arena( parsec_zgetrf_qrf->arenas[PARSEC_zgetrf_qrf_UPPER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zgetrf_qrf->arenas[PARSEC_zgetrf_qrf_LITTLE_T_ARENA  ] );
    parsec_matrix_del2arena( parsec_zgetrf_qrf->arenas[PARSEC_zgetrf_qrf_ReduceVec_ARENA ] );
    parsec_matrix_del2arena( parsec_zgetrf_qrf->arenas[PARSEC_zgetrf_qrf_CHOICE_ARENA    ] );

    parsec_private_memory_fini( parsec_zgetrf_qrf->_g_p_work );
    parsec_private_memory_fini( parsec_zgetrf_qrf->_g_p_tau  );

    if ( parsec_zgetrf_qrf->_g_getrfdata != NULL )
        free( parsec_zgetrf_qrf->_g_getrfdata );
    free( parsec_zgetrf_qrf->_g_W );
    free( parsec_zgetrf_qrf->_g_p_work );
    free( parsec_zgetrf_qrf->_g_p_tau  );

    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgetrf_qrf_New - Generates the handle that computes an hybrid LU-QR
 * factorization of a M-by-N matrix A.
 *
 * This algorithm tries to take advantage of the low number of flops of the LU
 * factorization with no pivoting, and of the stability of the QR factorization
 * to compensate the loss due to the no pivoting strategy.
 * See dplasma_hqr_init() and dplasma_zgeqrf_param() for further detail on how
 * configuring the tree of the QR part.
 *
 * See following paper for further details:
 * [1] Designing LU-QR hybrid solvers for performance and stability. M. Faverge,
 * J. Herrmann, J. Langou, B. Lowery, Y. Robert, and J. Dongarra
 *
 * Other variants of LU decomposition are available in the library wioth the
 * following function:
 *     - dplasma_zgetrf_incpiv_New() that performs tile incremental pivoting
 *       algorithm.
 *     - dplasma_zgetrf_nopiv_New() that performs LU decomposition with no
 *       pivoting if the matrix is known as beeing diagonal dominant.
 *     - dplasma_zgetrf_New() that performs an LU decomposition with partial
 *       pivoting.
 *
 * WARNING:
 *    - This algorithm is a prototype and its interface might change in future
 *      release of the library
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
 *          On exit, the factors L and U from the factorization
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 * @param[out] IPIV
 *          Descriptor of the IPIV matrix. Should be of size 1-by-min(M,N).
 *          On exit, contains the pivot indices; for 1 <= i <= min(M,N), row i
 *          of the matrix was interchanged with row IPIV(i).
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
 * @param[in] criteria
 *          Defines the criteria used to switch from LU to QR factorization.
 *          @arg DEFAULT_CRITERIUM: Even steps are LU, odd ones are QR.
 *          @arg HIGHAM_CRITERIUM:
 *          @arg MUMPS_CRITERIUM:
 *          @arg LU_ONLY_CRITERIUM:
 *          @arg QR_ONLY_CRITERIUM:
 *          @arg RANDOM_CRITERIUM:
 *          @arg HIGHAM_SUM_CRITERIUM:
 *          @arg HIGHAM_MAX_CRITERIUM:
 *          @arg HIGHAM_MOY_CRITERIUM:
 *          @arg -1: The default is ...
 *
 * @param[in] alpha
 *
 *
 * @param[out] lu_tab
 *          Integer array of size min(A.mt,A.nt)
 *          On exit, lu_tab[i] = 1, if an LU factorization has been performed at the ith step.
 *          lu_tab[i] = 0, otherwise.
 *
 * @param[out] INFO
 *          On algorithm completion: equal to 0 on success, i if the ith
 *          diagonal value is equal to 0. That implies incoherent result.
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
 * @sa dplasma_zgetrf_qrf_New
 * @sa dplasma_zgetrf_qrf_Destruct
 * @sa dplasma_cgetrf_qrf
 * @sa dplasma_dgetrf_qrf
 * @sa dplasma_sgetrf_qrf
 *
 ******************************************************************************/
int
dplasma_zgetrf_qrf( parsec_context_t *parsec,
                    dplasma_qrtree_t *qrtree,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *IPIV,
                    tiled_matrix_desc_t *TS,
                    tiled_matrix_desc_t *TT,
                    int criteria, double alpha, int* lu_tab,
                    int* INFO )
{
    parsec_taskpool_t *parsec_zgetrf_qrf = NULL;

    parsec_zgetrf_qrf = dplasma_zgetrf_qrf_New(qrtree, A, IPIV, TS, TT, criteria, alpha, lu_tab, INFO);

    parsec_enqueue(parsec, (parsec_taskpool_t*)parsec_zgetrf_qrf);
    dplasma_wait_until_completion(parsec);

    dplasma_zgetrf_qrf_Destruct( parsec_zgetrf_qrf );
    return 0;
}
