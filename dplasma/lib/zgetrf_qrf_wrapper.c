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

#include "dplasma.h"
#include <math.h>
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/memory_pool.h"
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
 * dplasma_zgetrf_qrf_New - Generates the object that computes an hybrid LU-QR
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
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
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
dague_handle_t*
dplasma_zgetrf_qrf_New( dplasma_qrtree_t *qrtree,
                        tiled_matrix_desc_t *A,
                        tiled_matrix_desc_t *IPIV,
                        tiled_matrix_desc_t *TS,
                        tiled_matrix_desc_t *TT,
                        int criteria, double alpha, int* lu_tab,
                        int* INFO)
{
    dague_zgetrf_qrf_handle_t* object;
    int ib = TS->mb;
    size_t sizeW = 1;
    size_t sizeReduceVec = 1;

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

    if ( A->storage == matrix_Tile ) {
        CORE_zgetrf_rectil_init();
    } else {
        CORE_zgetrf_reclap_init();
    }

    object = dague_zgetrf_qrf_new( (dague_ddesc_t*)A,
                                   (dague_ddesc_t*)IPIV,
                                   (dague_ddesc_t*)TS,
                                   (dague_ddesc_t*)TT,
                                   lu_tab, *qrtree,
                                   ib, criteria, alpha,
                                   NULL, NULL, NULL,
                                   INFO);

    object->W = (double*)malloc(sizeW * sizeof(double));

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * TS->nb * sizeof(dague_complex64_t) );

    object->p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_tau, TS->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zgetrf_qrf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zgetrf_qrf_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* IPIV */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgetrf_qrf_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( object->arenas[DAGUE_zgetrf_qrf_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgetrf_qrf_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, TS->mb, TS->nb, -1);

    /* ReduceVec */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgetrf_qrf_ReduceVec_ARENA],
                                 sizeReduceVec * sizeof(double), DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE, sizeReduceVec, 1, -1);

    /* Choice */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgetrf_qrf_CHOICE_ARENA],
                                 sizeof(int), DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, 1, 1, -1);

    return (dague_handle_t*)object;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgetrf_qrf_Destruct - Free the data structure associated to an object
 *  created with dplasma_zgetrf_qrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf_qrf_New
 * @sa dplasma_zgetrf_qrf
 *
 ******************************************************************************/
void
dplasma_zgetrf_qrf_Destruct( dague_handle_t *o )
{
    dague_zgetrf_qrf_handle_t *dague_zgetrf_qrf = (dague_zgetrf_qrf_handle_t *)o;

    dague_matrix_del2arena( dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_DEFAULT_ARENA   ] );
    dague_matrix_del2arena( dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_LOWER_TILE_ARENA] );
    dague_matrix_del2arena( dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_PIVOT_ARENA     ] );
    dague_matrix_del2arena( dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_UPPER_TILE_ARENA] );
    dague_matrix_del2arena( dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_LITTLE_T_ARENA  ] );
    dague_matrix_del2arena( dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_ReduceVec_ARENA ] );
    dague_matrix_del2arena( dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_CHOICE_ARENA    ] );

    dague_private_memory_fini( dague_zgetrf_qrf->p_work );
    dague_private_memory_fini( dague_zgetrf_qrf->p_tau  );

    free( dague_zgetrf_qrf->W );
    free( dague_zgetrf_qrf->p_work );
    free( dague_zgetrf_qrf->p_tau  );

    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgetrf_qrf_New - Generates the object that computes an hybrid LU-QR
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
dplasma_zgetrf_qrf( dague_context_t *dague,
                    dplasma_qrtree_t *qrtree,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *IPIV,
                    tiled_matrix_desc_t *TS,
                    tiled_matrix_desc_t *TT,
                    int criteria, double alpha, int* lu_tab,
                    int* INFO )
{
    dague_handle_t *dague_zgetrf_qrf = NULL;

    dague_zgetrf_qrf = dplasma_zgetrf_qrf_New(qrtree, A, IPIV, TS, TT, criteria, alpha, lu_tab, INFO);

    dague_enqueue(dague, (dague_handle_t*)dague_zgetrf_qrf);
    dplasma_progress(dague);

    dplasma_zgetrf_qrf_Destruct( dague_zgetrf_qrf );
    return 0;
}
