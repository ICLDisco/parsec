/*
 * Copyright (c) 2011-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "parsec.h"
#include <plasma.h>
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "zgetrf_fusion.h"

#define LDV  3
#define IB  32

parsec_handle_t*
dplasma_zgetrf_fusion_New( tiled_matrix_desc_t *A,
                           tiled_matrix_desc_t *IPIV,
                           int P, int Q,
                           int *info )
{
    parsec_zgetrf_fusion_handle_t *parsec_zgetrf_fusion = NULL;
    int nb = A->nb;

    /* The code has to be fixed for N >> M */
    assert( A->m >= A->n );

    *info = 0;
    parsec_zgetrf_fusion = parsec_zgetrf_fusion_new( A, IPIV, IB,
                                                     P, Q, info);

    /* A */
    dplasma_add2arena_tile( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* SWAP */
    dplasma_add2arena_rectangle( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_SWAP_ARENA],
                                 (2*nb+1)*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, 2*nb+1, 1, -1 );

    /* MAXL */
    dplasma_add2arena_rectangle( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_MAXL_ARENA],
                                 (nb+1)*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, 1, nb+1, -1 );

    /* UMES */
    dplasma_add2arena_rectangle( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_UMES_ARENA],
                                 IB*nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, IB, nb, -1 );

    /* PIVOT */
    dplasma_add2arena_rectangle( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_int_t, IPIV->mb, IPIV->nb, -1 );

    /* PERMUT */
    dplasma_add2arena_rectangle( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_PERMUT_ARENA],
                                 2 * nb * sizeof(int),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_int_t, 2, nb, -1 );

    return (parsec_handle_t*)parsec_zgetrf_fusion;
}

void
dplasma_zgetrf_fusion_Destruct( parsec_handle_t *handle )
{
    parsec_zgetrf_fusion_handle_t *parsec_zgetrf_fusion = (parsec_zgetrf_fusion_handle_t *)handle;

    parsec_matrix_del2arena( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_DEFAULT_ARENA] );
    parsec_matrix_del2arena( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_SWAP_ARENA   ] );
    parsec_matrix_del2arena( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_MAXL_ARENA   ] );
    parsec_matrix_del2arena( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_UMES_ARENA   ] );
    parsec_matrix_del2arena( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_PIVOT_ARENA  ] );
    parsec_matrix_del2arena( parsec_zgetrf_fusion->arenas[PARSEC_zgetrf_fusion_PERMUT_ARENA ] );

    parsec_handle_free(handle);
}

int
dplasma_zgetrf_fusion( parsec_context_t *parsec,
                       tiled_matrix_desc_t *A,
                       tiled_matrix_desc_t *IPIV )
{
    int info = 0, ginfo = 0 ;
    parsec_handle_t *parsec_zgetrf_fusion = NULL;

    int P = ((two_dim_block_cyclic_t*)A)->grid.rows;
    int Q = ((two_dim_block_cyclic_t*)A)->grid.cols;

    parsec_zgetrf_fusion = dplasma_zgetrf_fusion_New(A, IPIV, P, Q, &info);

    if ( parsec_zgetrf_fusion != NULL )
    {
        parsec_enqueue( parsec, (parsec_handle_t*)parsec_zgetrf_fusion);
        dplasma_progress(parsec);
        dplasma_zgetrf_fusion_Destruct( parsec_zgetrf_fusion );
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}
