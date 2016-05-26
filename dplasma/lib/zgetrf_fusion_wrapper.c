/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "zgetrf_fusion.h"

#define LDV  3
#define IB  32

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };

dague_handle_t* dplasma_zgetrf_fusion_New( tiled_matrix_desc_t *A,
                                           tiled_matrix_desc_t *IPIV,
                                           int P,
                                           int Q,
                                           int *info )
{
    dague_zgetrf_fusion_handle_t *dague_zgetrf_fusion = NULL;
    int nb = A->nb;

    /* The code has to be fixed for N >> M */
    assert( A->m >= A->n );

    *info = 0;
    dague_zgetrf_fusion = dague_zgetrf_fusion_new((dague_ddesc_t*)A,
                                                  (dague_ddesc_t*)IPIV,
                                                  IB,
                                                  P,
                                                  Q,
                                                  info);

    /* A */
    dplasma_add2arena_tile( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            dague_datatype_double_complex_t, A->mb );

    /* SWAP */
    dplasma_add2arena_rectangle( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_SWAP_ARENA],
                                 (2*nb+1)*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_double_complex_t, 2*nb+1, 1, -1 );

    /* MAXL */
    dplasma_add2arena_rectangle( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_MAXL_ARENA],
                                 (nb+1)*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_double_complex_t, 1, nb+1, -1 );

    /* UMES */
    dplasma_add2arena_rectangle( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_UMES_ARENA],
                                 IB*nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_double_complex_t, IB, nb, -1 );

    /* PIVOT */
    dplasma_add2arena_rectangle( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_int_t, IPIV->mb, IPIV->nb, -1 );

    /* PERMUT */
    dplasma_add2arena_rectangle( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_PERMUT_ARENA],
                                 2 * nb * sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_int_t, 2, nb, -1 );

    return (dague_handle_t*)dague_zgetrf_fusion;
}

void
dplasma_zgetrf_fusion_Destruct( dague_handle_t *handle )
{
    dague_zgetrf_fusion_handle_t *dague_zgetrf_fusion = (dague_zgetrf_fusion_handle_t *)handle;

    dague_matrix_del2arena( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_DEFAULT_ARENA] );
    dague_matrix_del2arena( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_SWAP_ARENA   ] );
    dague_matrix_del2arena( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_MAXL_ARENA   ] );
    dague_matrix_del2arena( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_UMES_ARENA   ] );
    dague_matrix_del2arena( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_PIVOT_ARENA  ] );
    dague_matrix_del2arena( dague_zgetrf_fusion->arenas[DAGUE_zgetrf_fusion_PERMUT_ARENA ] );

    dague_handle_free(handle);
}

int dplasma_zgetrf_fusion( dague_context_t *dague,
                   tiled_matrix_desc_t *A,
                   tiled_matrix_desc_t *IPIV)
{
    int info = 0, ginfo = 0 ;
    dague_handle_t *dague_zgetrf_fusion = NULL;

    int P = ((two_dim_block_cyclic_t*)A)->grid.rows;
    int Q = ((two_dim_block_cyclic_t*)A)->grid.cols;

    dague_zgetrf_fusion = dplasma_zgetrf_fusion_New(A, IPIV, P, Q, &info);

    if ( dague_zgetrf_fusion != NULL )
    {
        dague_enqueue( dague, (dague_handle_t*)dague_zgetrf_fusion);
        dplasma_progress(dague);
        dplasma_zgetrf_fusion_Destruct( dague_zgetrf_fusion );
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}
