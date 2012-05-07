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

#include "zgetrf_panel.h"

#define LDV  3

dague_object_t* dplasma_zgetrf_panel_New( tiled_matrix_desc_t *A,
                                        tiled_matrix_desc_t *IPIV,
                                        int P,
                                        int Q,
                                        int *info )
{
    dague_object_t *dague_zgetrf_panel = NULL;
    two_dim_block_cyclic_t *V;
    int nb = A->nb;

    /* Create the workspaces */
    V = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*V), two_dim_block_cyclic,
        (V, matrix_ComplexDouble, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank,
         LDV,   nb,      /* Dimesions of the tile                */
         LDV*P, nb*Q,    /* Dimensions of the matrix             */
         0,    0,        /* Starting points (not important here) */
         LDV*P, nb*Q,    /* Dimensions of the submatrix          */
         1, 1, P));

    *info = 0;
    dague_zgetrf_panel = (dague_object_t*)dague_zgetrf_panel_new((dague_ddesc_t*)A,
                                                             (dague_ddesc_t*)IPIV,
                                                             (dague_ddesc_t*)V,
                                                             P,
                                                             Q,
                                                             info);

    /* A */
    dplasma_add2arena_tile( ((dague_zgetrf_panel_object_t*)dague_zgetrf_panel)->arenas[DAGUE_zgetrf_panel_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* SWAP */
    dplasma_add2arena_rectangle( ((dague_zgetrf_panel_object_t*)dague_zgetrf_panel)->arenas[DAGUE_zgetrf_panel_SWAP_ARENA],
                                 LDV*A->nb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, 3, A->nb, -1 );

    return (dague_object_t*)dague_zgetrf_panel;
}

void
dplasma_zgetrf_panel_Destruct( dague_object_t *o )
{
    dague_zgetrf_panel_object_t *dague_zgetrf_panel = (dague_zgetrf_panel_object_t *)o;

    two_dim_block_cyclic_t *V = (two_dim_block_cyclic_t*)(dague_zgetrf_panel->V);
    dague_data_free( V->mat );
    dague_ddesc_destroy( dague_zgetrf_panel->V );
    free( dague_zgetrf_panel->V );

    dplasma_datatype_undefine_type( &(dague_zgetrf_panel->arenas[DAGUE_zgetrf_panel_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_panel->arenas[DAGUE_zgetrf_panel_SWAP_ARENA]->opaque_dtt) );

    dague_zgetrf_panel_destroy(dague_zgetrf_panel);
}

int dplasma_zgetrf_panel( dague_context_t *dague,
                   tiled_matrix_desc_t *A,
                   tiled_matrix_desc_t *IPIV)
{
    int info = 0, ginfo = 0 ;
    dague_object_t *dague_zgetrf_panel = NULL;

    int P = ((two_dim_block_cyclic_t*)A)->grid.rows;
    int Q = ((two_dim_block_cyclic_t*)A)->grid.cols;

    dague_zgetrf_panel = dplasma_zgetrf_panel_New(A, IPIV, P, Q, &info);

    if ( dague_zgetrf_panel != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zgetrf_panel);
        dplasma_progress(dague);
        dplasma_zgetrf_panel_Destruct( dague_zgetrf_panel );
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}

