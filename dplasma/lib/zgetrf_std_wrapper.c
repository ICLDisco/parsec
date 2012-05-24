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

#include "zgetrf_std.h"

dague_object_t* dplasma_zgetrf_std_New( tiled_matrix_desc_t *A,
                                        tiled_matrix_desc_t *IPIV,
                                        int P,
                                        int Q,
                                        int *info )
{
    dague_object_t *dague_zgetrf_std = NULL;
    two_dim_block_cyclic_t *BUFFER, *ACOPY;
    int mb = A->mb, nb = A->nb;

    /* Create the workspaces */
    BUFFER = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*BUFFER), two_dim_block_cyclic,
        (BUFFER, matrix_ComplexDouble, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank,
         mb,   nb,      /* Dimesions of the tile                */
         mb*P, nb*A->nt,/* Dimensions of the matrix             */
         0,    0,       /* Starting points (not important here) */
         mb*P, nb*A->nt,/* Dimensions of the submatrix          */
         1, 1, P));

    ACOPY  = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*ACOPY), two_dim_block_cyclic,
        (ACOPY, matrix_ComplexDouble, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank,
         mb,   nb,      /* Dimesions of the tile                */
         mb*P, nb*A->nt,/* Dimensions of the matrix             */
         0,    0,       /* Starting points (not important here) */
         mb*P, nb*A->nt,/* Dimensions of the submatrix          */
         1, 1, P));

    *info = 0;
    dague_zgetrf_std = (dague_object_t*)dague_zgetrf_std_new((dague_ddesc_t*)A,
                                                             (dague_ddesc_t*)IPIV,
                                                             (dague_ddesc_t*)BUFFER,
                                                             (dague_ddesc_t*)ACOPY,
                                                             P,
                                                             Q,
                                                             info);

    /* A */
    dplasma_add2arena_tile( ((dague_zgetrf_std_object_t*)dague_zgetrf_std)->arenas[DAGUE_zgetrf_std_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* IPIV */
    dplasma_add2arena_rectangle( ((dague_zgetrf_std_object_t*)dague_zgetrf_std)->arenas[DAGUE_zgetrf_std_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    return (dague_object_t*)dague_zgetrf_std;
}

void
dplasma_zgetrf_std_Destruct( dague_object_t *o )
{
    dague_zgetrf_std_object_t *dague_zgetrf_std = (dague_zgetrf_std_object_t *)o;

    two_dim_block_cyclic_t *BUFFER = (two_dim_block_cyclic_t*)(dague_zgetrf_std->BUFFER);
    dague_data_free( BUFFER->mat );
    dague_ddesc_destroy( dague_zgetrf_std->BUFFER );
    free( dague_zgetrf_std->BUFFER );

    two_dim_block_cyclic_t *ACOPY = (two_dim_block_cyclic_t*)(dague_zgetrf_std->ACOPY);
    dague_data_free( ACOPY->mat );
    dague_ddesc_destroy( dague_zgetrf_std->ACOPY );
    free( dague_zgetrf_std->ACOPY );


    dplasma_datatype_undefine_type( &(dague_zgetrf_std->arenas[DAGUE_zgetrf_std_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_std->arenas[DAGUE_zgetrf_std_PIVOT_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

int dplasma_zgetrf_std( dague_context_t *dague,
                   tiled_matrix_desc_t *A,
                   tiled_matrix_desc_t *IPIV)
{
    int info = 0, ginfo = 0 ;
    dague_object_t *dague_zgetrf_std = NULL;

    int P = ((two_dim_block_cyclic_t*)A)->grid.rows;
    int Q = ((two_dim_block_cyclic_t*)A)->grid.cols;

    dague_zgetrf_std = dplasma_zgetrf_std_New(A, IPIV, P, Q, &info);

    if ( dague_zgetrf_std != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zgetrf_std);
        dplasma_progress(dague);
        dplasma_zgetrf_std_Destruct( dague_zgetrf_std );
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}

