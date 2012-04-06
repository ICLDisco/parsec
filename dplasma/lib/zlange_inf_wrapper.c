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
#include "map2.h"

#include "zlange_inf_cyclic.h"

dague_object_t* dplasma_zlange_inf_New( tiled_matrix_desc_t *A,
                                        tiled_matrix_desc_t *W,
                                        tiled_matrix_desc_t *S,
                                        int P, int Q)
{
    dague_object_t *dague_zlange_inf = NULL;

    dague_zlange_inf = (dague_object_t*)dague_zlange_inf_cyclic_new(*A, (dague_ddesc_t*)A,
                                                                    *W, (dague_ddesc_t*)W,
                                                                    *S, (dague_ddesc_t*)S,
                                                                    P, Q);

    dplasma_add2arena_tile(((dague_zlange_inf_cyclic_object_t*)dague_zlange_inf)->arenas[DAGUE_zlange_inf_cyclic_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);
    dplasma_add2arena_tile(((dague_zlange_inf_cyclic_object_t*)dague_zlange_inf)->arenas[DAGUE_zlange_inf_cyclic_WTAB_ARENA],
                           A->mb*sizeof(double),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE, A->mb);
    dplasma_add2arena_tile(((dague_zlange_inf_cyclic_object_t*)dague_zlange_inf)->arenas[DAGUE_zlange_inf_cyclic_STAB_ARENA],
                           sizeof(double),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE, 1);

    return dague_zlange_inf;
}

void
dplasma_zlange_inf_Destruct( dague_object_t *o )
{
    dague_zlange_inf_cyclic_object_t *dague_zlange = (dague_zlange_inf_cyclic_object_t *)o;
    dague_zlange_inf_cyclic_destroy(dague_zlange);
}

double dplasma_zlange_inf( dague_context_t *dague,
                           PLASMA_enum ntype,
                           tiled_matrix_desc_t *A)
{
    two_dim_block_cyclic_t workW, workS;
    double result;
    dague_object_t *dague_zlange_inf = NULL;

    switch( ntype )
    {
    case PlasmaInfNorm:
        PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
            workW, two_dim_block_cyclic,
            (&workW, matrix_RealDouble, matrix_Tile, A->super.nodes, A->super.cores, A->super.myrank,
             A->mb, 1,          /* Dimesions of the tile */
             A->m, ((two_dim_block_cyclic_t*)A)->grid.cols, /* Dimensions of the matrix */
             0, 0,                                           /* Starting points (not important here) */
             A->m, ((two_dim_block_cyclic_t*)A)->grid.cols, /* Dimensions of the submatrix */
             ((two_dim_block_cyclic_t*)A)->grid.strows, ((two_dim_block_cyclic_t*)A)->grid.stcols,
             ((two_dim_block_cyclic_t*)A)->grid.rows));
        PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
            workS, two_dim_block_cyclic,
            (&workS, matrix_RealDouble, matrix_Tile, A->super.nodes, A->super.cores, A->super.myrank,
             1, 1,          /* Dimesions of the tile */
             ((two_dim_block_cyclic_t*)A)->grid.rows, ((two_dim_block_cyclic_t*)A)->grid.cols, /* Dimensions of the matrix */
             0, 0,                                           /* Starting points (not important here) */
             ((two_dim_block_cyclic_t*)A)->grid.rows, ((two_dim_block_cyclic_t*)A)->grid.cols, /* Dimensions of the submatrix */
             ((two_dim_block_cyclic_t*)A)->grid.strows, ((two_dim_block_cyclic_t*)A)->grid.stcols,
             ((two_dim_block_cyclic_t*)A)->grid.rows));
        break;

    default:
        return -1.0;
    }

    dague_zlange_inf = dplasma_zlange_inf_New(A,
                                              (tiled_matrix_desc_t *)&workW,
                                              (tiled_matrix_desc_t *)&workS,
                                              ((two_dim_block_cyclic_t*)A)->grid.rows,
                                              ((two_dim_block_cyclic_t*)A)->grid.cols);

    if ( dague_zlange_inf != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zlange_inf);
        dplasma_progress(dague);
        dplasma_zlange_inf_Destruct( dague_zlange_inf );
    }

    if ( workS.super.super.myrank == 0 )
        result = ((double*)workS.mat)[0];

    dague_data_free(workW.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&workW);
    dague_data_free(workS.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&workS);


#if defined(HAVE_MPI)
    MPI_Bcast(&result, 1, MPI_DOUBLE, 0, dplasma_comm);
#endif

    return result;
}

