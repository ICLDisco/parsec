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
                                        int P, int Q,
                                        double *result)
{
    two_dim_block_cyclic_t *W;
    dague_zlange_inf_cyclic_object_t *dague_zlange_inf = NULL;

    /* Create the workspace */
    W = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));

    PASTE_CODE_INIT_AND_ALLOCATE_MATRIX(
        (*W), two_dim_block_cyclic,
        (W, matrix_RealDouble, matrix_Tile,
         A->super.nodes, A->super.cores, A->super.myrank,
         A->mb, 1,  /* Dimesions of the tile                */
         A->m, Q,   /* Dimensions of the matrix             */
         0, 0,      /* Starting points (not important here) */
         A->m, Q,   /* Dimensions of the submatrix          */
         1, 1, P));

    /* Create the DAG */
    dague_zlange_inf = dague_zlange_inf_cyclic_new(*A, (dague_ddesc_t*)A,
                                                   (dague_ddesc_t*)W,
                                                   P, Q,
                                                   result);

    /* Set the datatypes */
    dplasma_add2arena_tile(dague_zlange_inf->arenas[DAGUE_zlange_inf_cyclic_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);
    dplasma_add2arena_tile(dague_zlange_inf->arenas[DAGUE_zlange_inf_cyclic_COL_ARENA],
                           A->mb*sizeof(double),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE, A->mb);
    dplasma_add2arena_tile(dague_zlange_inf->arenas[DAGUE_zlange_inf_cyclic_ELT_ARENA],
                           sizeof(double),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE, 1);

    return (dague_object_t*)dague_zlange_inf;
}

void
dplasma_zlange_inf_Destruct( dague_object_t *o )
{
    dague_zlange_inf_cyclic_object_t *dague_zlange = (dague_zlange_inf_cyclic_object_t *)o;
    two_dim_block_cyclic_t *W = (two_dim_block_cyclic_t*)(dague_zlange->W);

    dague_data_free( W->mat );
    dague_ddesc_destroy( dague_zlange->W );
    free( dague_zlange->W );

    dplasma_datatype_undefine_type( &(dague_zlange->arenas[DAGUE_zlange_inf_cyclic_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zlange->arenas[DAGUE_zlange_inf_cyclic_COL_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zlange->arenas[DAGUE_zlange_inf_cyclic_ELT_ARENA]->opaque_dtt) );

    dague_zlange_inf_cyclic_destroy(dague_zlange);
}

double dplasma_zlange_inf( dague_context_t *dague,
                           PLASMA_enum ntype,
                           tiled_matrix_desc_t *A)
{
    double result;
    dague_object_t *dague_zlange_inf = NULL;

    dague_zlange_inf = dplasma_zlange_inf_New(A,
                                              ((two_dim_block_cyclic_t*)A)->grid.rows,
                                              ((two_dim_block_cyclic_t*)A)->grid.cols,
                                              &result);

    if ( dague_zlange_inf != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zlange_inf);
        dplasma_progress(dague);
        dplasma_zlange_inf_Destruct( dague_zlange_inf );
    }

    return result;
}

