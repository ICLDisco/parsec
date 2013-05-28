/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> c d s
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zplrnt_perso.h"

/***************************************************************************//**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zplrnt_perso_New - Generate a random matrix by tiles.
 *
 *******************************************************************************
 *
 * @param[out] A
 *          On exit, The random hermitian matrix A generated.
 *
 * @param[in] seed
 *          The seed used in the random generation.
 *
 ******************************************************************************/
dague_object_t* dplasma_zplrnt_perso_New( tiled_matrix_desc_t *A,
                                          enum matrix_init_e type,
                                          unsigned long long int seed )
{
    dague_zplrnt_perso_object_t* object;

    object = dague_zplrnt_perso_new( type, seed, (dague_ddesc_t*)A );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zplrnt_perso_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    return (dague_object_t*)object;
}

int dplasma_zplrnt_perso( dague_context_t *dague,
                          tiled_matrix_desc_t *A,
                          enum matrix_init_e type,
                          unsigned long long int seed )
{
    dague_object_t *dague_zplrnt_perso = NULL;

    switch( type ) {
    case MATRIX_HOUSE:
    {
        two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A;
        two_dim_block_cyclic_t V, T;
        tiled_matrix_desc_t *subA;
        two_dim_block_cyclic_init( &V, matrix_ComplexDouble, matrix_Tile,
                                   A->super.nodes, A->super.cores, A->super.myrank,
                                   A->mb, A->nb, A->m, 1, 0, 0, A->m, 1,
                                   twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows );
        V.mat = dague_data_allocate((size_t)V.super.nb_local_tiles *
                                    (size_t)V.super.bsiz *
                                    (size_t)dague_datadist_getsizeoftype(V.super.mtype));
        dague_ddesc_set_key((dague_ddesc_t*)&V, "V");

        two_dim_block_cyclic_init( &T, matrix_ComplexDouble, matrix_Tile,
                                   A->super.nodes, A->super.cores, A->super.myrank,
                                   32, A->nb, ( A->m + 31 ) / 32, 1, 0, 0, ( A->m + 31 ) / 32, 1,
                                   twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows );
        T.mat = dague_data_allocate((size_t)T.super.nb_local_tiles *
                                    (size_t)T.super.bsiz *
                                    (size_t)dague_datadist_getsizeoftype(T.super.mtype));
        dague_ddesc_set_key((dague_ddesc_t*)&T, "T");

        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&V, 3456 );
        dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&T);

        subA = tiled_matrix_submatrix( A, 0, 0, A->m, 1 );

        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&V, subA );
        free( subA );
        dplasma_zgeqrf( dague, (tiled_matrix_desc_t *)&V, (tiled_matrix_desc_t *)&T );
        dplasma_zungqr( dague,
                        (tiled_matrix_desc_t *)&V,
                        (tiled_matrix_desc_t *)&T,
                        (tiled_matrix_desc_t *)&A );

    }
    break;

    default:
        dague_zplrnt_perso = dplasma_zplrnt_perso_New( A, type, seed );

        dague_enqueue(dague, (dague_object_t*)dague_zplrnt_perso);
        dplasma_progress(dague);

        dplasma_zplrnt_perso_Destruct( dague_zplrnt_perso );
    }
    return 0;
}

void
dplasma_zplrnt_perso_Destruct( dague_object_t *o )
{
    dague_zplrnt_perso_object_t *dague_zplrnt_perso = (dague_zplrnt_perso_object_t *)o;
    dplasma_datatype_undefine_type( &(dague_zplrnt_perso->arenas[DAGUE_zplrnt_perso_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(dague_zplrnt_perso);
}
