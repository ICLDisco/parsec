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
#include <core_blas.h>
#include <lapacke.h>
#include <math.h>
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "data_dist/matrix/vector_two_dim_cyclic.h"
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
        vector_two_dim_cyclic_t V;
        dague_complex64_t *Vmat, tau;

        vector_two_dim_cyclic_init( &V, matrix_ComplexDouble, matrix_Tile,
                                    A->super.nodes, A->super.cores, A->super.myrank,
                                    A->mb, A->m, 0, A->m, twodA->grid.strows, 1 );
        V.mat = dague_data_allocate((size_t)V.super.nb_local_tiles *
                                    (size_t)V.super.bsiz *
                                    (size_t)dague_datadist_getsizeoftype(V.super.mtype));
        dague_ddesc_set_key((dague_ddesc_t*)&V, "V");
        Vmat = (dague_complex64_t*)(V.mat);

        /* generate random vector */
        dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&V, 3456 );

        /* generate householder vector */
        /* Could be done in // for one vector */
        if (A->super.myrank == 0) {
            LAPACKE_zlarfg( A->m, Vmat, Vmat+1, 1, &tau );
            Vmat[0] = 1.;
        }

#if defined(HAVE_MPI)
        MPI_Bcast( &tau, 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD );
#endif

        /* Compute the Householder matrix I - tau v * v' */
        dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., A);
        dplasma_zger( dague, -tau,
                      (tiled_matrix_desc_t*)&V,
                      (tiled_matrix_desc_t*)&V,
                      A );
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
