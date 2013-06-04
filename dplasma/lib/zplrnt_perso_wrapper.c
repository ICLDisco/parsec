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

#include "zplrnt_toeppd.h"
#include "zplrnt_perso.h"

extern int GKK_getLeaderNbr(int me, int ne, int *nleaders, int **leaders);

int dplasma_zplrnt_toeppd( dague_context_t *dague,
                           tiled_matrix_desc_t *A,
                           unsigned long long int seed )
{
    dague_zplrnt_toeppd_object_t* object;

    object = dague_zplrnt_toeppd_new( seed,
                                      (dague_ddesc_t*)A );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zplrnt_toeppd_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Vector type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zplrnt_toeppd_VECTOR_ARENA],
                            A->mb*2*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE, A->mb );

    dague_enqueue(dague, (dague_object_t*)object);
    dplasma_progress(dague);

    dplasma_datatype_undefine_type( &(object->arenas[DAGUE_zplrnt_toeppd_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(object->arenas[DAGUE_zplrnt_toeppd_VECTOR_ARENA ]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(object);

    return 0;
}


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
        /* gallery('house', random, 0 ) */
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

        dague_data_free(V.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&V);
    }
    break;

    case MATRIX_CONDEX:
    {
        /* gallery('condex', A->m, 4, 100.) */
        dague_complex64_t theta = 100.;
        two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A;
        two_dim_block_cyclic_t Q;
        two_dim_block_cyclic_init( &Q, matrix_ComplexDouble, matrix_Tile,
                                   1, A->super.cores, A->super.myrank,
                                   A->mb, A->nb, A->m, 3, 0, 0, A->m, 3, twodA->grid.strows, twodA->grid.stcols, 1 );
        Q.mat = dague_data_allocate((size_t)Q.super.nb_local_tiles *
                                    (size_t)Q.super.bsiz *
                                    (size_t)dague_datadist_getsizeoftype(Q.super.mtype));
        dague_ddesc_set_key((dague_ddesc_t*)&Q, "Q");

        if (A->super.myrank == 0) {
            dague_complex64_t *Qmat;
            dague_complex64_t tau[3];
            int i;

            Qmat = (dague_complex64_t*)(Q.mat);

            /* first column is ones */
            for( i=0; i < Q.super.lm; i++, Qmat++ )
                *Qmat = (PLASMA_Complex64_t)1.0;

            /* Second column is [1 0 0 ... 0] */
            *Qmat = (PLASMA_Complex64_t)1.;
            Qmat++;
            for( i=1; i<Q.super.lm; i++, Qmat++ )
                *Qmat = (PLASMA_Complex64_t)0.;

            /* third column is ... */
            for( i=0; i<Q.super.lm; i++, Qmat++ )
                *Qmat = (PLASMA_Complex64_t)( pow( -1.0, (double)i ) * (1.0 + (double)i/(A->n-1) ) );

            /* generate orthogonal projector */
            LAPACKE_zgeqrf( LAPACK_COL_MAJOR, A->m, 3,    Q.mat, Q.super.lm, tau );
            LAPACKE_zungqr( LAPACK_COL_MAJOR, A->m, 3, 3, Q.mat, Q.super.lm, tau );

            /*
             * Conversion to tile layout
             */
            Qmat = (dague_complex64_t*)(Q.mat);
            {
                dague_complex64_t *W = (dague_complex64_t*) malloc (A->mb * sizeof(dague_complex64_t) );
                int *leaders = NULL;
                int i, nleaders;

                /* Get all the cycles leaders and length
                 * They are the same for each independent problem (each panel) */
                GKK_getLeaderNbr( Q.super.lmt, A->nb, &nleaders, &leaders );

                /* shift cycles. */
                for(i=0; i<nleaders; i++) {

                    /* cycle #i belongs to this thread, so shift it */
                    memcpy(W, Qmat + leaders[i*3] * A->mb, A->mb * sizeof(dague_complex64_t) );
                    CORE_zshiftw(leaders[i*3], leaders[i*3+1], A->mt, A->nb, A->mb, Qmat, W);
                }

                free(leaders); free(W);
            }
        }

        dplasma_zlaset( dague, PlasmaUpperLower, 0., 1. + theta, A );
        dplasma_zgemm( dague, PlasmaNoTrans, PlasmaConjTrans,
                       -theta, (tiled_matrix_desc_t*)&Q,
                               (tiled_matrix_desc_t*)&Q,
                       1.,     A );

        dague_data_free(Q.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&Q);
    }
    break;

    case MATRIX_TOEPPD:
    {
        dplasma_zplrnt_toeppd( dague, A, seed );
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
