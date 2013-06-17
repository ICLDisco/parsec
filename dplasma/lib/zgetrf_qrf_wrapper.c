/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "zgetrf_qrf.h"

void CORE_zgetrf_reclap_init(void);
void CORE_zgetrf_rectil_init(void);

dague_object_t* dplasma_zgetrf_qrf_New( dplasma_qrtree_t *qrtree,
                                        tiled_matrix_desc_t *A,
                                        tiled_matrix_desc_t *IPIV,
                                        tiled_matrix_desc_t *TS,
                                        tiled_matrix_desc_t *TT,
                                        int criteria, double alpha, int* lu_tab,
                                        int* INFO)
{
    dague_zgetrf_qrf_object_t* object;
    int ib = TS->mb;
    size_t sizeW = 1;
    size_t sizeReduceVec = 1;

    /*
     * Compute W size according to criteria used.
     */
    if ((criteria == HIGHAM_CRITERIUM) || (criteria == HIGHAM_SUM_CRITERIUM) || (criteria == HIGHAM_MAX_CRITERIUM) || (criteria == HIGHAM_MOY_CRITERIUM)) {
        int P = ((two_dim_block_cyclic_t*)A)->grid.rows;
        sizeReduceVec = P;
        sizeW = (A->mt + P - 1) / P;
    }
    else if (criteria == MUMPS_CRITERIUM) {
        sizeReduceVec = 2 * A->nb;
        sizeW         =     A->nb;
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

    return (dague_object_t*)object;
}

int dplasma_zgetrf_qrf( dague_context_t *dague,
                        dplasma_qrtree_t *qrtree,
                        tiled_matrix_desc_t *A,
                        tiled_matrix_desc_t *IPIV,
                        tiled_matrix_desc_t *TS,
                        tiled_matrix_desc_t *TT,
                        int criteria, double alpha, int* lu_tab,
                        int* INFO )
{
    dague_object_t *dague_zgetrf_qrf = NULL;

    dague_zgetrf_qrf = dplasma_zgetrf_qrf_New(qrtree, A, IPIV, TS, TT, criteria, alpha, lu_tab, INFO);

    dague_enqueue(dague, (dague_object_t*)dague_zgetrf_qrf);
    dplasma_progress(dague);

    dplasma_zgetrf_qrf_Destruct( dague_zgetrf_qrf );
    return 0;
}

void
dplasma_zgetrf_qrf_Destruct( dague_object_t *o )
{
    dague_zgetrf_qrf_object_t *dague_zgetrf_qrf = (dague_zgetrf_qrf_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_PIVOT_ARENA     ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_LITTLE_T_ARENA  ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_ReduceVec_ARENA ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_qrf->arenas[DAGUE_zgetrf_qrf_CHOICE_ARENA    ]->opaque_dtt) );

    dague_private_memory_fini( dague_zgetrf_qrf->p_work );
    dague_private_memory_fini( dague_zgetrf_qrf->p_tau  );

    free( dague_zgetrf_qrf->W );
    free( dague_zgetrf_qrf->p_work );
    free( dague_zgetrf_qrf->p_tau  );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}
