/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include <plasma.h>
#include <dague.h>
#include <scheduling.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"
#include "memory_pool.h"

#include "generated/zgelqf.h"

dague_object_t* dplasma_zgelqf_New( tiled_matrix_desc_t* ddescA,
                                    tiled_matrix_desc_t *T )
{
    dague_zgelqf_object_t* object;
    dague_remote_dep_datatype_t tile_ddt, lower_ddt, upper_ddt, littlet_ddt;
#if defined(HAVE_MPI)
    MPI_Aint lb = 0, extent = 0;
#else
    int64_t extent = 0;
#endif  /* defined(HAVE_MPI) */
    
#warning Alm1 / Alm -> see with Hatem & Piotr, tuesday.
    object = dague_zgelqf_new( (dague_ddesc_t*)ddescA, ddescA->mt, ddescA->nt, ddescA->mb, ddescA->nb, ddescA->m, ddescA->n, /* lm1 */ ddescA->mt, /*lm*/ ddescA->mt, ddescA->i, (dague_ddesc_t*)T, T->mb, T->nb, NULL, NULL, T->mb);
    object->pool_Tnb = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->pool_Tnb, T->nb * sizeof(Dague_Complex64_t) );
    object->pool_ibTnb = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->pool_ibTnb, T->mb * T->nb * sizeof(Dague_Complex64_t) );

    dplasma_datatype_define_tile(MPI_DOUBLE_COMPLEX, ddescA->nb, &tile_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(tile_ddt, &lb, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgelqf_DEFAULT_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, tile_ddt);

    dplasma_datatype_define_lower(MPI_DOUBLE_COMPLEX, ddescA->nb, 0, &lower_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(lower_ddt, &lb, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgelqf_LOWER_TILE_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, lower_ddt);

    dplasma_datatype_define_upper(MPI_DOUBLE_COMPLEX, ddescA->nb, 1, &upper_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(upper_ddt, &lb, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgelqf_UPPER_TILE_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, upper_ddt);

    dplasma_datatype_define_rectangle(MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1,  &littlet_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(littlet_ddt, &lb, &extent);
#else
    extent = T->mb * T->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgelqf_LITTLE_T_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, littlet_ddt);

    return (dague_object_t*)object;
}

int dplasma_zgelqf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T) 
{
    dague_object_t *dague_zgelqf = NULL;

    dague_zgelqf = dplasma_zgelqf_New(A, T);

    dague_enqueue(dague, (dague_object_t*)dague_zgelqf);
    dague_progress(dague);

    return 0;
}
