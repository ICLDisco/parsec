/*
 * Copyright (c) 2010      The University of Tennessee and The University
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

#include "generated/zgeqrf.h"

dague_object_t*
dplasma_zgeqrf_New( tiled_matrix_desc_t* ddescA,
		    tiled_matrix_desc_t *T )
{
    dague_zgeqrf_object_t* object;
    dague_remote_dep_datatype_t tile_ddt, lower_ddt, upper_ddt, littlet_ddt;
    int pri_change = dplasma_aux_get_priority( "GEQRF", ddescA ), MINMTNT;
#if defined(USE_MPI)
    MPI_Aint lb = 0, extent = 0;
#else
    int64_t lb = 0, extent = 0;
#endif  /* defined(USE_MPI) */
    
    MINMTNT = ((ddescA->lmt < ddescA->lnt)  ? ddescA->lmt : ddescA->lnt);
    object = dague_zgeqrf_new( (dague_ddesc_t*)T, (dague_ddesc_t*)ddescA,
			       ddescA->mb, ddescA->nb,
			       ddescA->m, ddescA->n, NULL, NULL, ddescA->lmt, ddescA->lnt,
			       MINMTNT );
    object->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->work_pool, ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t) );
    object->tau_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->tau_pool, ddescA->nb * sizeof(Dague_Complex64_t) );

    dplasma_aux_create_tile_type(MPITYPE, ddescA->nb, &tile_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(tile_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[dplasma_zgeqrf_DEFAULT_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &tile_ddt);

    dplasma_aux_create_lower_type(MPITYPE, ddescA->nb, &lower_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(lower_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[dplasma_zgeqrf_LOWER_TILE_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &lower_ddt);

    dplasma_aux_create_upper_type(MPITYPE, ddescA->nb, &upper_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(upper_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[dplasma_zgeqrf_UPPER_TILE_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &upper_ddt);

    dplasma_aux_create_rectangle_type(MPITYPE, T->mb, ddescA->nb, -1,  &littlet_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(littlet_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[dplasma_zgeqrf_LITTLE_T_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &littlet_ddt);

    return (dague_object_t*)object;
}

int
dplasma_zgeqrf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T) 
{
    dague_object_t *dague_zgeqrf = NULL;

    dague_zgeqrf = dague_zgeqrf_New(A, T, &info);

    dague_enqueue( dague, (dague_object_t*)dague_zgeqrf);
    dague_progress(dague);

    return 1;
}
