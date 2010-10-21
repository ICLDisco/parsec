/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"
#include "memory_pool.h"
#ifdef DAGCOMPLEX
#ifdef DAGDOUBLE
#include "zgeqrf.h"
#else /* DAGSINGLE */
#include "cgeqrf.h"
#endif
#else /* DAGREAL */
#ifdef DAGDOUBLE
#include "dgeqrf.h"
#else
#include "sgeqrf.h"
#endif
#endif

dague_object_t*
DAGUEprefix(geqrf_New)(tiled_matrix_desc_t *T,
                       tiled_matrix_desc_t* ddescA,
                       int IB)
{
    dagueprefix(geqrf_object_t)* object;
    dague_remote_dep_datatype_t tile_ddt, lower_ddt, upper_ddt, littlet_ddt;
    int pri_change = dplasma_aux_get_priority( "GEQRF", ddescA ), MINMTNT;
#if defined(USE_MPI)
    MPI_Aint lb = 0, extent = 0;
#else
    int64_t lb = 0, extent = 0;
#endif  /* defined(USE_MPI) */

    MINMTNT = ((ddescA->lmt < ddescA->lnt)  ? ddescA->lmt : ddescA->lnt);
    object = dagueprefix(geqrf_new)( (dague_ddesc_t*)T, (dague_ddesc_t*)ddescA,
                                     ddescA->mb, ddescA->nb,
                                     ddescA->m, ddescA->n, NULL, NULL, ddescA->lmt, ddescA->lnt,
                                     MINMTNT );
    object->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->work_pool, ddescA->mb * ddescA->nb * sizeof(TYPENAME) );
    object->tau_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->tau_pool, ddescA->nb * sizeof(TYPENAME) );

    dplasma_aux_create_tile_type(MPITYPE, ddescA->nb, &tile_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(tile_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(geqrf_DEFAULT_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &tile_ddt);

    dplasma_aux_create_lower_type(MPITYPE, ddescA->nb, &lower_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(lower_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(geqrf_LOWER_TILE_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &lower_ddt);

    dplasma_aux_create_upper_type(MPITYPE, ddescA->nb, &upper_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(upper_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(geqrf_UPPER_TILE_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &upper_ddt);

    dplasma_aux_create_rectangle_type(MPITYPE, IB, ddescA->nb, -1,  &littlet_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(littlet_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(geqrf_LITTLE_T_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &littlet_ddt);

    return (dague_object_t*)object;
}
