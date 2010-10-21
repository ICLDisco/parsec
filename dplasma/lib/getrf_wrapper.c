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
#if   defined(PRECISION_z)
#include "zgetrf.h"
#include "zgetrf_sd.h"
#elif defined(PRECISION_c)
#include "cgetrf.h"
#include "cgetrf_sd.h"
#elif defined(PRECISION_d)
#include "dgetrf.h"
#include "dgetrf_sd.h"
#elif defined(PRECISION_s)
#include "sgetrf.h"
#include "sgetrf_sd.h"
#endif

dague_object_t*
DAGUEprefix(getrf_New)(tiled_matrix_desc_t *L,
                       tiled_matrix_desc_t *IPIV,
                       tiled_matrix_desc_t* ddescA,
                       int IB,
                       int* INFO)
{
    dagueprefix(getrf_object_t)* object;
    dague_remote_dep_datatype_t tile_ddt, lower_ddt, upper_ddt, pivot_ddt, littlel_ddt;
    int pri_change = dplasma_aux_get_priority( "GETRF", ddescA );
#if defined(USE_MPI)
    MPI_Aint lb = 0, extent = 0;
#else
    int64_t lb = 0, extent = 0;
#endif  /* defined(USE_MPI) */

    object = dagueprefix(getrf_new)( (dague_ddesc_t*)L, (dague_ddesc_t*)IPIV, (dague_ddesc_t*)ddescA,
                                     ddescA->n, ddescA->nb, ddescA->nt, IB, NULL, INFO );
    object->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->work_pool, IB * ddescA->nb * sizeof(TYPENAME) );

    dplasma_aux_create_tile_type(MPITYPE, ddescA->nb, &tile_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(tile_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(getrf_DEFAULT_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &tile_ddt);

    dplasma_aux_create_lower_type(MPITYPE, ddescA->nb, &lower_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(lower_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(getrf_LOWER_TILE_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &lower_ddt);

    dplasma_aux_create_upper_type(MPITYPE, ddescA->nb, &upper_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(upper_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(getrf_UPPER_TILE_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &upper_ddt);

    dplasma_aux_create_rectangle_type(MPITYPE, 1, ddescA->nb, -1, &pivot_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(pivot_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(getrf_PIVOT_VECT_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &pivot_ddt);

    dplasma_aux_create_rectangle_type(MPITYPE, IB, ddescA->nb,-1,  &littlel_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(littlel_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(getrf_LITTLE_L_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &littlel_ddt);

    return (dague_object_t*)object;
}

dague_object_t*
DAGUEprefix(getrf_sd_New)(tiled_matrix_desc_t *LIPIV,
                          tiled_matrix_desc_t* ddescA,
                          int IB,
                          int* INFO)
{
    dagueprefix(getrf_sd_object_t)* object;
    dague_remote_dep_datatype_t tile_ddt, lower_ddt, upper_ddt, pivot_ddt, littlel_pivot_ddt;
    int pri_change = dplasma_aux_get_priority( "GETRF", ddescA );
#if defined(USE_MPI)
    MPI_Aint lb = 0, extent = 0;
#else
    int64_t lb = 0, extent = 0;
#endif  /* defined(USE_MPI) */

    object = dagueprefix(getrf_sd_new)( (dague_ddesc_t*)LIPIV, (dague_ddesc_t*)ddescA,
                                        ddescA->n, ddescA->nb, ddescA->nt, IB, NULL, INFO );
    object->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->work_pool, IB * ddescA->nb * sizeof(TYPENAME) );

    dplasma_aux_create_tile_type(MPITYPE, ddescA->nb, &tile_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(tile_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(getrf_sd_DEFAULT_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &tile_ddt);

    dplasma_aux_create_lower_type(MPITYPE, ddescA->nb, &lower_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(lower_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(getrf_sd_LOWER_TILE_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &lower_ddt);

    dplasma_aux_create_upper_type(MPITYPE, ddescA->nb, &upper_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(upper_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(getrf_sd_UPPER_TILE_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &upper_ddt);

    dplasma_aux_create_rectangle_type(MPITYPE, 1, ddescA->nb, ddescA->nb*ddescA->nb, &pivot_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(pivot_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(getrf_sd_PIVOT_VECT_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &pivot_ddt);

    dplasma_aux_create_rectangle_type(MPITYPE, IB+1, ddescA->nb, ddescA->nb*ddescA->nb, &littlel_pivot_ddt);
#if defined(USE_MPI)
    MPI_Type_get_extent(littlel_pivot_ddt, &lb, &extent);
#endif  /* defined(USE_MPI) */
    dague_arena_construct(object->arenas[DAGUEprefix(getrf_sd_LITTLE_L_PIVOT_VECT_ARENA)], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, &littlel_pivot_ddt);

    return (dague_object_t*)object;
}
