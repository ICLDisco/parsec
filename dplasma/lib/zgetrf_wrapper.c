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

#include "generated/zgetrf.h"
#include "generated/zgetrf_sd.h"

dague_object_t* dplasma_zgetrf_New(tiled_matrix_desc_t* ddescA,
                                   tiled_matrix_desc_t *L,
                                   tiled_matrix_desc_t *IPIV,
                                   int* INFO)
{
    dague_zgetrf_object_t* object;
    dague_remote_dep_datatype_t tile_ddt, lower_ddt, upper_ddt, pivot_ddt, littlel_ddt;
    int pri_change = dplasma_aux_get_priority( "GETRF", ddescA );
#if defined(HAVE_MPI)
    MPI_Aint lb = 0, extent = 0;
#else
    int64_t extent = 0;
#endif  /* defined(HAVE_MPI) */

    object = dague_zgetrf_new( (dague_ddesc_t*)L, (dague_ddesc_t*)IPIV, (dague_ddesc_t*)ddescA,
                               ddescA->n, ddescA->nb, ddescA->nt, L->mb, NULL, INFO );
    object->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->work_pool, L->mb * L->nb * sizeof(Dague_Complex64_t) );

    /* datatype for A */
    dplasma_datatype_define_tile(MPI_DOUBLE_COMPLEX, ddescA->nb, &tile_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(tile_ddt, &lb, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_DEFAULT_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, tile_ddt);

    /* datatype for A lower triangle */
    dplasma_datatype_define_lower(MPI_DOUBLE_COMPLEX, ddescA->nb, 0, &lower_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(lower_ddt, &lb, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_LOWER_TILE_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, lower_ddt);

    /* datatype for A upper triangle */
    dplasma_datatype_define_upper(MPI_DOUBLE_COMPLEX, ddescA->nb, 1, &upper_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(upper_ddt, &lb, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_UPPER_TILE_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, upper_ddt);

    /* datatype for IPIV */
    dplasma_datatype_define_rectangle(MPI_INT, IPIV->mb, IPIV->nb, -1, &pivot_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(pivot_ddt, &lb, &extent);
#else
    extent = IPIV->mb * IPIV->nb * sizeof(int);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_PIVOT_VECT_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, pivot_ddt);

    /* datatype for dL */
    dplasma_datatype_define_rectangle(MPI_INT, L->mb, L->nb, -1, &littlel_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(littlel_ddt, &lb, &extent);
#else
    extent = L->mb * L->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_LITTLE_L_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, littlel_ddt);

    return (dague_object_t*)object;
}

int dplasma_zgetrf( dague_context_t *dague, tiled_matrix_desc_t *A, 
                    tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV ) 
{
    dague_object_t *dague_zgetrf = NULL;

    int info;
    dague_zgetrf = dplasma_zgetrf_New(A, L, IPIV, &info);

    dague_enqueue( dague, (dague_object_t*)dague_zgetrf);
    dague_progress(dague);

    return info;
}

dague_object_t* dplasma_zgetrf_sd_New( tiled_matrix_desc_t* ddescA,
                                       tiled_matrix_desc_t *LIPIV,
                                       int* INFO)
{
    dague_zgetrf_sd_object_t* object;
    dague_remote_dep_datatype_t tile_ddt, lower_ddt, upper_ddt, pivot_ddt, littlel_pivot_ddt;
    int pri_change = dplasma_aux_get_priority( "GETRF", ddescA );
#if defined(HAVE_MPI)
    MPI_Aint lb = 0, extent = 0;
#else
    int64_t extent = 0;
#endif  /* defined(HAVE_MPI) */

    object = dague_zgetrf_sd_new( (dague_ddesc_t*)LIPIV, (dague_ddesc_t*)ddescA,
                                  ddescA->n, ddescA->nb, ddescA->nt, (LIPIV->mb-1), NULL, INFO );
    object->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));

    /* TODO: check if we should use mb-1 or mb here */
    dague_private_memory_init( object->work_pool, (LIPIV->mb-1) * LIPIV->nb * sizeof(Dague_Complex64_t) );

    /* datatype for A */
    dplasma_datatype_define_tile(MPI_DOUBLE_COMPLEX, ddescA->nb, &tile_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(tile_ddt, &lb, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_sd_DEFAULT_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, tile_ddt);

    /* datatype for A lower triangle */
    dplasma_datatype_define_lower(MPI_DOUBLE_COMPLEX, ddescA->nb, 0, &lower_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(lower_ddt, &lb, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_sd_LOWER_TILE_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, lower_ddt);

    /* datatype for A upper triangle */
    dplasma_datatype_define_upper(MPI_DOUBLE_COMPLEX, ddescA->nb, 1, &upper_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(upper_ddt, &lb, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_sd_UPPER_TILE_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, upper_ddt);

    /* datatype for IPIV */
    dplasma_datatype_define_rectangle(MPI_INT, 1, LIPIV->nb, -1, &pivot_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(pivot_ddt, &lb, &extent);
#else
    extent = LIPIV->nb * sizeof(int);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_sd_PIVOT_VECT_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, pivot_ddt);

    /* datatype for IPIV and dL combined */
    dplasma_datatype_define_rectangle(MPI_DOUBLE_COMPLEX, LIPIV->mb, LIPIV->nb, LIPIV->mb*LIPIV->nb, &littlel_pivot_ddt);
#if defined(HAVE_MPI)
    MPI_Type_get_extent(littlel_pivot_ddt, &lb, &extent);
#else
    extent = LIPIV->nb * LIPIV->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_sd_LITTLE_L_PIVOT_VECT_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, littlel_pivot_ddt);
    
    return (dague_object_t*)object;
}

