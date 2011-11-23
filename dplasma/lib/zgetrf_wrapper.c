/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "zgetrf.h"
#include "zgetrf_sd.h"

dague_object_t* dplasma_zgetrf_New(tiled_matrix_desc_t *A,
                                   tiled_matrix_desc_t *L,
                                   tiled_matrix_desc_t *IPIV,
                                   int *INFO)
{
    dague_zgetrf_object_t *dague_getrf;

    dague_getrf = dague_zgetrf_new( *A, (dague_ddesc_t*)A, 
                                    *L, (dague_ddesc_t*)L, 
                                    (dague_ddesc_t*)IPIV, 
                                    NULL, INFO, L->mb);

    dague_getrf->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( dague_getrf->work_pool, L->mb * L->nb * sizeof(Dague_Complex64_t) );

    /* A */
    dplasma_add2arena_tile( dague_getrf->arenas[DAGUE_zgetrf_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* Lower part of A without diagonal part */
    dplasma_add2arena_lower( dague_getrf->arenas[DAGUE_zgetrf_LOWER_TILE_ARENA], 
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );
    
    /* Upper part of A with diagonal part */
    dplasma_add2arena_upper( dague_getrf->arenas[DAGUE_zgetrf_UPPER_TILE_ARENA], 
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* IPIV */
    dplasma_add2arena_rectangle( dague_getrf->arenas[DAGUE_zgetrf_PIVOT_ARENA], 
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    /* L */
    dplasma_add2arena_rectangle( dague_getrf->arenas[DAGUE_zgetrf_SMALL_L_ARENA], 
                                 L->mb*L->nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, L->mb, L->nb, -1);

    return (dague_object_t*)dague_getrf;
}

void
dplasma_zgetrf_Destruct( dague_object_t *o )
{
    dague_zgetrf_object_t *dague_zgetrf = (dague_zgetrf_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_SMALL_L_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_PIVOT_ARENA     ]->opaque_dtt) );
      
    dague_private_memory_fini( dague_zgetrf->work_pool );
    free( dague_zgetrf->work_pool );
 
    dague_zgetrf_destroy(dague_zgetrf);
}

int dplasma_zgetrf( dague_context_t *dague, tiled_matrix_desc_t *A, 
                    tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV ) 
{
    dague_object_t *dague_zgetrf = NULL;

    int info = 0;
    dague_zgetrf = dplasma_zgetrf_New(A, L, IPIV, &info);

    dague_enqueue( dague, (dague_object_t*)dague_zgetrf);
    dplasma_progress(dague);

    return info;
}

dague_object_t* dplasma_zgetrf_sd_New( tiled_matrix_desc_t* ddescA,
                                       tiled_matrix_desc_t *LIPIV,
                                       int* INFO)
{
    dague_zgetrf_sd_object_t* object;
    dague_remote_dep_datatype_t tile_ddt, lower_ddt, upper_ddt, pivot_ddt, littlel_pivot_ddt;
    //int pri_change = dplasma_aux_get_priority( "GETRF", ddescA );
#if defined(HAVE_MPI)
    MPI_Aint extent = 0;
#else
    int64_t extent = 0;
#endif  /* defined(HAVE_MPI) */

    object = dague_zgetrf_sd_new( (dague_ddesc_t*)LIPIV, (dague_ddesc_t*)ddescA,
                                  ddescA->n, ddescA->nb, ddescA->nt, (LIPIV->mb-1), NULL, INFO );
    object->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));

    /* TODO: check if we should use mb-1 or mb here */
    dague_private_memory_init( object->work_pool, (LIPIV->mb) * LIPIV->nb * sizeof(Dague_Complex64_t) );

    /* datatype for A */
    dplasma_datatype_define_tile(MPI_DOUBLE_COMPLEX, ddescA->nb, &tile_ddt);
#if defined(HAVE_MPI)
    dplasma_get_extent(tile_ddt, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_sd_DEFAULT_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, tile_ddt);

    /* datatype for A lower triangle */
    dplasma_datatype_define_lower(MPI_DOUBLE_COMPLEX, ddescA->nb, 0, &lower_ddt);
#if defined(HAVE_MPI)
    dplasma_get_extent(lower_ddt, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_sd_LOWER_TILE_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, lower_ddt);

    /* datatype for A upper triangle */
    dplasma_datatype_define_upper(MPI_DOUBLE_COMPLEX, ddescA->nb, 1, &upper_ddt);
#if defined(HAVE_MPI)
    dplasma_get_extent(upper_ddt, &extent);
#else
    extent = ddescA->mb * ddescA->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_sd_UPPER_TILE_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, upper_ddt);

    /* datatype for IPIV */
    dplasma_datatype_define_rectangle(MPI_INT, 1, LIPIV->nb, -1, &pivot_ddt);
#if defined(HAVE_MPI)
    dplasma_get_extent(pivot_ddt, &extent);
#else
    extent = LIPIV->nb * sizeof(int);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_sd_PIVOT_VECT_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, pivot_ddt);

    /* datatype for IPIV and dL combined */
    dplasma_datatype_define_rectangle(MPI_DOUBLE_COMPLEX, LIPIV->mb, LIPIV->nb, LIPIV->mb*LIPIV->nb, &littlel_pivot_ddt);
#if defined(HAVE_MPI)
    dplasma_get_extent(littlel_pivot_ddt, &extent);
#else
    extent = LIPIV->nb * LIPIV->nb * sizeof(Dague_Complex64_t);
#endif  /* defined(HAVE_MPI) */
    dague_arena_construct(object->arenas[DAGUE_zgetrf_sd_LITTLE_L_PIVOT_VECT_ARENA], extent,
                          DAGUE_ARENA_ALIGNMENT_SSE, littlel_pivot_ddt);
    
    return (dague_object_t*)object;
}

