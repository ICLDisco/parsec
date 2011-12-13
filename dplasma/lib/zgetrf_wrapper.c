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

/****************************************************************/
/*
 * Single data version grouping L and IPIV in L
 */
dague_object_t* dplasma_zgetrf_sd_New( tiled_matrix_desc_t *A,
                                       tiled_matrix_desc_t *L,
                                       int* INFO)
{
    int Lmb = L->mb-1;
    dague_zgetrf_sd_object_t *dague_getrf_sd;

    dague_getrf_sd = dague_zgetrf_sd_new( *A, (dague_ddesc_t*)A, 
                                          *L, (dague_ddesc_t*)L, 
                                          NULL, INFO, Lmb);

    dague_getrf_sd->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( dague_getrf_sd->work_pool, Lmb * L->nb * sizeof(Dague_Complex64_t) );

    /* A */
    dplasma_add2arena_tile( dague_getrf_sd->arenas[DAGUE_zgetrf_sd_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* Lower part of A without diagonal part */
    dplasma_add2arena_lower( dague_getrf_sd->arenas[DAGUE_zgetrf_sd_LOWER_TILE_ARENA], 
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );
    
    /* Upper part of A with diagonal part */
    dplasma_add2arena_upper( dague_getrf_sd->arenas[DAGUE_zgetrf_sd_UPPER_TILE_ARENA], 
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* IPIV */
    dplasma_add2arena_rectangle( dague_getrf_sd->arenas[DAGUE_zgetrf_sd_PIVOT_ARENA], 
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    /* L */
    dplasma_add2arena_rectangle( dague_getrf_sd->arenas[DAGUE_zgetrf_sd_L_PIVOT_ARENA], 
                                 L->mb*L->nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, L->mb, L->nb, -1);

    return (dague_object_t*)dague_getrf_sd;
}

void
dplasma_zgetrf_sd_Destruct( dague_object_t *o )
{
    dague_zgetrf_sd_object_t *dague_zgetrf = (dague_zgetrf_sd_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_sd_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_sd_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_sd_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_sd_PIVOT_ARENA     ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_sd_L_PIVOT_ARENA   ]->opaque_dtt) );
      
    dague_private_memory_fini( dague_zgetrf->work_pool );
    free( dague_zgetrf->work_pool );
 
    dague_zgetrf_sd_destroy(dague_zgetrf);
}

