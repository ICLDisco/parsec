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

#include "generated/zherbt_L.h"

dague_object_t *
dplasma_zherbt_New( PLASMA_enum uplo,
                    int ib,
                    PLASMA_desc descA,
                    tiled_matrix_desc_t *A,
                    PLASMA_desc descT,
                    tiled_matrix_desc_t *T)
{
    dague_object_t *dague_zherbt = NULL;
    dague_memory_pool_t *pool[4];

    if( PlasmaLower != uplo ) {
        dplasma_error("dplasma_zherbt_New", "illegal value of uplo");
        return NULL;
    }

    pool[0] = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));  /* tau */
    dague_private_memory_init( pool[0], descT.nb * sizeof(Dague_Complex64_t) );
    pool[1] = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));  /* work */
    dague_private_memory_init( pool[1], descT.mb * descT.nb * sizeof(Dague_Complex64_t) );
    pool[2] = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));  /* work for HERFB1 */
    dague_private_memory_init( pool[2], 2 * descT.nb * descT.nb * sizeof(Dague_Complex64_t) );
    pool[3] = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));  /* work for the TSMQRLR */
    dague_private_memory_init( pool[3], 4 * descT.nb * descT.nb * sizeof(Dague_Complex64_t) );

    if( PlasmaLower == uplo ) {
        dague_zherbt = (dague_object_t *)dague_zherbt_L_new(uplo, 
                                                            descA, &A->super, 
                                                            descT, &T->super,
                                                            ib,
                                                            pool[0], pool[1], pool[2], pool[3]);
#if 0
        dplasma_add2arena_rectangle( ((dague_zherbt_L_object_t *)dague_zherbt)->arenas[DAGUE_zherbt_L_LITTLE_T_ARENA], 
                                     descT.mb*descT.nb*sizeof(Dague_Complex64_t),
                                     DAGUE_ARENA_ALIGNMENT_SSE,
                                     MPI_DOUBLE_COMPLEX, descT.mb, descT.nb, -1);
#endif
    }

    return dague_zherbt;
}

void dplasma_zherbt_Destruct( dague_object_t *o )
{
    dague_zherbt_L_object_t *dague_zherbt = (dague_zherbt_L_object_t *)o;
    
    if( PlasmaLower == dague_zherbt->uplo ) {
        free( dague_zherbt->pool_0 );
        free( dague_zherbt->pool_1 );
        free( dague_zherbt->pool_2 );
        free( dague_zherbt->pool_3 );
        dague_zherbt_L_destroy(dague_zherbt);
    }
}
