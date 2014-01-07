/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "zgerbb_1.h"
#include "zgerbb_2.h"

dague_object_t* dplasma_zgerbb_New( tiled_matrix_desc_t *A,
                                    tiled_matrix_desc_t *T,
                                    int ib )
{
    dague_object_t* __dague_object;
    dague_memory_pool_t *pool[2];

    if( A->m >= A->n ) {
        pool[0] = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));  /* tau */
        dague_private_memory_init( pool[0], zgerbb_1_pool_0_SIZE );
        pool[1] = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));  /* tau */
        dague_private_memory_init( pool[1], zgerbb_1_pool_1_SIZE );

        dague_sgerbb_1_object_t *obj = dague_sgerbb_1_new(PLASMA_desc desc_A,
                                                          A,
                                                          PLASMA_desc desc_T,
                                                          T,
                                                          pool[0], pool[1],
                                                          ib);
        __dague_object = (dague_object_t*)obj;
    } else {
        pool[0] = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));  /* tau */
        dague_private_memory_init( pool[0], zgerbb_2_pool_0_SIZE );
        pool[1] = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));  /* tau */
        dague_private_memory_init( pool[1], zgerbb_2_pool_1_SIZE );

        dague_sgerbb_2_object_t *obj = dague_sgerbb_2_new(PLASMA_desc desc_A,
                                                          A,
                                                          PLASMA_desc desc_T,
                                                          T,
                                                          pool[0], pool[1],
                                                          ib);
        __dague_object = (dague_object_t*)obj;
    }
    return __dague_object;
}

void
dplasma_zgerbb_Destruct( dague_object_t *o )
{
    dague_zgerbb_object_t *dague_zgerbb = (dague_zgerbb_object_t *)o;

    dague_private_memory_fini( dague_zgerbb->pool_0 );
    dague_private_memory_fini( dague_zgerbb->pool_1 );
    free( dague_zgerbb->p_work );
    free( dague_zgerbb->p_tau  );
 
    DAGUE_INTERNAL_OBJECT_DESTRUCT(dague_zgerbb);
}

int dplasma_zgerbb( dague_context_t *dague, 
                          tiled_matrix_desc_t *A, 
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT) 
{
    dague_object_t *dague_zgerbb = NULL;

    dague_zgerbb = dplasma_zgerbb_New(A, TS, TT);

    dague_enqueue(dague, (dague_object_t*)dague_zgerbb);
    dplasma_progress(dague);

    dplasma_zgerbb_Destruct( dague_zgerbb );
    return 0;
}
