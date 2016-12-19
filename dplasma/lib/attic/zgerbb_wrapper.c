/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "parsec_internal.h"
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "zgerbb_1.h"
#include "zgerbb_2.h"

parsec_handle_t* dplasma_zgerbb_New( tiled_matrix_desc_t *A,
                                    tiled_matrix_desc_t *T,
                                    int ib )
{
    parsec_handle_t* __parsec_handle;
    parsec_memory_pool_t *pool[2];

    if( A->m >= A->n ) {
        pool[0] = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));  /* tau */
        parsec_private_memory_init( pool[0], zgerbb_1_pool_0_SIZE );
        pool[1] = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));  /* tau */
        parsec_private_memory_init( pool[1], zgerbb_1_pool_1_SIZE );

        parsec_sgerbb_1_handle_t *obj = parsec_sgerbb_1_new(PLASMA_desc desc_A,
                                                          A,
                                                          PLASMA_desc desc_T,
                                                          T,
                                                          pool[0], pool[1],
                                                          ib);
        __parsec_handle = (parsec_handle_t*)obj;
    } else {
        pool[0] = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));  /* tau */
        parsec_private_memory_init( pool[0], zgerbb_2_pool_0_SIZE );
        pool[1] = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));  /* tau */
        parsec_private_memory_init( pool[1], zgerbb_2_pool_1_SIZE );

        parsec_sgerbb_2_handle_t *obj = parsec_sgerbb_2_new(PLASMA_desc desc_A,
                                                          A,
                                                          PLASMA_desc desc_T,
                                                          T,
                                                          pool[0], pool[1],
                                                          ib);
        __parsec_handle = (parsec_handle_t*)obj;
    }
    return __parsec_handle;
}

void
dplasma_zgerbb_Destruct( parsec_handle_t *o )
{
    parsec_zgerbb_handle_t *parsec_zgerbb = (parsec_zgerbb_handle_t *)o;

    parsec_private_memory_fini( parsec_zgerbb->pool_0 );
    parsec_private_memory_fini( parsec_zgerbb->pool_1 );
    free( parsec_zgerbb->p_work );
    free( parsec_zgerbb->p_tau  );
 
    PARSEC_INTERNAL_HANDLE_DESTRUCT(parsec_zgerbb);
}

int dplasma_zgerbb( parsec_context_t *parsec, 
                    tiled_matrix_desc_t *A, 
                    tiled_matrix_desc_t *TS,
                    tiled_matrix_desc_t *TT) 
{
    parsec_handle_t *parsec_zgerbb = NULL;

    parsec_zgerbb = dplasma_zgerbb_New(A, TS, TT);

    parsec_enqueue(parsec, (parsec_handle_t*)parsec_zgerbb);
    dplasma_progress(parsec);

    dplasma_zgerbb_Destruct( parsec_zgerbb );
    return 0;
}
