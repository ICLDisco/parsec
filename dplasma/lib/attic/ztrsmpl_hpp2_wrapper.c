/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "parsec.h"
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "ztrsmpl_hpp2.h"

parsec_handle_t* dplasma_ztrsmpl_hpp2_New( qr_piv_t *qrpiv,
                                            tiled_matrix_desc_t *A,
                                            tiled_matrix_desc_t *B,
                                            tiled_matrix_desc_t *IPIV,
                                            tiled_matrix_desc_t *LT,
                                            tiled_matrix_desc_t *LT2,
                                            int* INFO )
{
    parsec_ztrsmpl_hpp2_handle_t* object;
    int ib = LT->mb;

    /*
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf
     */

    object = parsec_ztrsmpl_hpp2_new( *A,  (parsec_ddesc_t*)A,
                                     *B,  (parsec_ddesc_t*)B,
                                            (parsec_ddesc_t*)IPIV,
                                       *LT, (parsec_ddesc_t*)LT,
                                       *LT2, (parsec_ddesc_t*)LT2,
                                       qrpiv, ib,
                                       NULL, NULL,
                                       INFO);

    object->p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( object->p_work, ib * LT->nb * sizeof(parsec_complex64_t) );

    object->p_tau = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( object->p_tau, LT->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[PARSEC_ztrsmpl_hpp2_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[PARSEC_ztrsmpl_hpp2_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* IPIV */
    dplasma_add2arena_rectangle( object->arenas[PARSEC_ztrsmpl_hpp2_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    return (parsec_handle_t*)object;
}

int dplasma_ztrsmpl_hpp2( parsec_context_t *parsec,
                            qr_piv_t *qrpiv,
                            tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *B,
                            tiled_matrix_desc_t *IPIV,
                            tiled_matrix_desc_t *LT,
                            tiled_matrix_desc_t *LT2,
                            int* INFO )
{
    parsec_handle_t *parsec_ztrsmpl_hpp2 = NULL;

    parsec_ztrsmpl_hpp2 = dplasma_ztrsmpl_hpp2_New(qrpiv, A, B, IPIV, LT, LT2, INFO);

    parsec_enqueue(parsec, (parsec_handle_t*)parsec_ztrsmpl_hpp2);
    dplasma_progress(parsec);

    dplasma_ztrsmpl_hpp2_Destruct( parsec_ztrsmpl_hpp2 );
    return 0;
}

void
dplasma_ztrsmpl_hpp2_Destruct( parsec_handle_t *o )
{
    parsec_ztrsmpl_hpp2_handle_t *parsec_ztrsmpl_hpp2 = (parsec_ztrsmpl_hpp2_handle_t *)o;

    dplasma_datatype_undefine_type( &(parsec_ztrsmpl_hpp2->arenas[PARSEC_ztrsmpl_hpp2_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(parsec_ztrsmpl_hpp2->arenas[PARSEC_ztrsmpl_hpp2_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(parsec_ztrsmpl_hpp2->arenas[PARSEC_ztrsmpl_hpp2_PIVOT_ARENA     ]->opaque_dtt) );

    parsec_private_memory_fini( parsec_ztrsmpl_hpp2->p_work );
    parsec_private_memory_fini( parsec_ztrsmpl_hpp2->p_tau  );

    free( parsec_ztrsmpl_hpp2->p_work );
    free( parsec_ztrsmpl_hpp2->p_tau  );

    PARSEC_INTERNAL_HANDLE_DESTRUCT(o);
}

