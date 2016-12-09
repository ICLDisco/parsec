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

#include "zgetrf_hpp.h"

parsec_handle_t* dplasma_zgetrf_hpp_New( qr_piv_t *qrpiv,
                                            tiled_matrix_desc_t *A,
                                            tiled_matrix_desc_t *IPIV,
                                            tiled_matrix_desc_t *LT,
                                            int* INFO )
{
    parsec_zgetrf_hpp_handle_t* object;
    int ib = LT->mb;

    /*
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf
     */

    object = parsec_zgetrf_hpp_new( *A,  (parsec_ddesc_t*)A,
                                   (parsec_ddesc_t*)IPIV,
                                   *LT, (parsec_ddesc_t*)LT,
                                   qrpiv, ib,
                                   NULL, NULL,
                                   INFO);

    object->p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( object->p_work, ib * LT->nb * sizeof(parsec_complex64_t) );

    object->p_tau = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( object->p_tau, LT->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[PARSEC_zgetrf_hpp_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( object->arenas[PARSEC_zgetrf_hpp_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* IPIV */
    dplasma_add2arena_rectangle( object->arenas[PARSEC_zgetrf_hpp_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[PARSEC_zgetrf_hpp_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[PARSEC_zgetrf_hpp_LITTLE_T_ARENA],
                                 LT->mb*LT->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, LT->mb, LT->nb, -1);

    return (parsec_handle_t*)object;
}

int dplasma_zgetrf_hpp( parsec_context_t *parsec,
                            qr_piv_t *qrpiv,
                            tiled_matrix_desc_t *A,
                            tiled_matrix_desc_t *IPIV,
                            tiled_matrix_desc_t *LT,
                            int* INFO )
{
    parsec_handle_t *parsec_zgetrf_hpp = NULL;

    parsec_zgetrf_hpp = dplasma_zgetrf_hpp_New(qrpiv, A, IPIV, LT, INFO);

    parsec_enqueue(parsec, (parsec_handle_t*)parsec_zgetrf_hpp);
    dplasma_progress(parsec);

    dplasma_zgetrf_hpp_Destruct( parsec_zgetrf_hpp );
    return 0;
}

void
dplasma_zgetrf_hpp_Destruct( parsec_handle_t *o )
{
    parsec_zgetrf_hpp_handle_t *parsec_zgetrf_hpp = (parsec_zgetrf_hpp_handle_t *)o;

    dplasma_datatype_undefine_type( &(parsec_zgetrf_hpp->arenas[PARSEC_zgetrf_hpp_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(parsec_zgetrf_hpp->arenas[PARSEC_zgetrf_hpp_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(parsec_zgetrf_hpp->arenas[PARSEC_zgetrf_hpp_PIVOT_ARENA     ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(parsec_zgetrf_hpp->arenas[PARSEC_zgetrf_hpp_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(parsec_zgetrf_hpp->arenas[PARSEC_zgetrf_hpp_LITTLE_T_ARENA  ]->opaque_dtt) );

    parsec_private_memory_fini( parsec_zgetrf_hpp->p_work );
    parsec_private_memory_fini( parsec_zgetrf_hpp->p_tau  );

    free( parsec_zgetrf_hpp->p_work );
    free( parsec_zgetrf_hpp->p_tau  );

    PARSEC_INTERNAL_HANDLE_DESTRUCT(o);
}

