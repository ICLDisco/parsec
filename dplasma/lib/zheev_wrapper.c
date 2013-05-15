/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zherbt_L.h"
#include "zherbt_U.h"
#include "data_dist/matrix/diag_band_to_rect.h"
#include "zhbrbt_L.h"
#include "zhbrbt_U.h"

               const int transA, const int transB,
               const dague_complex64_t alpha, const tiled_matrix_desc_t *A,
                                              const tiled_matrix_desc_t *B,
               const dague_complex64_t beta,        tiled_matrix_desc_t *C)


/*    SUBROUTINE PZHEEV( JOBZ, UPLO, N, A, IA, JA, DESCA, W, Z, IZ, JZ,
     $                   DESCZ, WORK, LWORK, RWORK, LRWORK, INFO ) */
dague_object_t*
dplasma_zheev_New( PLASMA_Enum jobz, PLASMA_Enum uplo,
                    tiled_matrix_desc_t* A, 
                    tiled_matrix_desc_t* Z,
                    int* info )
{
    dague_object_t* zheev_object;
    dague_arena_t* arena;

    /* Check input arguments */
    if( jobz != PlasmaNoVec && jobz != PlasmaVec ) {
        dplasma_error("DPLASMA_zheev", "illegal value of jobz");
        *info = -1;
        return NULL;
    }
    if( uplo != PlasmaLower && uplo != PlasmaUpper ) {
        dplasma_error("DPLASMA_zheev", "illegal value of uplo");
        *info = -2;
        return NULL;
    }

    /* TODO: remove this when implemented */
    if( jobz == PlasmaVec ) {
        dplasma_error("DPLASMA_zheev", "PlasmaVec jobz is not implemented (yet)");
        *info = -1;
        return NULL;
    }

    if( PlasmaUpper == uplo )
        dague_zherbt_U_object = dague_zherbt_U_object_new()


    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_NN_object_t* object;
            object = dague_zgemm_NN_new(transA, transB, alpha, beta,
                                        *A, (dague_ddesc_t*)A,
                                        *B, (dague_ddesc_t*)B,
                                        *C, (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_NN_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        } else {
            dague_zgemm_NT_object_t* object;
            object = dague_zgemm_NT_new(transA, transB, alpha, beta,
                                        *A, (dague_ddesc_t*)A,
                                        *B, (dague_ddesc_t*)B,
                                        *C, (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_NT_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        }
    } else {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_TN_object_t* object;
            object = dague_zgemm_TN_new(transA, transB, alpha, beta,
                                        *A, (dague_ddesc_t*)A,
                                        *B, (dague_ddesc_t*)B,
                                        *C, (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_TN_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        } else {
            dague_zgemm_TT_object_t* object;
            object = dague_zgemm_TT_new(transA, transB, alpha, beta,
                                        *A, (dague_ddesc_t*)A,
                                        *B, (dague_ddesc_t*)B,
                                        *C, (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_TT_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        }
    }

    dplasma_add2arena_tile(arena,
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return zheev_object;
}

void
dplasma_zheev_Destruct( dague_object_t *o )
{
    dplasma_datatype_undefine_type( &(((dague_zgemm_NN_object_t *)o)->arenas[DAGUE_zgemm_NN_DEFAULT_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

int
dplasma_zheev( dague_context_t *dague,
               const int transA, const int transB,
               const dague_complex64_t alpha, const tiled_matrix_desc_t *A,
                                              const tiled_matrix_desc_t *B,
               const dague_complex64_t beta,        tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zheev = NULL;

    /* Check input arguments */
    if ((transA != PlasmaNoTrans) && (transA != PlasmaTrans) && (transA != PlasmaConjTrans)) {
        dplasma_error("PLASMA_zgemm", "illegal value of transA");
        return -1;
    }
    if ((transB != PlasmaNoTrans) && (transB != PlasmaTrans) && (transB != PlasmaConjTrans)) {
        dplasma_error("PLASMA_zgemm", "illegal value of transB");
        return -2;
    }

    dague_zheev = dplasma_zheev_New(transA, transB,
                                    alpha, A, B,
                                    beta, C);

    if ( dague_zheev != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zheev);
        dplasma_progress(dague);
        dplasma_zheev_Destruct( dague_zheev );
        return 0;
    }
    else {
        return -101;
    }
}


