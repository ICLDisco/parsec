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

#include "generated/zgemm_NN.h"
#include "generated/zgemm_NT.h"
#include "generated/zgemm_TN.h"
#include "generated/zgemm_TT.h"

dague_object_t*
dplasma_zgemm_New( const int transA, const int transB,
                   const Dague_Complex64_t alpha, const tiled_matrix_desc_t* ddescA, const tiled_matrix_desc_t* ddescB,
                   const Dague_Complex64_t beta,  tiled_matrix_desc_t* ddescC)
{
    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            return (dague_object_t*)dague_zgemm_NN_new((dague_ddesc_t*)ddescC,
                                                       (dague_ddesc_t*)ddescB,
                                                       (dague_ddesc_t*)ddescA,
                                                       ddescA->nb,
                                                       ddescC->mt, ddescC->nt, ddescA->nt,
                                                       transA, transB,
                                                       alpha, beta);
        } else {
            return (dague_object_t*)dague_zgemm_NT_new((dague_ddesc_t*)ddescC,
                                                       (dague_ddesc_t*)ddescB,
                                                       (dague_ddesc_t*)ddescA,
                                                       ddescA->nb,
                                                       ddescC->mt, ddescC->nt, ddescA->nt,
                                                       transA, transB,
                                                       alpha, beta);
        }
    } else {
        if( PlasmaNoTrans == transB ) {
            return (dague_object_t*)dague_zgemm_TN_new((dague_ddesc_t*)ddescC,
                                                       (dague_ddesc_t*)ddescB,
                                                       (dague_ddesc_t*)ddescA,
                                                       ddescA->nb,
                                                       ddescC->mt, ddescC->nt, ddescA->mt,
                                                       transA, transB,
                                                       alpha, beta);
        } else {
            return (dague_object_t*)dague_zgemm_TT_new((dague_ddesc_t*)ddescC,
                                                       (dague_ddesc_t*)ddescB,
                                                       (dague_ddesc_t*)ddescA,
                                                       ddescA->nb,
                                                       ddescC->mt, ddescC->nt, ddescA->nt,
                                                       transA, transB,
                                                       alpha, beta);
        }
    }
}

void
dplasma_zgemm( dague_context_t *dague, const int transA, const int transB,
               const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
               const Dague_Complex64_t beta,  tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zgemm = NULL;

    dague_zgemm = dplasma_zgemm_New(transA, transB, 
				    alpha, A, B,
				    beta, C);

    dague_enqueue( dague, (dague_object_t*)dague_zgemm);
    dague_progress(dague);
}
