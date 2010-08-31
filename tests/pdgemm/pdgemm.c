/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "dague.h"
#include "data_dist/matrix/matrix.h"
#include "pdgemm.h"
#include "dgemm_NN.h"
#include "dgemm_NT.h"
#include "dgemm_TN.h"
#include "dgemm_TT.h"
#include "plasma.h"

dague_object_t* dague_pdgemm_new( int TRANSA, int TRANSB,
                                  int M, int N, int K,
                                  float ALPHA, const tiled_matrix_desc_t* ddescA,
                                  const tiled_matrix_desc_t* ddescB,
                                  float BETA, tiled_matrix_desc_t* ddescC)
{
    if( PlasmaNoTrans == TRANSA ) {
        if( PlasmaNoTrans == TRANSB ) {
            return (dague_object_t*)dague_dgemm_NN_new((dague_ddesc_t*)ddescC,
                                                       (dague_ddesc_t*)ddescB,
                                                       (dague_ddesc_t*)ddescA,
                                                       ddescA->nb,
                                                       M, N, K,
                                                       TRANSA, TRANSB,
                                                       ALPHA, BETA);
        } else {
            return (dague_object_t*)dague_dgemm_NT_new((dague_ddesc_t*)ddescC,
                                                       (dague_ddesc_t*)ddescB,
                                                       (dague_ddesc_t*)ddescA,
                                                       ddescA->nb,
                                                       M, N, K,
                                                       TRANSA, TRANSB,
                                                       ALPHA, BETA);
        }
    } else {
        if( PlasmaNoTrans == TRANSB ) {
            return (dague_object_t*)dague_dgemm_TN_new((dague_ddesc_t*)ddescC,
                                                       (dague_ddesc_t*)ddescB,
                                                       (dague_ddesc_t*)ddescA,
                                                       ddescA->nb,
                                                       M, N, K,
                                                       TRANSA, TRANSB,
                                                       ALPHA, BETA);
        } else {
            return (dague_object_t*)dague_dgemm_TT_new((dague_ddesc_t*)ddescC,
                                                       (dague_ddesc_t*)ddescB,
                                                       (dague_ddesc_t*)ddescA,
                                                       ddescA->nb,
                                                       M, N, K,
                                                       TRANSA, TRANSB,
                                                       ALPHA, BETA);
        }
    }
}
