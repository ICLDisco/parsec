/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"

#define HIGH_TO_LOW 0
#define LOW_TO_HIGH 1

static void multilevel_zgebmm(dague_context_t *dague, tiled_matrix_desc_t* B, PLASMA_Complex64_t *U_but_vec, int level, int trans, int order, int *info){
    int cur_level, L;
    dague_object_t **op;

    for( L=0; L <= level; L++ ){
        int i_block, j_block, block_count;

        if( LOW_TO_HIGH == order ){
            cur_level = L;
        }else{
            cur_level = level-L;
        }
        block_count = 1<<cur_level;

#if defined(DEBUG_BUTTERFLY)
        printf(" ===== Applying zgebmm() at level: %d\n",cur_level);
        fflush(stdout);
#endif

        op = (dague_object_t **)calloc( block_count*block_count, sizeof(dague_object_t *));

        for(i_block=0; i_block < block_count; i_block++){
            for(j_block=0; j_block < block_count; j_block++){
                op[i_block*block_count+j_block] = dplasma_zgebmm_New( B, U_but_vec, i_block, j_block, cur_level, trans, info);
                dague_enqueue(dague, op[i_block*block_count+j_block]);
            }
        }

        dplasma_progress(dague);

        for(i_block=0; i_block < block_count; i_block++){
            for(j_block=0; j_block < block_count; j_block++){
                dplasma_zgebmm_Destruct( op[i_block*block_count+j_block] );
            }
        }

        free(op);
    }
}

int
dplasma_zhetrs(dague_context_t *dague, int uplo, const tiled_matrix_desc_t* A, tiled_matrix_desc_t* B, PLASMA_Complex64_t *U_but_vec, int level)
{
    int info;
#if defined(DEBUG_BUTTERFLY)
    int i;
#endif

    if( uplo != PlasmaLower ){
        dplasma_error("dplasma_zhetrs", "illegal value for \"uplo\".  Only PlasmaLower is currently supported");
    }

#if defined(DEBUG_BUTTERFLY)
    for(i=0; i<A->lm; i++){
        printf("U[%d]: %lf\n",i,creal(U_but_vec[i]));
    }
#endif
    // B = U_but_vec^T * B 
    multilevel_zgebmm(dague, B, U_but_vec, level, PlasmaConjTrans, HIGH_TO_LOW, &info);

    dplasma_ztrsm( dague, PlasmaLeft, uplo, (uplo == PlasmaUpper) ? PlasmaConjTrans : PlasmaNoTrans, PlasmaUnit, 1.0, A, B );
    dplasma_ztrdsm( dague, A, B );
    dplasma_ztrsm( dague, PlasmaLeft, uplo, (uplo == PlasmaUpper) ? PlasmaNoTrans : PlasmaConjTrans, PlasmaUnit, 1.0, A, B );

    // X = U_but_vec * X  (here X is B)
    multilevel_zgebmm(dague, B, U_but_vec, level, PlasmaNoTrans, LOW_TO_HIGH, &info);

    return 0;
}

