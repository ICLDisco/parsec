/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <core_blas.h>
#include "dplasma.h"
#include "dplasmatypes.h"

#define HIGH_TO_LOW 0
#define LOW_TO_HIGH 1

static void multilevel_zgebmm(parsec_context_t *parsec, parsec_tiled_matrix_dc_t* B, PLASMA_Complex64_t *U_but_vec, int level, int trans, int order, int *info){
    int cur_level, L;
    parsec_taskpool_t **op;

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

        op = (parsec_taskpool_t **)calloc( block_count*block_count, sizeof(parsec_taskpool_t *));

        for(i_block=0; i_block < block_count; i_block++){
            for(j_block=0; j_block < block_count; j_block++){
                op[i_block*block_count+j_block] = dplasma_zgebmm_New( B, U_but_vec, i_block, j_block, cur_level, trans, info);
                parsec_context_add_taskpool(parsec, op[i_block*block_count+j_block]);
            }
        }

        dplasma_wait_until_completion(parsec);

        for(i_block=0; i_block < block_count; i_block++){
            for(j_block=0; j_block < block_count; j_block++){
                dplasma_zgebmm_Destruct( op[i_block*block_count+j_block] );
            }
        }

        free(op);
    }
}

int
dplasma_zhetrs(parsec_context_t *parsec, int uplo, const parsec_tiled_matrix_dc_t* A, parsec_tiled_matrix_dc_t* B, PLASMA_Complex64_t *U_but_vec, int level)
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
    multilevel_zgebmm(parsec, B, U_but_vec, level, PlasmaConjTrans, HIGH_TO_LOW, &info);

    dplasma_ztrsm( parsec, PlasmaLeft, uplo, (uplo == PlasmaUpper) ? PlasmaConjTrans : PlasmaNoTrans, PlasmaUnit, 1.0, A, B );
    dplasma_ztrdsm( parsec, A, B );
    dplasma_ztrsm( parsec, PlasmaLeft, uplo, (uplo == PlasmaUpper) ? PlasmaNoTrans : PlasmaConjTrans, PlasmaUnit, 1.0, A, B );

    // X = U_but_vec * X  (here X is B)
    multilevel_zgebmm(parsec, B, U_but_vec, level, PlasmaNoTrans, LOW_TO_HIGH, &info);

    return 0;
}

