/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "trsm_wrapper.h"

dague_object_t * 
DAGUEprefix(trsm_New)(const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
		      const TYPENAME alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B)
{
    dague_object_t *dague_trsm = NULL;

    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
		dague_trsm = (dague_object_t*)dagueprefix(trsm_LLN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B, 
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n, 
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dagueprefix(trsm_LLT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    trans, diag, alpha);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dagueprefix(trsm_LUN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dagueprefix(trsm_LUT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    trans, diag, alpha);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dagueprefix(trsm_RLN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dagueprefix(trsm_RLT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    trans, diag, alpha);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dagueprefix(trsm_RUN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dagueprefix(trsm_RUT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    trans, diag, alpha);
            }
        }
    }

    printf("TRSM A:%ux%u (%ux%u) and B:%ux%u (%ux%u) has %u tasks to run.\n", 
           A->m, A->n, A->mt, A->nt, 
           B->m, B->n, B->mt, B->nt, 
           dague_trsm->nb_local_tasks);
    
    return dague_trsm;
}

