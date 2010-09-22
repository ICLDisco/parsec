#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "trsm.h"

dague_object_t * 
DAGUEprefix(trsm_New)(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
		      TYPENAME alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B)
{

    dague_object_t *dague_trsm = NULL;

    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
		dague_trsm = (dague_object_t*)dagueprefix(trsm_LLN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B, 
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n, 
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dagueprefix(trsm_LLT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    trans, diag, alpha);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dagueprefix(trsm_LUN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dagueprefix(trsm_LUT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    trans, diag, alpha);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dagueprefix(trsm_RLN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dagueprefix(trsm_RLT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    trans, diag, alpha);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dagueprefix(trsm_RUN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dagueprefix(trsm_RUT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    trans, diag, alpha);
            }
        }
    }

    printf("TRSM A:%ux%u (%ux%u) and B:%ux%u (%ux%u) has %u tasks to run.\n", 
           A->super.m, A->super.n, A->super.mt, A->super.nt, 
           B->super.m, B->super.n, B->super.mt, B->super.nt, 
           dague_trsm->nb_local_tasks);
    
    return dague_trsm;
}

