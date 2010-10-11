#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "trmm.h"

dague_object_t * 
DAGUEprefix(trmm_New)(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
		      TYPENAME alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B)
{

    dague_object_t *dague_trmm = NULL;

    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
		dague_trmm = (dague_object_t*)dagueprefix(trmm_LLN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B, 
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n, 
		    side, uplo, trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dagueprefix(trmm_LLT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    side, uplo, trans, diag, alpha);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dagueprefix(trmm_LUN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    side, uplo, trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dagueprefix(trmm_LUT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    side, uplo, trans, diag, alpha);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dagueprefix(trmm_RLN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    side, uplo, trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dagueprefix(trmm_RLT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    side, uplo, trans, diag, alpha);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dagueprefix(trmm_RUN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    side, uplo, trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dagueprefix(trmm_RUT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
		    B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
		    side, uplo, trans, diag, alpha);
            }
        }
    }

    printf("TRMM A:%ux%u (%ux%u) and B:%ux%u (%ux%u) has %u tasks to run.\n", 
           A->super.m, A->super.n, A->super.mt, A->super.nt, 
           B->super.m, B->super.n, B->super.mt, B->super.nt, 
           dague_trmm->nb_local_tasks);
    
    return dague_trmm;
}

