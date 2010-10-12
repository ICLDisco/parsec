#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "trmm.h"

dague_object_t * 
DAGUEprefix(trmm_New)( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
		       const TYPENAME alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B)
{
    dague_object_t *dague_trmm = NULL;

    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
		dague_trmm = (dague_object_t*)dagueprefix(trmm_LLN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B, 
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n, 
		    side, uplo, trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dagueprefix(trmm_LLT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    side, uplo, trans, diag, alpha);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dagueprefix(trmm_LUN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    side, uplo, trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dagueprefix(trmm_LUT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    side, uplo, trans, diag, alpha);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dagueprefix(trmm_RLN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    side, uplo, trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dagueprefix(trmm_RLT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    side, uplo, trans, diag, alpha);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dagueprefix(trmm_RUN_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    side, uplo, trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dagueprefix(trmm_RUT_new)( 
		    (dague_ddesc_t*)A, (dague_ddesc_t*)B,
		    A->mb, A->nb, A->mt, A->nt, A->m, A->n,
		    B->mb, B->nb, B->mt, B->nt, B->m, B->n,
		    side, uplo, trans, diag, alpha);
            }
        }
    }

    printf("TRMM A:%ux%u (%ux%u) and B:%ux%u (%ux%u) has %u tasks to run.\n", 
           A->m, A->n, A->mt, A->nt, 
           B->m, B->n, B->mt, B->nt, 
           dague_trmm->nb_local_tasks);
    
    return dague_trmm;
}

