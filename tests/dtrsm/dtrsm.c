#include <plasma.h>

#include "dague.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"
#include "dtrsm.h"

#include "trsm_LLN.h"
#include "trsm_LLT.h"
#include "trsm_LUN.h"
#include "trsm_LUT.h"
#include "trsm_RLN.h"
#include "trsm_RLT.h"
#include "trsm_RUN.h"
#include "trsm_RUT.h"

dague_object_t * 
DAGUE_dtrsm_getObject(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                      double alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B)
{

    dague_object_t *dague_trsm = NULL;

    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_trsm_LLN_new( (dague_ddesc_t*)A, (dague_ddesc_t*)B, 
                                                                  B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n, 
                                                                  A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
                                                                  trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_trsm_LLT_new( (dague_ddesc_t*)A, (dague_ddesc_t*)B,
                                                                  B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
                                                                  A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
                                                                  trans, diag, alpha);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_trsm_LUN_new( (dague_ddesc_t*)A, (dague_ddesc_t*)B,
                                                                  B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
                                                                  A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
                                                                  trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_trsm_LUT_new( (dague_ddesc_t*)A, (dague_ddesc_t*)B,
                                                                  B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
                                                                  A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
                                                                  trans, diag, alpha);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_trsm_RLN_new( (dague_ddesc_t*)A, (dague_ddesc_t*)B,
                                                                  B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
                                                                  A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
                                                                  trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_trsm_RLT_new( (dague_ddesc_t*)A, (dague_ddesc_t*)B,
                                                                  B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
                                                                  A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
                                                                  trans, diag, alpha);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_trsm_RUN_new( (dague_ddesc_t*)A, (dague_ddesc_t*)B,
                                                                  B->super.mb, B->super.nb, B->super.mt, B->super.nt, B->super.m, B->super.n,
                                                                  A->super.mb, A->super.nb, A->super.mt, A->super.nt, A->super.m, A->super.n,
                                                                  trans, diag, alpha);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_trsm_RUT_new( (dague_ddesc_t*)A, (dague_ddesc_t*)B,
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

