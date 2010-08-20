#ifndef DTRSM_H
#define DTRSM_H

dague_object_t * DAGUE_dtrsm_getObject(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
				       double alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B);

void DAGUE_dtrsm_setExecContext(dague_object_t *trsm, dague_execution_context_t *ctxt);

#endif /*DTRSM_H */

