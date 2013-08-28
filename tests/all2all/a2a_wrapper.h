#ifndef _a2a_wrapper_h
#define _a2a_wrapper_h

#include <dague.h>
#include <data_dist/matrix/matrix.h>

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the dague object to schedule.
 */
dague_object_t *a2a_new(tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, int size, int repeat);

/**
 * @param [INOUT] o the dague object to destroy
 */
void a2a_destroy(dague_object_t *o);

#endif 
