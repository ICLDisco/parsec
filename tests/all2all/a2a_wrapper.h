#ifndef _a2a_wrapper_h
#define _a2a_wrapper_h

#include <parsec.h>
#include <data_dist/matrix/matrix.h>

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the parsec handle to schedule.
 */
parsec_taskpool_t *a2a_new(tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, int size, int repeat);

/**
 * @param [INOUT] o the parsec handle to destroy
 */
void a2a_destroy(parsec_taskpool_t *o);

#endif 
