#ifndef _rtt_wrapper_h
#define _rtt_wrapper_h

#include <dague.h>
#include <data_distribution.h>

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the dague object to schedule.
 */
dague_handle_t *rtt_new(dague_ddesc_t *A, int size, int nb);

/**
 * @param [INOUT] o the dague object to destroy
 */
void rtt_destroy(dague_handle_t *o);

#endif 
