#ifndef _rtt_wrapper_h
#define _rtt_wrapper_h

#include "dague.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the dague object to schedule.
 */
dague_object_t *rtt_new(dague_ddesc_t *A, int size, int nb);

/**
 * @param [INOUT] o the dague object to destroy
 */
void rtt_destroy(dague_object_t *o);

#endif 
