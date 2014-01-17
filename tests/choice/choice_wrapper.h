#ifndef _choice_wrapper_h
#define _choice_wrapper_h

#include <dague.h>
#include <data_distribution.h>

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the dague object to schedule.
 */
dague_object_t *choice_new(dague_ddesc_t *A, int size, int *decision, int nb, int world);

/**
 * @param [INOUT] o the dague object to destroy
 */
void choice_destroy(dague_object_t *o);

#endif 
