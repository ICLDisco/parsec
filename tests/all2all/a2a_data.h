#ifndef a2a_data_h
#define a2a_data_h

#include "parsec.h"
#include "data_dist/matrix/matrix.h"

tiled_matrix_desc_t *create_and_distribute_data(int rank, int world, int size);
void free_data(tiled_matrix_desc_t *d);

#endif
