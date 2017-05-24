#ifndef a2a_data_h
#define a2a_data_h

#include "parsec.h"
#include "data_dist/matrix/matrix.h"

parsec_tiled_matrix_dc_t *create_and_distribute_data(int rank, int world, int size);
void free_data(parsec_tiled_matrix_dc_t *d);

#endif
