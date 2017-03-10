#if !defined(_REDUCE_DATA_H_)
#define _REDUCE_DATA_H_

#include "parsec.h"
#include "data_dist/matrix/matrix.h"

tiled_matrix_desc_t *create_and_distribute_data(int rank, int world, int nb, int nt);
tiled_matrix_desc_t *create_and_distribute_empty_data(int rank, int world, int nb, int nt);
void free_data(tiled_matrix_desc_t *d);

#endif
