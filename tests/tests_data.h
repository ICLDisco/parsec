#if !defined(_TESTS_DATA_H_)
#define _TESTS_DATA_H_

#include "parsec/runtime.h"
#include "parsec/data_dist/matrix/matrix.h"

parsec_tiled_matrix_t *create_and_distribute_data(int rank, int world, int nb, int nt);
parsec_tiled_matrix_t *create_and_distribute_empty_data(int rank, int world, int nb, int nt);
void free_data(parsec_tiled_matrix_t *d);

#endif
