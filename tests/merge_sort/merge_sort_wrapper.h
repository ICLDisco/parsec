#if !defined(_MISSING_MERGE_SORT_WRAPPER_H_)
#define _MISSING_MERGE_SORT_WRAPPER_H_

#include "parsec.h"
#include "parsec/data_distribution.h"
#include "data_dist/matrix/matrix.h"

parsec_taskpool_t *merge_sort_new(tiled_matrix_desc_t *A, int size, int nt);

#endif
