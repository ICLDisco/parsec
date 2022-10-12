#ifndef DTD_TP_H_INCLUDED
#define DTD_TP_H_INCLUDED

#include "parsec.h"
#include "parsec/interfaces/dtd/insert_function.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

void new_dtd_taskpool(parsec_taskpool_t *dtd_tp, int TILE_FULL, parsec_matrix_block_cyclic_t *A, int delta);

#endif /* DTD_TP_H_INCLUDED */