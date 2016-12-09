#ifndef CHOLESKY_DATA
#define CHOLESKY_DATA

#include <starpu.h>
#include "parsec.h"
//#include "parsec/data_distribution.h"

extern int nblocks;
extern int matrix_rank;
extern int BLOCKSIZE;
extern int NMAX_BLOCKS;

parsec_ddesc_t *create_and_distribute_data(int rank, int world, int cores,int mat_r, int bs);


#ifdef __cplusplus
extern "C" {
#endif
void *get_data_handle_of(void *h);
#ifdef __cplusplus
}
#endif

void free_data(parsec_ddesc_t *d);





#endif /* CHOLESKY_DATA */
