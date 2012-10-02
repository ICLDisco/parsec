#include "cholesky_data.h"
#include <plasma.h>
#include "stdarg.h"
#include <starpu.h>
//#include "data_distribution.h"
#include "precision.h"

int matrix_rank;
int BLOCKSIZE;// = 1100;
int bline; 
int bcolumn;
int nblocks;

#define NMAX_BLOCKS 128

typedef struct {
    dague_ddesc_t super;
    float *matrix[NMAX_BLOCKS][NMAX_BLOCKS];
    starpu_data_handle_t matrix_handle[NMAX_BLOCKS][NMAX_BLOCKS];
} my_datatype_t;

static uint32_t rank_of(dague_ddesc_t *desc, ...)
{
    (void) desc;
    return 0;
}

static void *data_of(dague_ddesc_t *desc, ...)
{
    int i, j;
    my_datatype_t *dat = (my_datatype_t*) desc;        
    va_list ap;
    va_start(ap, desc);
    i = va_arg(ap, int);
    j = va_arg(ap, int);
    va_end(ap);

    return (void*) &(dat->matrix_handle[i][j]);
}


void*
get_data_handle_of(void *h)
{
    return (void*) STARPU_MATRIX_GET_PTR(h);
}

   
dague_ddesc_t *create_and_distribute_data(int rank, int world, int cores, int mat_r, int bs)
{

    my_datatype_t *m = (my_datatype_t*) calloc(1, sizeof(my_datatype_t));
    dague_ddesc_t *d = &(m->super);
    int i, j;
    matrix_rank = mat_r;
    BLOCKSIZE = bs;

    bline = matrix_rank/BLOCKSIZE;
    bcolumn = matrix_rank/BLOCKSIZE;
    nblocks = bline*bcolumn;

    d->myrank = rank;
    d->cores  = cores;
    d->nodes  = world;
    d->rank_of = rank_of;
    d->data_of = data_of;

    // Matrix allocation & handle assignment

    for(i = 0; i<bline; i++)
	for(j = 0; j<bcolumn; j++)
	{
	    starpu_malloc((void **)&(m->matrix[i][j]), (size_t) BLOCKSIZE*BLOCKSIZE*sizeof(float));
//	    m->matrix[i][j] = malloc(BLOCKSIZE*BLOCKSIZE*sizeof(PLASMA_Complex64_t));
	    starpu_matrix_data_register(&m->matrix_handle[i][j], 0, (uintptr_t)m->matrix[i][j], BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, sizeof(float)); 
	    //    starpu_data_set_sequential_consistency_flag(m->matrix_handle[i][j], 0);
	}
    
    return d;
}



void free_data(dague_ddesc_t *d)
{
    int i, j;
    my_datatype_t *m = (my_datatype_t*)d;
    
    for(i = 0; i<bline; i++)
	for(j = 0; j<bcolumn; j++)
	{
//	    starpu_data_release(m->matrix_handle[i][j]);
	    starpu_data_unregister(m->matrix_handle[i][j]);
	    starpu_free(m->matrix[i][j]);
	}
    
    dague_ddesc_destroy(d);
    free(d);
}




