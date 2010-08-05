#ifndef _cholesky_h_
#define _cholesky_h_
#include <dague.h>

typedef struct dague_cholesky_object {
  dague_object_t super;
  /* The list of globals */
  int NB;
  int SIZE;
  /* The list of data */
  dague_ddesc_t *A;
} dague_cholesky_object_t;
dague_cholesky_object_t *dague_cholesky_new(dague_ddesc_t *A, int NB, int SIZE);
#endif /* _cholesky_h_ */ 
