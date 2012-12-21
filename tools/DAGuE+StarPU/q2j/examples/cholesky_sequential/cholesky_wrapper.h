#ifndef CHOLESKY_WRAPPER
#define CHOLESKY_WRAPPER

#include "dague.h"
#include <starpu.h>
#include <plasma.h>

dague_handle_t *cholesky_new(dague_ddesc_t *A, int nb, int size, PLASMA_enum uplo, int *info);

void cholesky_destroy(dague_handle_t *o);

#endif
