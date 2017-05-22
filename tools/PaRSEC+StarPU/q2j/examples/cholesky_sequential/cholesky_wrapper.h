#ifndef CHOLESKY_WRAPPER
#define CHOLESKY_WRAPPER

#include "parsec.h"
#include <starpu.h>
#include <plasma.h>

parsec_taskpool_t *cholesky_new(parsec_ddesc_t *A, int nb, int size, PLASMA_enum uplo, int *info);

void cholesky_destroy(parsec_taskpool_t *o);

#endif
