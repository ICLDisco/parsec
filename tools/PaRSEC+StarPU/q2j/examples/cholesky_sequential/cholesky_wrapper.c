#include <starpu.h>
#include "parsec.h"
#include <plasma.h>

#include "cholesky.h"
#include "cholesky_wrapper.h"

parsec_taskpool_t *cholesky_new(parsec_ddesc_t *_A_, int _nb_, int _size_, PLASMA_enum _uplo_, int *_info_)
{
    parsec_cholesky_taskpool_t *tp = NULL;

    tp = parsec_cholesky_new(_A_, _nb_, _size_, _uplo_, _info_);

    return (parsec_taskpool_t*)tp;
}


void cholesky_destroy(parsec_taskpool_t *tp)
{
    parsec_cholesky_destroy( (parsec_cholesky_taskpool_t*) tp);
}
