#include <starpu.h>
#include "parsec.h"
#include <plasma.h>

#include "cholesky.h"
#include "cholesky_wrapper.h"

parsec_handle_t *cholesky_new(parsec_ddesc_t *_A_, int _nb_, int _size_, PLASMA_enum _uplo_, int *_info_)
{
    parsec_cholesky_handle_t *o = NULL;
    
    o = parsec_cholesky_new(_A_, _nb_, _size_, _uplo_, _info_);
    
    return (parsec_handle_t*) o;
}


void cholesky_destroy(parsec_handle_t *o)
{
    parsec_cholesky_destroy( (parsec_cholesky_handle_t*) o);
}
