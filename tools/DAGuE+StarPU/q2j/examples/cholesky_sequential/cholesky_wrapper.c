#include <starpu.h>
#include "dague.h"
#include <plasma.h>

#include "cholesky.h"
#include "cholesky_wrapper.h"

dague_handle_t *cholesky_new(dague_ddesc_t *_A_, int _nb_, int _size_, PLASMA_enum _uplo_, int *_info_)
{
    dague_cholesky_handle_t *o = NULL;
    
    o = dague_cholesky_new(_A_, _nb_, _size_, _uplo_, _info_);
    
    return (dague_handle_t*) o;
}


void cholesky_destroy(dague_handle_t *o)
{
    dague_cholesky_destroy( (dague_cholesky_handle_t*) o);
}
