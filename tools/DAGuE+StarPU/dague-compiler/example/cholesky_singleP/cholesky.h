#line 2 "cholesky.jdf"
#include <plasma.h>

#include "magma.h"
#include <core_blas.h>
#include "cublas.h"
#include "time.h"
#include "cuda.h"
#include <starpu.h>
#include "dague.h"
#include "cholesky_data.h"

#include "precision.h"

#include "magma_s.h"



#line 2 "cholesky.c"
#ifndef _cholesky_h_
#define _cholesky_h_
#include <dague.h>
#include <debug.h>
#include <assert.h>

#define MAX_GLOBAL_COUNT 50
#define DAGUE_cholesky_DEFAULT_ARENA    0

typedef struct dague_cholesky_object {
  dague_handle_t super;
  /* The list of globals */
  dague_ddesc_t* A /* data A */;
  int NB;
  int SIZE;
  PLASMA_enum uplo;
  int* INFO;
  /* The list of global handles */
  starpu_data_handle_t A_handle;
  starpu_data_handle_t NB_handle;
  starpu_data_handle_t SIZE_handle;
  starpu_data_handle_t uplo_handle;
  starpu_data_handle_t INFO_handle;
  /* The array of datatypes DEFAULT */
  dague_arena_t* arenas[1];
} dague_cholesky_handle_t;

extern dague_cholesky_handle_t *dague_cholesky_new(dague_ddesc_t* A /* data A */, int NB, int SIZE, PLASMA_enum uplo, int* INFO);
extern void dague_cholesky_destroy( dague_cholesky_handle_t *o );

struct callback_args {
    dague_execution_unit_t     *exec_unit;
    dague_execution_context_t  *exec_context;
};
struct func_args {
    int var[MAX_PARAM_COUNT];
    //struct callback_args *callback;
    void *obj;
    void *glob[MAX_GLOBAL_COUNT];
};
void generic_scheduling_func(dague_execution_unit_t *context, dague_list_item_t *elt);
#ifdef __cplusplus
extern "C" {
#endif
void hook_of_cholesky_GEMM_callback_function(dague_execution_unit_t*, dague_execution_context_t*);
void hook_of_cholesky_HERK_callback_function(dague_execution_unit_t*, dague_execution_context_t*);
void hook_of_cholesky_TRSM_callback_function(dague_execution_unit_t*, dague_execution_context_t*);
void hook_of_cholesky_POTRF_callback_function(dague_execution_unit_t*, dague_execution_context_t*);
#ifdef __cplusplus
}
#endif
#endif /* _cholesky_h_ */ 
