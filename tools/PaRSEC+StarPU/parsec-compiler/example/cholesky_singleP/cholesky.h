#line 2 "cholesky.jdf"
#include <plasma.h>

#include "magma.h"
#include <core_blas.h>
#include "cublas.h"
#include "time.h"
#include "cuda.h"
#include <starpu.h>
#include "parsec.h"
#include "cholesky_data.h"

#include "precision.h"

#include "magma_s.h"



#line 2 "cholesky.c"
#ifndef _cholesky_h_
#define _cholesky_h_
#include "parsec.h"
#include "parsec/debug.h"
#include <assert.h>

#define MAX_GLOBAL_COUNT 50
#define PARSEC_cholesky_DEFAULT_ARENA    0

typedef struct parsec_cholesky_object {
  parsec_handle_t super;
  /* The list of globals */
  parsec_ddesc_t* A /* data A */;
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
  parsec_arena_t* arenas[1];
} parsec_cholesky_handle_t;

extern parsec_cholesky_handle_t *parsec_cholesky_new(parsec_ddesc_t* A /* data A */, int NB, int SIZE, PLASMA_enum uplo, int* INFO);
extern void parsec_cholesky_destroy( parsec_cholesky_handle_t *o );

struct callback_args {
    parsec_execution_unit_t     *exec_unit;
    parsec_execution_context_t  *exec_context;
};
struct func_args {
    int var[MAX_PARAM_COUNT];
    //struct callback_args *callback;
    void *obj;
    void *glob[MAX_GLOBAL_COUNT];
};
void generic_scheduling_func(parsec_execution_unit_t *context, parsec_list_item_t *elt);
#ifdef __cplusplus
extern "C" {
#endif
void hook_of_cholesky_GEMM_callback_function(parsec_execution_unit_t*, parsec_execution_context_t*);
void hook_of_cholesky_HERK_callback_function(parsec_execution_unit_t*, parsec_execution_context_t*);
void hook_of_cholesky_TRSM_callback_function(parsec_execution_unit_t*, parsec_execution_context_t*);
void hook_of_cholesky_POTRF_callback_function(parsec_execution_unit_t*, parsec_execution_context_t*);
#ifdef __cplusplus
}
#endif
#endif /* _cholesky_h_ */ 
