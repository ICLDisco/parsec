#ifndef _zgetrf_param_h_
#define _zgetrf_param_h_
#include <dague.h>
#include <debug.h>
#include <assert.h>

#define DAGUE_zgetrf_param_DEFAULT_ARENA    0
#define DAGUE_zgetrf_param_LOWER_TILE_ARENA    1
#define DAGUE_zgetrf_param_UPPER_TILE_ARENA    2
#define DAGUE_zgetrf_param_PIVOT_ARENA    3
#define DAGUE_zgetrf_param_SMALL_L_ARENA    4
#define DAGUE_zgetrf_param_ARENA_INDEX_MIN 5

typedef struct dague_zgetrf_param_object {
  dague_object_t super;
#define zgetrf_param_p_work_SIZE (sizeof(PLASMA_Complex64_t)*ib*(descL.nb))
#define zgetrf_param_p_tau_SIZE (sizeof(PLASMA_Complex64_t)   *(descL.nb))
  /* The list of globals */
  tiled_matrix_desc_t descA;
  dague_ddesc_t * A /* data A */;
  tiled_matrix_desc_t descL;
  dague_ddesc_t * L /* data L */;
  tiled_matrix_desc_t descL2;
  dague_ddesc_t * L2;
  qr_piv_t* pivfct;
  int ib;
  dague_memory_pool_t * p_work;
  dague_memory_pool_t * p_tau;
  int param_p;
  int param_a;
  int param_d;
  dague_ddesc_t * IPIV /* data IPIV */;
  int* INFO;
  dague_memory_pool_t* work_pool;
  /* The array of datatypes DEFAULT,LOWER_TILE,UPPER_TILE,PIVOT,SMALL_L and the others */
  dague_arena_t** arenas;
  int arenas_size;
} dague_zgetrf_param_object_t;

#define zgetrf_param_p_work_SIZE (sizeof(PLASMA_Complex64_t)*ib*(descL.nb))
#define zgetrf_param_p_tau_SIZE (sizeof(PLASMA_Complex64_t)   *(descL.nb))
extern dague_zgetrf_param_object_t *dague_zgetrf_param_new(tiled_matrix_desc_t descA, dague_ddesc_t * A /* data A */, tiled_matrix_desc_t descL, dague_ddesc_t * L /* data L */, tiled_matrix_desc_t descL2, dague_ddesc_t * L2, qr_piv_t* pivfct, int ib, dague_memory_pool_t * p_work, dague_memory_pool_t * p_tau, dague_ddesc_t * IPIV /* data IPIV */, int* INFO, dague_memory_pool_t* work_pool);
extern void dague_zgetrf_param_destroy( dague_zgetrf_param_object_t *o );
#endif /* _zgetrf_param_h_ */ 
#define zgetrf_param_p_work_SIZE (sizeof(PLASMA_Complex64_t)*ib*(descL.nb))
#define zgetrf_param_p_tau_SIZE (sizeof(PLASMA_Complex64_t)   *(descL.nb))
