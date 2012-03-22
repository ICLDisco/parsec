#line 2 "zgetrf_param.jdf"
/*
 *  Copyright (c) 2010
 *
 *  The University of Tennessee and The University
 *  of Tennessee Research Foundation.  All rights
 *  reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#define PRECISION_z

#include "dague.h"
#include <math.h>
#include <plasma.h>
#include <core_blas.h>

#include "data_distribution.h"
#include "data_dist/matrix/precision.h"
#include "data_dist/matrix/matrix.h"
#include "dplasma/lib/memory_pool.h"
#include "dplasma/lib/dplasmajdf.h"
#include "dplasma_qr_pivgen.h"

#define MYMIN(a, b) ((a)<(b)?(a):(b))
#define MYMAX(a, b) ((a)>(b)?(a):(b))
#define min( __a, __b ) ( (__a) < (__b) ? (__a) : (__b) )

//#define PRIO_YVES1

#if defined(PRIO_YVES1)
#define GETPRIO_PANEL( __m, __n )      descA.mt * descA.nt - ((descA.nt - (__n) - 1) * descA.mt + (__m) + 1)
#define GETPRIO_UPDTE( __m, __n, __k ) descA.mt * descA.nt - ((descA.nt - (__n) - 1) * descA.mt + (__m) + 1)
#elif defined(PRIO_YVES2)
#define GETPRIO_PANEL( __m, __n )      descA.mt * descA.nt - ((__m) * descA.nt + descA.nt - (__n))
#define GETPRIO_UPDTE( __m, __n, __k ) descA.mt * descA.nt - ((__m) * descA.nt + descA.nt - (__n))
#elif defined(PRIO_MATHIEU1)
#define GETPRIO_PANEL( __m, __n )      (descA.mt + (__n) - (__m) - 1) * descA.nt + (__n)
#define GETPRIO_UPDTE( __m, __n, __k ) (descA.mt + (__n) - (__m) - 1) * descA.nt + (__n)
#elif defined(PRIO_MATHIEU2)
#define GETPRIO_PANEL( __m, __n )      ((MYMAX(descA.mt, descA.nt) - MYMAX( (__n) - (__m), (__m) - (__n) ) -1 ) * 12 + (__n))
#define GETPRIO_UPDTE( __m, __n, __k ) ((MYMAX(descA.mt, descA.nt) - MYMAX( (__n) - (__m), (__m) - (__n) ) -1 ) * 12 + (__n))
#elif defined(PRIO_MATYVES)
#define FORMULE( __x ) ( ( -1. + sqrt( 1. + 4.* (__x) * (__x)) ) * 0.5 )
#define GETPRIO_PANEL( __m, __k )      (int)( 22. * (__k) + 6. * ( FORMULE( descA.mt ) - FORMULE( (__m) - (__k) + 1. ) ) )
#define GETPRIO_UPDTE( __m, __n, __k ) (int)( (__m) < (__n) ? GETPRIO_PANEL( (__n), (__n) ) - 22. * ( (__m) - (__k) ) - 6. * ( (__n) - (__m) ) \
                                              :               GETPRIO_PANEL( (__m), (__n) ) - 22. * ( (__n) - (__k) ) )
#else
  /*#warning running without priority*/
#define GETPRIO_PANEL( __m, __n )      0
#define GETPRIO_UPDTE( __m, __n, __k ) 0
#endif


#line 57 "zgetrf_param.c"
#include <dague.h>
#include <scheduling.h>
#include <remote_dep.h>
#include <datarepo.h>
#if defined(HAVE_PAPI)
#include <papime.h>
#endif
#include "zgetrf_param.h"

#define DAGUE_zgetrf_param_NB_FUNCTIONS 7
#define DAGUE_zgetrf_param_NB_DATA 3
#if defined(DAGUE_PROF_TRACE)
int zgetrf_param_profiling_array[2*DAGUE_zgetrf_param_NB_FUNCTIONS] = {-1};
#define TAKE_TIME(context, key, eid, refdesc, refid) do {   \
   dague_profile_ddesc_info_t info;                         \
   info.desc = (dague_ddesc_t*)refdesc;                     \
   info.id = refid;                                         \
   dague_profiling_trace(context->eu_profile,               \
                         __dague_object->super.super.profiling_array[(key)],\
                         eid, (void*)&info);                \
  } while(0);
#else
#define TAKE_TIME(context, key, id, refdesc, refid)
#endif
#include "dague_prof_grapher.h"
#include <mempool.h>
typedef struct __dague_zgetrf_param_internal_object {
 dague_zgetrf_param_object_t super;
  /* The list of data repositories */
  data_repo_t *zttmqr_repository;
  data_repo_t *zttmqr_out_repository;
  data_repo_t *zttqrt_repository;
  data_repo_t *zttqrt_out_A1_repository;
  data_repo_t *zgessm_repository;
  data_repo_t *zgetrf_param_repository;
  data_repo_t *zgetrf_param_out_repository;
} __dague_zgetrf_param_internal_object_t;

/* Globals */
#define descA (__dague_object->super.descA)
#define descL (__dague_object->super.descL)
#define descL2 (__dague_object->super.descL2)
#define L2 (__dague_object->super.L2)
#define pivfct (__dague_object->super.pivfct)
#define ib (__dague_object->super.ib)
#define p_work (__dague_object->super.p_work)
#define p_tau (__dague_object->super.p_tau)
#define param_p (__dague_object->super.param_p)
#define param_a (__dague_object->super.param_a)
#define param_d (__dague_object->super.param_d)
#define INFO (__dague_object->super.INFO)
#define work_pool (__dague_object->super.work_pool)

/* Data Access Macros */
#define L(L0,L1)  (((dague_ddesc_t*)__dague_object->super.L)->data_of((dague_ddesc_t*)__dague_object->super.L, (L0), (L1)))

#define IPIV(IPIV0,IPIV1)  (((dague_ddesc_t*)__dague_object->super.IPIV)->data_of((dague_ddesc_t*)__dague_object->super.IPIV, (IPIV0), (IPIV1)))

#define A(A0,A1)  (((dague_ddesc_t*)__dague_object->super.A)->data_of((dague_ddesc_t*)__dague_object->super.A, (A0), (A1)))


/* Functions Predicates */
#define zttmqr_pred(k, m, n, p, nextp, prevp, prevm, type, type1, ip, im, im1) (((dague_ddesc_t*)(__dague_object->super.A))->myrank == ((dague_ddesc_t*)(__dague_object->super.A))->rank_of((dague_ddesc_t*)__dague_object->super.A, m, n))
#define zttmqr_out_pred(k, n, prevp) (((dague_ddesc_t*)(__dague_object->super.A))->myrank == ((dague_ddesc_t*)(__dague_object->super.A))->rank_of((dague_ddesc_t*)__dague_object->super.A, k, n))
#define zttqrt_pred(k, m, p, nextp, prevp, prevm, type, ip, im) (((dague_ddesc_t*)(__dague_object->super.A))->myrank == ((dague_ddesc_t*)(__dague_object->super.A))->rank_of((dague_ddesc_t*)__dague_object->super.A, m, k))
#define zttqrt_out_A1_pred(k, prevp) (((dague_ddesc_t*)(__dague_object->super.A))->myrank == ((dague_ddesc_t*)(__dague_object->super.A))->rank_of((dague_ddesc_t*)__dague_object->super.A, k, k))
#define zgessm_pred(k, i, n, m, nextm) (((dague_ddesc_t*)(__dague_object->super.A))->myrank == ((dague_ddesc_t*)(__dague_object->super.A))->rank_of((dague_ddesc_t*)__dague_object->super.A, m, n))
#define zgetrf_param_pred(k, i, m, nextm) (((dague_ddesc_t*)(__dague_object->super.A))->myrank == ((dague_ddesc_t*)(__dague_object->super.A))->rank_of((dague_ddesc_t*)__dague_object->super.A, m, k))
#define zgetrf_param_out_pred(k, i, m) (((dague_ddesc_t*)(__dague_object->super.A))->myrank == ((dague_ddesc_t*)(__dague_object->super.A))->rank_of((dague_ddesc_t*)__dague_object->super.A, m, k))

/* Data Repositories */
#define zttmqr_repo (__dague_object->zttmqr_repository)
#define zttmqr_out_repo (__dague_object->zttmqr_out_repository)
#define zttqrt_repo (__dague_object->zttqrt_repository)
#define zttqrt_out_A1_repo (__dague_object->zttqrt_out_A1_repository)
#define zgessm_repo (__dague_object->zgessm_repository)
#define zgetrf_param_repo (__dague_object->zgetrf_param_repository)
#define zgetrf_param_out_repo (__dague_object->zgetrf_param_out_repository)
/* Dependency Tracking Allocation Macro */
#define ALLOCATE_DEP_TRACKING(DEPS, vMIN, vMAX, vNAME, vSYMBOL, PREVDEP, FLAG)               \
do {                                                                                         \
  int _vmin = (vMIN);                                                                        \
  int _vmax = (vMAX);                                                                        \
  (DEPS) = (dague_dependencies_t*)calloc(1, sizeof(dague_dependencies_t) +                   \
                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \
  DEBUG3(("Allocate %d spaces for loop %s (min %d max %d) 0x%p last_dep 0x%p\n",    \
           (_vmax - _vmin + 1), (vNAME), _vmin, _vmax, (void*)(DEPS), (void*)(PREVDEP)));    \
  (DEPS)->flags = DAGUE_DEPENDENCIES_FLAG_ALLOCATED | (FLAG);                                \
  DAGUE_STAT_INCREASE(mem_bitarray,  sizeof(dague_dependencies_t) + STAT_MALLOC_OVERHEAD +   \
                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \
  (DEPS)->symbol = (vSYMBOL);                                                                \
  (DEPS)->min = _vmin;                                                                       \
  (DEPS)->max = _vmax;                                                                       \
  (DEPS)->prev = (PREVDEP); /* chain them backward */                                        \
} while (0)                                                                                  

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };                     

static inline int dague_imax(int a, int b) { return (a >= b) ? a : b; };                     

static inline int zgetrf_param_inline_c_expr1_line_356(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 356 "zgetrf_param.jdf"
 return type == 0 ? GETPRIO_UPDTE(p, n, k) : GETPRIO_UPDTE(m, n, k); 
#line 180 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr2_line_330(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 330 "zgetrf_param.jdf"
 return type == 0 ? 12 : 6; 
#line 205 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr3_line_328(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 328 "zgetrf_param.jdf"
 return dplasma_qr_geti(    pivfct, k+1, m ); 
#line 230 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr4_line_327(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 327 "zgetrf_param.jdf"
 return dplasma_qr_geti(    pivfct, k,   m ); 
#line 255 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr5_line_326(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 326 "zgetrf_param.jdf"
 return dplasma_qr_geti(    pivfct, k,   p ); 
#line 280 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr6_line_325(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 325 "zgetrf_param.jdf"
 return dplasma_qr_gettype( pivfct, k+1, m ); 
#line 305 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr7_line_324(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 324 "zgetrf_param.jdf"
 return dplasma_qr_gettype( pivfct, k,   m ); 
#line 330 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr8_line_323(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 323 "zgetrf_param.jdf"
 return dplasma_qr_prevpiv(pivfct, m, k, m); 
#line 355 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr9_line_322(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 322 "zgetrf_param.jdf"
 return dplasma_qr_prevpiv(pivfct, p, k, m); 
#line 380 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr10_line_321(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 321 "zgetrf_param.jdf"
 return dplasma_qr_nextpiv(pivfct, p, k, m); 
#line 405 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr11_line_320(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

#line 320 "zgetrf_param.jdf"
 return dplasma_qr_currpiv(pivfct, m, k); 
#line 430 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr12_line_305(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttmqr_out */
  int k = assignments[0].value;
  int n = assignments[1].value;
  int prevp = assignments[2].value;

  (void)k;  (void)n;  (void)prevp;

#line 305 "zgetrf_param.jdf"
 return dplasma_qr_prevpiv(pivfct, k, k, k); 
#line 446 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr13_line_243(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttqrt */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;
  int prevp = assignments[4].value;
  int prevm = assignments[5].value;
  int type = assignments[6].value;
  int ip = assignments[7].value;
  int im = assignments[8].value;

  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

#line 243 "zgetrf_param.jdf"
 return type == 0 ? GETPRIO_PANEL(p, k) : GETPRIO_PANEL(m, k); 
#line 468 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr14_line_218(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttqrt */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;
  int prevp = assignments[4].value;
  int prevm = assignments[5].value;
  int type = assignments[6].value;
  int ip = assignments[7].value;
  int im = assignments[8].value;

  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

#line 218 "zgetrf_param.jdf"
 return type == 0 ? 6 : 2; 
#line 490 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr15_line_216(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttqrt */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;
  int prevp = assignments[4].value;
  int prevm = assignments[5].value;
  int type = assignments[6].value;
  int ip = assignments[7].value;
  int im = assignments[8].value;

  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

#line 216 "zgetrf_param.jdf"
 return dplasma_qr_geti(    pivfct, k, m ); 
#line 512 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr16_line_215(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttqrt */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;
  int prevp = assignments[4].value;
  int prevm = assignments[5].value;
  int type = assignments[6].value;
  int ip = assignments[7].value;
  int im = assignments[8].value;

  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

#line 215 "zgetrf_param.jdf"
 return dplasma_qr_geti(    pivfct, k, p ); 
#line 534 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr17_line_214(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttqrt */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;
  int prevp = assignments[4].value;
  int prevm = assignments[5].value;
  int type = assignments[6].value;
  int ip = assignments[7].value;
  int im = assignments[8].value;

  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

#line 214 "zgetrf_param.jdf"
 return dplasma_qr_gettype( pivfct, k, m ); 
#line 556 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr18_line_213(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttqrt */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;
  int prevp = assignments[4].value;
  int prevm = assignments[5].value;
  int type = assignments[6].value;
  int ip = assignments[7].value;
  int im = assignments[8].value;

  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

#line 213 "zgetrf_param.jdf"
 return dplasma_qr_prevpiv(pivfct, m, k, m); 
#line 578 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr19_line_212(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttqrt */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;
  int prevp = assignments[4].value;
  int prevm = assignments[5].value;
  int type = assignments[6].value;
  int ip = assignments[7].value;
  int im = assignments[8].value;

  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

#line 212 "zgetrf_param.jdf"
 return dplasma_qr_prevpiv(pivfct, p, k, m); 
#line 600 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr20_line_211(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttqrt */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;
  int prevp = assignments[4].value;
  int prevm = assignments[5].value;
  int type = assignments[6].value;
  int ip = assignments[7].value;
  int im = assignments[8].value;

  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

#line 211 "zgetrf_param.jdf"
 return dplasma_qr_nextpiv(pivfct, p, k, m); 
#line 622 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr21_line_210(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttqrt */
  int k = assignments[0].value;
  int m = assignments[1].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;
  int prevp = assignments[4].value;
  int prevm = assignments[5].value;
  int type = assignments[6].value;
  int ip = assignments[7].value;
  int im = assignments[8].value;

  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

#line 210 "zgetrf_param.jdf"
 return dplasma_qr_currpiv(pivfct, m, k); 
#line 644 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr22_line_196(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zttqrt_out_A1 */
  int k = assignments[0].value;
  int prevp = assignments[1].value;

  (void)k;  (void)prevp;

#line 196 "zgetrf_param.jdf"
 return dplasma_qr_prevpiv(pivfct, k, k, k); 
#line 659 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr23_line_172(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zgessm */
  int k = assignments[0].value;
  int i = assignments[1].value;
  int n = assignments[2].value;
  int m = assignments[3].value;
  int nextm = assignments[4].value;

  (void)k;  (void)i;  (void)n;  (void)m;  (void)nextm;

#line 172 "zgetrf_param.jdf"
 return GETPRIO_UPDTE(m, n, k); 
#line 677 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr24_line_154(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zgessm */
  int k = assignments[0].value;
  int i = assignments[1].value;
  int n = assignments[2].value;
  int m = assignments[3].value;
  int nextm = assignments[4].value;

  (void)k;  (void)i;  (void)n;  (void)m;  (void)nextm;

#line 154 "zgetrf_param.jdf"
 return dplasma_qr_nextpiv( pivfct, m, k, descA.mt); 
#line 695 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr25_line_153(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zgessm */
  int k = assignments[0].value;
  int i = assignments[1].value;
  int n = assignments[2].value;
  int m = assignments[3].value;
  int nextm = assignments[4].value;

  (void)k;  (void)i;  (void)n;  (void)m;  (void)nextm;

#line 153 "zgetrf_param.jdf"
 return dplasma_qr_getm( pivfct, k, i); 
#line 713 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr26_line_151(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zgessm */
  int k = assignments[0].value;
  int i = assignments[1].value;
  int n = assignments[2].value;
  int m = assignments[3].value;
  int nextm = assignments[4].value;

  (void)k;  (void)i;  (void)n;  (void)m;  (void)nextm;

#line 151 "zgetrf_param.jdf"
 return dplasma_qr_getnbgeqrf( pivfct, k, descA.mt ) - 1; 
#line 731 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr27_line_116(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zgetrf_param */
  int k = assignments[0].value;
  int i = assignments[1].value;
  int m = assignments[2].value;
  int nextm = assignments[3].value;

  (void)k;  (void)i;  (void)m;  (void)nextm;

#line 116 "zgetrf_param.jdf"
 return GETPRIO_PANEL(m, k); 
#line 748 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr28_line_96(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zgetrf_param */
  int k = assignments[0].value;
  int i = assignments[1].value;
  int m = assignments[2].value;
  int nextm = assignments[3].value;

  (void)k;  (void)i;  (void)m;  (void)nextm;

#line 96 "zgetrf_param.jdf"
 return dplasma_qr_nextpiv( pivfct, m, k, descA.mt); 
#line 765 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr29_line_95(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zgetrf_param */
  int k = assignments[0].value;
  int i = assignments[1].value;
  int m = assignments[2].value;
  int nextm = assignments[3].value;

  (void)k;  (void)i;  (void)m;  (void)nextm;

#line 95 "zgetrf_param.jdf"
 return dplasma_qr_getm( pivfct, k, i); 
#line 782 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr30_line_94(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zgetrf_param */
  int k = assignments[0].value;
  int i = assignments[1].value;
  int m = assignments[2].value;
  int nextm = assignments[3].value;

  (void)k;  (void)i;  (void)m;  (void)nextm;

#line 94 "zgetrf_param.jdf"
 return dplasma_qr_getnbgeqrf( pivfct, k, descA.mt ) - 1; 
#line 799 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr31_line_78(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zgetrf_param_out */
  int k = assignments[0].value;
  int i = assignments[1].value;
  int m = assignments[2].value;

  (void)k;  (void)i;  (void)m;

#line 78 "zgetrf_param.jdf"
 return dplasma_qr_getm( pivfct, k, i); 
#line 815 "zgetrf_param.c"
}

static inline int zgetrf_param_inline_c_expr32_line_77(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  (void)__dague_object;
  /* This inline C function was declared in the context of the task zgetrf_param_out */
  int k = assignments[0].value;
  int i = assignments[1].value;
  int m = assignments[2].value;

  (void)k;  (void)i;  (void)m;

#line 77 "zgetrf_param.jdf"
 return dplasma_qr_getnbgeqrf( pivfct, k, descA.mt ) - 1; 
#line 831 "zgetrf_param.c"
}

static inline uint64_t zttmqr_hash(const __dague_zgetrf_param_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  int k_range = ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)) - k_min + 1;
  int m = assignments[1].value;
  int m_min = (k + 1);
  int m_range = (descA.mt - 1) - m_min + 1;
  int n = assignments[2].value;
  int n_min = (k + 1);
  __h += (k - k_min);
  __h += (m - m_min) * k_range;
  __h += (n - n_min) * k_range * m_range;
  return __h;
}

static inline uint64_t zttmqr_out_hash(const __dague_zgetrf_param_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  int k_range = ((descA.mt < descA.nt) ? (descA.mt - 2) : (descA.nt - 2)) - k_min + 1;
  int n = assignments[1].value;
  int n_min = (k + 1);
  __h += (k - k_min);
  __h += (n - n_min) * k_range;
  return __h;
}

static inline uint64_t zttqrt_hash(const __dague_zgetrf_param_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  int k_range = ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)) - k_min + 1;
  int m = assignments[1].value;
  int m_min = (k + 1);
  __h += (k - k_min);
  __h += (m - m_min) * k_range;
  return __h;
}

static inline uint64_t zttqrt_out_A1_hash(const __dague_zgetrf_param_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  __h += (k - k_min);
  return __h;
}

static inline uint64_t zgessm_hash(const __dague_zgetrf_param_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  int k_range = ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)) - k_min + 1;
  int i = assignments[1].value;
  int i_min = 0;
  int i_range = zgetrf_param_inline_c_expr26_line_151((const dague_object_t*)__dague_object, assignments) - i_min + 1;
  int n = assignments[2].value;
  int n_min = (k + 1);
  __h += (k - k_min);
  __h += (i - i_min) * k_range;
  __h += (n - n_min) * k_range * i_range;
  return __h;
}

static inline uint64_t zgetrf_param_hash(const __dague_zgetrf_param_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  int k_range = ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)) - k_min + 1;
  int i = assignments[1].value;
  int i_min = 0;
  __h += (k - k_min);
  __h += (i - i_min) * k_range;
  return __h;
}

static inline uint64_t zgetrf_param_out_hash(const __dague_zgetrf_param_internal_object_t *__dague_object, const assignment_t *assignments)
{
  uint64_t __h = 0;
  (void)__dague_object;
  int k = assignments[0].value;
  int k_min = 0;
  int k_range = ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)) - k_min + 1;
  int i = assignments[1].value;
  int i_min = 0;
  __h += (k - k_min);
  __h += (i - i_min) * k_range;
  return __h;
}

/** Predeclarations of the dague_function_t objects */
static const dague_function_t zgetrf_param_zttmqr;
static inline int priority_of_zgetrf_param_zttmqr_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments);
static const dague_function_t zgetrf_param_zttmqr_out;
static const dague_function_t zgetrf_param_zttqrt;
static inline int priority_of_zgetrf_param_zttqrt_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments);
static const dague_function_t zgetrf_param_zttqrt_out_A1;
static const dague_function_t zgetrf_param_zgessm;
static inline int priority_of_zgetrf_param_zgessm_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments);
static const dague_function_t zgetrf_param_zgetrf_param;
static inline int priority_of_zgetrf_param_zgetrf_param_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments);
static const dague_function_t zgetrf_param_zgetrf_param_out;
/** Declarations of the pseudo-dague_function_t objects for data */
static const dague_function_t zgetrf_param_L = {
  .name = "L",
  .flags = 0x0,
  .dependencies_goal = 0x0,
  .nb_parameters = 0,
  .nb_definitions = 0,
  .params = { NULL, },
  .locals = { NULL, },
  .pred = NULL,
  .in = { NULL, },
  .out = { NULL, },
  .priority = NULL,
  .deps = -1,
  .hook = NULL,
  .release_deps = NULL,
  .body = NULL,
#if defined(DAGUE_SCHED_CACHE_AWARE)
  .cache_rank_function = NULL,
#endif /* defined(DAGUE_SCHED_CACHE_AWARE) */
};
static const dague_function_t zgetrf_param_IPIV = {
  .name = "IPIV",
  .flags = 0x0,
  .dependencies_goal = 0x0,
  .nb_parameters = 0,
  .nb_definitions = 0,
  .params = { NULL, },
  .locals = { NULL, },
  .pred = NULL,
  .in = { NULL, },
  .out = { NULL, },
  .priority = NULL,
  .deps = -1,
  .hook = NULL,
  .release_deps = NULL,
  .body = NULL,
#if defined(DAGUE_SCHED_CACHE_AWARE)
  .cache_rank_function = NULL,
#endif /* defined(DAGUE_SCHED_CACHE_AWARE) */
};
static const dague_function_t zgetrf_param_A = {
  .name = "A",
  .flags = 0x0,
  .dependencies_goal = 0x0,
  .nb_parameters = 0,
  .nb_definitions = 0,
  .params = { NULL, },
  .locals = { NULL, },
  .pred = NULL,
  .in = { NULL, },
  .out = { NULL, },
  .priority = NULL,
  .deps = -1,
  .hook = NULL,
  .release_deps = NULL,
  .body = NULL,
#if defined(DAGUE_SCHED_CACHE_AWARE)
  .cache_rank_function = NULL,
#endif /* defined(DAGUE_SCHED_CACHE_AWARE) */
};

/** Predeclarations of the parameters */
static const dague_flow_t flow_of_zgetrf_param_zttmqr_for_V;
static const dague_flow_t flow_of_zgetrf_param_zttmqr_for_C;
static const dague_flow_t flow_of_zgetrf_param_zttmqr_for_H;
static const dague_flow_t flow_of_zgetrf_param_zttmqr_for_L;
static const dague_flow_t flow_of_zgetrf_param_zttmqr_for_P;
static const dague_flow_t flow_of_zgetrf_param_zttmqr_out_for_A;
static const dague_flow_t flow_of_zgetrf_param_zttqrt_for_A;
static const dague_flow_t flow_of_zgetrf_param_zttqrt_for_C;
static const dague_flow_t flow_of_zgetrf_param_zttqrt_for_L;
static const dague_flow_t flow_of_zgetrf_param_zttqrt_for_P;
static const dague_flow_t flow_of_zgetrf_param_zttqrt_out_A1_for_A;
static const dague_flow_t flow_of_zgetrf_param_zgessm_for_A;
static const dague_flow_t flow_of_zgetrf_param_zgessm_for_P;
static const dague_flow_t flow_of_zgetrf_param_zgessm_for_C;
static const dague_flow_t flow_of_zgetrf_param_zgetrf_param_for_A;
static const dague_flow_t flow_of_zgetrf_param_zgetrf_param_for_P;
static const dague_flow_t flow_of_zgetrf_param_zgetrf_param_out_for_A;
/**********************************************************************************
 *                                    zttmqr                                    *
 **********************************************************************************/

static inline int minexpr_of_symb_zgetrf_param_zttmqr_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgetrf_param_zttmqr_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zttmqr_k_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zttmqr_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
}
static const expr_t maxexpr_of_symb_zgetrf_param_zttmqr_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zttmqr_k_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_k = {.min = &minexpr_of_symb_zgetrf_param_zttmqr_k, .max = &maxexpr_of_symb_zgetrf_param_zttmqr_k,  .flags = 0x0};

static inline int minexpr_of_symb_zgetrf_param_zttmqr_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t minexpr_of_symb_zgetrf_param_zttmqr_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zttmqr_m_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zttmqr_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return (descA.mt - 1);
}
static const expr_t maxexpr_of_symb_zgetrf_param_zttmqr_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zttmqr_m_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_m = {.min = &minexpr_of_symb_zgetrf_param_zttmqr_m, .max = &maxexpr_of_symb_zgetrf_param_zttmqr_m,  .flags = 0x0};

static inline int minexpr_of_symb_zgetrf_param_zttmqr_n_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t minexpr_of_symb_zgetrf_param_zttmqr_n = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zttmqr_n_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zttmqr_n_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return (descA.nt - 1);
}
static const expr_t maxexpr_of_symb_zgetrf_param_zttmqr_n = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zttmqr_n_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_n = {.min = &minexpr_of_symb_zgetrf_param_zttmqr_n, .max = &maxexpr_of_symb_zgetrf_param_zttmqr_n,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttmqr_p_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttmqr_p = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttmqr_p_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_p = {.min = &expr_of_symb_zgetrf_param_zttmqr_p, .max = &expr_of_symb_zgetrf_param_zttmqr_p,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttmqr_nextp_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttmqr_nextp = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttmqr_nextp_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_nextp = {.min = &expr_of_symb_zgetrf_param_zttmqr_nextp, .max = &expr_of_symb_zgetrf_param_zttmqr_nextp,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttmqr_prevp_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttmqr_prevp = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttmqr_prevp_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_prevp = {.min = &expr_of_symb_zgetrf_param_zttmqr_prevp, .max = &expr_of_symb_zgetrf_param_zttmqr_prevp,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttmqr_prevm_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttmqr_prevm = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttmqr_prevm_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_prevm = {.min = &expr_of_symb_zgetrf_param_zttmqr_prevm, .max = &expr_of_symb_zgetrf_param_zttmqr_prevm,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttmqr_type_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttmqr_type = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttmqr_type_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_type = {.min = &expr_of_symb_zgetrf_param_zttmqr_type, .max = &expr_of_symb_zgetrf_param_zttmqr_type,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttmqr_type1_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttmqr_type1 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttmqr_type1_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_type1 = {.min = &expr_of_symb_zgetrf_param_zttmqr_type1, .max = &expr_of_symb_zgetrf_param_zttmqr_type1,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttmqr_ip_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttmqr_ip = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttmqr_ip_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_ip = {.min = &expr_of_symb_zgetrf_param_zttmqr_ip, .max = &expr_of_symb_zgetrf_param_zttmqr_ip,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttmqr_im_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttmqr_im = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttmqr_im_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_im = {.min = &expr_of_symb_zgetrf_param_zttmqr_im, .max = &expr_of_symb_zgetrf_param_zttmqr_im,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttmqr_im1_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttmqr_im1 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttmqr_im1_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_im1 = {.min = &expr_of_symb_zgetrf_param_zttmqr_im1, .max = &expr_of_symb_zgetrf_param_zttmqr_im1,  .flags = 0x0};

static inline int pred_of_zgetrf_param_zttmqr_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int m = assignments[1].value;
  int n = assignments[2].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;
  int prevp = assignments[5].value;
  int prevm = assignments[6].value;
  int type = assignments[7].value;
  int type1 = assignments[8].value;
  int ip = assignments[9].value;
  int im = assignments[10].value;
  int im1 = assignments[11].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)m;
  (void)n;
  (void)p;
  (void)nextp;
  (void)prevp;
  (void)prevm;
  (void)type;
  (void)type1;
  (void)ip;
  (void)im;
  (void)im1;
  /* Compute Predicate */
  return zttmqr_pred(k, m, n, p, nextp, prevp, prevm, type, type1, ip, im, im1);
}
static const expr_t pred_of_zgetrf_param_zttmqr_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zgetrf_param_zttmqr_as_expr_fct
};
static inline int priority_of_zgetrf_param_zttmqr_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr1_line_356((const dague_object_t*)__dague_object, assignments);
}
static const expr_t priority_of_zgetrf_param_zttmqr_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = priority_of_zgetrf_param_zttmqr_as_expr_fct
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevp = assignments[5].value;

  (void)__dague_object; (void)assignments;
  return (prevp == descA.mt);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int ip = assignments[9].value;

  (void)__dague_object; (void)assignments;
  return ip;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340,
  .dague = &zgetrf_param_zgessm,
  .flow = &flow_of_zgetrf_param_zgessm_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340,
    &expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevp = assignments[5].value;

  (void)__dague_object; (void)assignments;
  return !(prevp == descA.mt);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevp = assignments[5].value;

  (void)__dague_object; (void)assignments;
  return prevp;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_V,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340,
    &expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int nextp = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return (nextp != descA.mt);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int nextp = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return nextp;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_V,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341,
    &expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return ((nextp == descA.mt) && (p == k));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int p = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return p;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342,
  .dague = &zgetrf_param_zttmqr_out,
  .flow = &flow_of_zgetrf_param_zttmqr_out_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int p = assignments[3].value;
  int nextp = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return ((nextp == descA.mt) && (p != k));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int p = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return p;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343,
    &expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttmqr_for_V = {
  .name = "V",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 0,
  .dep_in  = { &flow_of_zgetrf_param_zttmqr_for_V_dep1_iftrue_atline_340, &flow_of_zgetrf_param_zttmqr_for_V_dep1_iffalse_atline_340 },
  .dep_out = { &flow_of_zgetrf_param_zttmqr_for_V_dep2_atline_341, &flow_of_zgetrf_param_zttmqr_for_V_dep3_atline_342, &flow_of_zgetrf_param_zttmqr_for_V_dep4_atline_343 }
};

static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int type = assignments[7].value;

  (void)__dague_object; (void)assignments;
  return ((type == 0) && (k == 0));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344,
  .dague = &zgetrf_param_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int type = assignments[7].value;

  (void)__dague_object; (void)assignments;
  return ((type == 0) && (k != 0));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k - 1);
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345,
    &expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevm = assignments[6].value;
  int type = assignments[7].value;

  (void)__dague_object; (void)assignments;
  return ((type != 0) && (prevm == descA.mt));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int im = assignments[10].value;

  (void)__dague_object; (void)assignments;
  return im;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346,
  .dague = &zgetrf_param_zgessm,
  .flow = &flow_of_zgetrf_param_zgessm_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346,
    &expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevm = assignments[6].value;
  int type = assignments[7].value;

  (void)__dague_object; (void)assignments;
  return ((type != 0) && (prevm != descA.mt));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevm = assignments[6].value;

  (void)__dague_object; (void)assignments;
  return prevm;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_V,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347,
    &expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int n = assignments[2].value;
  int type1 = assignments[8].value;

  (void)__dague_object; (void)assignments;
  return ((type1 != 0) && (n == (k + 1)));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int im1 = assignments[11].value;

  (void)__dague_object; (void)assignments;
  return im1;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348,
  .dague = &zgetrf_param_zgetrf_param,
  .flow = &flow_of_zgetrf_param_zgetrf_param_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int n = assignments[2].value;
  int type1 = assignments[8].value;

  (void)__dague_object; (void)assignments;
  return ((type1 != 0) && (n > (k + 1)));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int im1 = assignments[11].value;

  (void)__dague_object; (void)assignments;
  return im1;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349,
  .dague = &zgetrf_param_zgessm,
  .flow = &flow_of_zgetrf_param_zgessm_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349,
    &expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int n = assignments[2].value;
  int type1 = assignments[8].value;

  (void)__dague_object; (void)assignments;
  return ((type1 == 0) && (n == (k + 1)));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int n = assignments[2].value;
  int type1 = assignments[8].value;

  (void)__dague_object; (void)assignments;
  return ((type1 == 0) && (n > (k + 1)));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352,
    &expr_of_p3_for_flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttmqr_for_C = {
  .name = "C",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 1,
  .dep_in  = { &flow_of_zgetrf_param_zttmqr_for_C_dep1_atline_344, &flow_of_zgetrf_param_zttmqr_for_C_dep2_atline_345, &flow_of_zgetrf_param_zttmqr_for_C_dep3_atline_346, &flow_of_zgetrf_param_zttmqr_for_C_dep4_atline_347 },
  .dep_out = { &flow_of_zgetrf_param_zttmqr_for_C_dep5_atline_348, &flow_of_zgetrf_param_zttmqr_for_C_dep6_atline_349, &flow_of_zgetrf_param_zttmqr_for_C_dep7_atline_350, &flow_of_zgetrf_param_zttmqr_for_C_dep8_atline_352 }
};

static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_H_dep1_atline_353_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_H_dep1_atline_353 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_H_dep1_atline_353_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_H_dep1_atline_353_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_H_dep1_atline_353 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_H_dep1_atline_353_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_H_dep1_atline_353 = {
  .cond = NULL,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_H_dep1_atline_353,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_H_dep1_atline_353
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttmqr_for_H = {
  .name = "H",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 2,
  .dep_in  = { &flow_of_zgetrf_param_zttmqr_for_H_dep1_atline_353 },
  .dep_out = { NULL }
};

static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_L_dep1_atline_353_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_L_dep1_atline_353 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_L_dep1_atline_353_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_L_dep1_atline_353_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_L_dep1_atline_353 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_L_dep1_atline_353_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_L_dep1_atline_353 = {
  .cond = NULL,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_L,
  .datatype = { .index = DAGUE_zgetrf_param_SMALL_L_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_L_dep1_atline_353,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_L_dep1_atline_353
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttmqr_for_L = {
  .name = "L",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 3,
  .dep_in  = { &flow_of_zgetrf_param_zttmqr_for_L_dep1_atline_353 },
  .dep_out = { NULL }
};

static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_P_dep1_atline_356_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_P_dep1_atline_356 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_P_dep1_atline_356_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_P_dep1_atline_356_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_P_dep1_atline_356 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_P_dep1_atline_356_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_for_P_dep1_atline_356 = {
  .cond = NULL,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_P,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_for_P_dep1_atline_356,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_for_P_dep1_atline_356
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttmqr_for_P = {
  .name = "P",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 4,
  .dep_in  = { &flow_of_zgetrf_param_zttmqr_for_P_dep1_atline_356 },
  .dep_out = { NULL }
};

static void
iterate_successors_of_zgetrf_param_zttmqr(dague_execution_unit_t *eu, dague_execution_context_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)this_task->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int __nb_elt = -1;
  int rank_src = 0, rank_dst = 0;
  int k = this_task->locals[0].value;
  int m = this_task->locals[1].value;
  int n = this_task->locals[2].value;
  int p = this_task->locals[3].value;
  int nextp = this_task->locals[4].value;
  int prevp = this_task->locals[5].value;
  int prevm = this_task->locals[6].value;
  int type = this_task->locals[7].value;
  int type1 = this_task->locals[8].value;
  int ip = this_task->locals[9].value;
  int im = this_task->locals[10].value;
  int im1 = this_task->locals[11].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object; (void)__nb_elt;
  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;
  nc.dague_object = this_task->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, m, n);
#endif
  /* Flow of Data V */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_DEFAULT_ARENA];
#endif
#if defined(DISTRIBUTED)
    __nb_elt = 1;
#endif  /* defined(DISTRIBUTED) */
    if( (nextp != descA.mt) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttmqr;
      {
        const int zttmqr_k = k;
        if( (zttmqr_k >= (0)) && (zttmqr_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttmqr_k;
          {
            const int zttmqr_m = nextp;
            if( (zttmqr_m >= ((zttmqr_k + 1))) && (zttmqr_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttmqr_m;
              {
                const int zttmqr_n = n;
                if( (zttmqr_n >= ((zttmqr_k + 1))) && (zttmqr_n <= ((descA.nt - 1))) ) {
                  nc.locals[2].value = zttmqr_n;
                  const int zttmqr_p = zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[3].value = zttmqr_p;
                  const int zttmqr_nextp = zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[4].value = zttmqr_nextp;
                  const int zttmqr_prevp = zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[5].value = zttmqr_prevp;
                  const int zttmqr_prevm = zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[6].value = zttmqr_prevm;
                  const int zttmqr_type = zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[7].value = zttmqr_type;
                  const int zttmqr_type1 = zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[8].value = zttmqr_type1;
                  const int zttmqr_ip = zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[9].value = zttmqr_ip;
                  const int zttmqr_im = zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[10].value = zttmqr_im;
                  const int zttmqr_im1 = zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[11].value = zttmqr_im1;
#if defined(DISTRIBUTED)
                    rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttmqr_m, zttmqr_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                  if( NULL != eu ) {
                    char tmp[128], tmp1[128];
                    DEBUG(("thread %d release deps of V:%s to V:%s (from node %d to %d)\n", eu->eu_id,
                           dague_service_to_string(this_task, tmp, 128),
                           dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                  }
#endif
                    nc.priority = priority_of_zgetrf_param_zttmqr_as_expr_fct(this_task->dague_object, nc.locals);
                    if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 0, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                      return;
      }
        }
          }
            }
              }
                }
    }
    if( ((nextp == descA.mt) && (p == k)) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttmqr_out;
      {
        const int zttmqr_out_k = p;
        if( (zttmqr_out_k >= (0)) && (zttmqr_out_k <= (((descA.mt < descA.nt) ? (descA.mt - 2) : (descA.nt - 2)))) ) {
          nc.locals[0].value = zttmqr_out_k;
          {
            const int zttmqr_out_n = n;
            if( (zttmqr_out_n >= ((zttmqr_out_k + 1))) && (zttmqr_out_n <= ((descA.nt - 1))) ) {
              nc.locals[1].value = zttmqr_out_n;
              const int zttmqr_out_prevp = zgetrf_param_inline_c_expr12_line_305((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[2].value = zttmqr_out_prevp;
#if defined(DISTRIBUTED)
                rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttmqr_out_k, zttmqr_out_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
              if( NULL != eu ) {
                char tmp[128], tmp1[128];
                DEBUG(("thread %d release deps of V:%s to A:%s (from node %d to %d)\n", eu->eu_id,
                       dague_service_to_string(this_task, tmp, 128),
                       dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
              }
#endif
                nc.priority = 0;
                if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 1, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                  return;
      }
        }
          }
            }
    }
    if( ((nextp == descA.mt) && (p != k)) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttmqr;
      {
        const int zttmqr_k = k;
        if( (zttmqr_k >= (0)) && (zttmqr_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttmqr_k;
          {
            const int zttmqr_m = p;
            if( (zttmqr_m >= ((zttmqr_k + 1))) && (zttmqr_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttmqr_m;
              {
                const int zttmqr_n = n;
                if( (zttmqr_n >= ((zttmqr_k + 1))) && (zttmqr_n <= ((descA.nt - 1))) ) {
                  nc.locals[2].value = zttmqr_n;
                  const int zttmqr_p = zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[3].value = zttmqr_p;
                  const int zttmqr_nextp = zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[4].value = zttmqr_nextp;
                  const int zttmqr_prevp = zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[5].value = zttmqr_prevp;
                  const int zttmqr_prevm = zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[6].value = zttmqr_prevm;
                  const int zttmqr_type = zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[7].value = zttmqr_type;
                  const int zttmqr_type1 = zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[8].value = zttmqr_type1;
                  const int zttmqr_ip = zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[9].value = zttmqr_ip;
                  const int zttmqr_im = zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[10].value = zttmqr_im;
                  const int zttmqr_im1 = zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[11].value = zttmqr_im1;
#if defined(DISTRIBUTED)
                    rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttmqr_m, zttmqr_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                  if( NULL != eu ) {
                    char tmp[128], tmp1[128];
                    DEBUG(("thread %d release deps of V:%s to C:%s (from node %d to %d)\n", eu->eu_id,
                           dague_service_to_string(this_task, tmp, 128),
                           dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                  }
#endif
                    nc.priority = priority_of_zgetrf_param_zttmqr_as_expr_fct(this_task->dague_object, nc.locals);
                    if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 2, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                      return;
      }
        }
          }
            }
              }
                }
    }
  }
  /* Flow of Data C */
  if( action_mask & (1 << 1) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_DEFAULT_ARENA];
#endif
#if defined(DISTRIBUTED)
    __nb_elt = 1;
#endif  /* defined(DISTRIBUTED) */
    if( ((type1 != 0) && (n == (k + 1))) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zgetrf_param;
      {
        const int zgetrf_param_k = (k + 1);
        if( (zgetrf_param_k >= (0)) && (zgetrf_param_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zgetrf_param_k;
          {
            const int zgetrf_param_i = im1;
            if( (zgetrf_param_i >= (0)) && (zgetrf_param_i <= (zgetrf_param_inline_c_expr30_line_94((const dague_object_t*)__dague_object, nc.locals))) ) {
              nc.locals[1].value = zgetrf_param_i;
              const int zgetrf_param_m = zgetrf_param_inline_c_expr29_line_95((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[2].value = zgetrf_param_m;
              const int zgetrf_param_nextm = zgetrf_param_inline_c_expr28_line_96((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[3].value = zgetrf_param_nextm;
#if defined(DISTRIBUTED)
                rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zgetrf_param_m, zgetrf_param_k);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
              if( NULL != eu ) {
                char tmp[128], tmp1[128];
                DEBUG(("thread %d release deps of C:%s to A:%s (from node %d to %d)\n", eu->eu_id,
                       dague_service_to_string(this_task, tmp, 128),
                       dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
              }
#endif
                nc.priority = priority_of_zgetrf_param_zgetrf_param_as_expr_fct(this_task->dague_object, nc.locals);
                if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 1, 0, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                  return;
      }
        }
          }
            }
    }
    if( ((type1 != 0) && (n > (k + 1))) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zgessm;
      {
        const int zgessm_k = (k + 1);
        if( (zgessm_k >= (0)) && (zgessm_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zgessm_k;
          {
            const int zgessm_i = im1;
            if( (zgessm_i >= (0)) && (zgessm_i <= (zgetrf_param_inline_c_expr26_line_151((const dague_object_t*)__dague_object, nc.locals))) ) {
              nc.locals[1].value = zgessm_i;
              {
                const int zgessm_n = n;
                if( (zgessm_n >= ((zgessm_k + 1))) && (zgessm_n <= ((descA.nt - 1))) ) {
                  nc.locals[2].value = zgessm_n;
                  const int zgessm_m = zgetrf_param_inline_c_expr25_line_153((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[3].value = zgessm_m;
                  const int zgessm_nextm = zgetrf_param_inline_c_expr24_line_154((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[4].value = zgessm_nextm;
#if defined(DISTRIBUTED)
                    rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zgessm_m, zgessm_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                  if( NULL != eu ) {
                    char tmp[128], tmp1[128];
                    DEBUG(("thread %d release deps of C:%s to C:%s (from node %d to %d)\n", eu->eu_id,
                           dague_service_to_string(this_task, tmp, 128),
                           dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                  }
#endif
                    nc.priority = priority_of_zgetrf_param_zgessm_as_expr_fct(this_task->dague_object, nc.locals);
                    if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 1, 1, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                      return;
      }
        }
          }
            }
              }
                }
    }
    if( ((type1 == 0) && (n == (k + 1))) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttqrt;
      {
        const int zttqrt_k = (k + 1);
        if( (zttqrt_k >= (0)) && (zttqrt_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttqrt_k;
          {
            const int zttqrt_m = m;
            if( (zttqrt_m >= ((zttqrt_k + 1))) && (zttqrt_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttqrt_m;
              const int zttqrt_p = zgetrf_param_inline_c_expr21_line_210((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[2].value = zttqrt_p;
              const int zttqrt_nextp = zgetrf_param_inline_c_expr20_line_211((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[3].value = zttqrt_nextp;
              const int zttqrt_prevp = zgetrf_param_inline_c_expr19_line_212((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[4].value = zttqrt_prevp;
              const int zttqrt_prevm = zgetrf_param_inline_c_expr18_line_213((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[5].value = zttqrt_prevm;
              const int zttqrt_type = zgetrf_param_inline_c_expr17_line_214((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[6].value = zttqrt_type;
              const int zttqrt_ip = zgetrf_param_inline_c_expr16_line_215((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[7].value = zttqrt_ip;
              const int zttqrt_im = zgetrf_param_inline_c_expr15_line_216((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[8].value = zttqrt_im;
#if defined(DISTRIBUTED)
                rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttqrt_m, zttqrt_k);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
              if( NULL != eu ) {
                char tmp[128], tmp1[128];
                DEBUG(("thread %d release deps of C:%s to C:%s (from node %d to %d)\n", eu->eu_id,
                       dague_service_to_string(this_task, tmp, 128),
                       dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
              }
#endif
                nc.priority = priority_of_zgetrf_param_zttqrt_as_expr_fct(this_task->dague_object, nc.locals);
                if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 1, 2, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                  return;
      }
        }
          }
            }
    }
    if( ((type1 == 0) && (n > (k + 1))) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttmqr;
      {
        const int zttmqr_k = (k + 1);
        if( (zttmqr_k >= (0)) && (zttmqr_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttmqr_k;
          {
            const int zttmqr_m = m;
            if( (zttmqr_m >= ((zttmqr_k + 1))) && (zttmqr_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttmqr_m;
              {
                const int zttmqr_n = n;
                if( (zttmqr_n >= ((zttmqr_k + 1))) && (zttmqr_n <= ((descA.nt - 1))) ) {
                  nc.locals[2].value = zttmqr_n;
                  const int zttmqr_p = zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[3].value = zttmqr_p;
                  const int zttmqr_nextp = zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[4].value = zttmqr_nextp;
                  const int zttmqr_prevp = zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[5].value = zttmqr_prevp;
                  const int zttmqr_prevm = zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[6].value = zttmqr_prevm;
                  const int zttmqr_type = zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[7].value = zttmqr_type;
                  const int zttmqr_type1 = zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[8].value = zttmqr_type1;
                  const int zttmqr_ip = zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[9].value = zttmqr_ip;
                  const int zttmqr_im = zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[10].value = zttmqr_im;
                  const int zttmqr_im1 = zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[11].value = zttmqr_im1;
#if defined(DISTRIBUTED)
                    rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttmqr_m, zttmqr_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                  if( NULL != eu ) {
                    char tmp[128], tmp1[128];
                    DEBUG(("thread %d release deps of C:%s to C:%s (from node %d to %d)\n", eu->eu_id,
                           dague_service_to_string(this_task, tmp, 128),
                           dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                  }
#endif
                    nc.priority = priority_of_zgetrf_param_zttmqr_as_expr_fct(this_task->dague_object, nc.locals);
                    if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 1, 3, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                      return;
      }
        }
          }
            }
              }
                }
    }
  }
  /* Flow of data H has only IN dependencies */
  /* Flow of data L has only IN dependencies */
  /* Flow of data P has only IN dependencies */
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zgetrf_param_zttmqr(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, zttmqr_repo, zttmqr_hash(__dague_object, context->locals) );
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
  }
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zgetrf_param_zttmqr(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(zttmqr_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int m = context->locals[1].value;
    int n = context->locals[2].value;
    int p = context->locals[3].value;
    int nextp = context->locals[4].value;
    int prevp = context->locals[5].value;
    int prevm = context->locals[6].value;
    int type = context->locals[7].value;
    int type1 = context->locals[8].value;
    int ip = context->locals[9].value;
    int im = context->locals[10].value;
    int im1 = context->locals[11].value;
    (void)k; (void)m; (void)n; (void)p; (void)nextp; (void)prevp; (void)prevm; (void)type; (void)type1; (void)ip; (void)im; (void)im1;

    if( (prevp == descA.mt) ) {
      data_repo_entry_used_once( eu, zgessm_repo, context->data[0].data_repo->key );
      (void)AUNREF(context->data[0].data);
    } else {
      data_repo_entry_used_once( eu, zttmqr_repo, context->data[0].data_repo->key );
      (void)AUNREF(context->data[0].data);
    }
    if( ((type == 0) && (k != 0)) ) {
      data_repo_entry_used_once( eu, zttmqr_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    }
    else if( ((type != 0) && (prevm == descA.mt)) ) {
      data_repo_entry_used_once( eu, zgessm_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    }
    else if( ((type != 0) && (prevm != descA.mt)) ) {
      data_repo_entry_used_once( eu, zttmqr_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    }
    data_repo_entry_used_once( eu, zttqrt_repo, context->data[2].data_repo->key );
    (void)AUNREF(context->data[2].data);
    data_repo_entry_used_once( eu, zttqrt_repo, context->data[3].data_repo->key );
    (void)AUNREF(context->data[3].data);
    data_repo_entry_used_once( eu, zttqrt_repo, context->data[4].data_repo->key );
    (void)AUNREF(context->data[4].data);
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zgetrf_param_zttmqr(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = this_task->locals[0].value;
  int m = this_task->locals[1].value;
  int n = this_task->locals[2].value;
  int p = this_task->locals[3].value;
  int nextp = this_task->locals[4].value;
  int prevp = this_task->locals[5].value;
  int prevm = this_task->locals[6].value;
  int type = this_task->locals[7].value;
  int type1 = this_task->locals[8].value;
  int ip = this_task->locals[9].value;
  int im = this_task->locals[10].value;
  int im1 = this_task->locals[11].value;
  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *V = NULL; (void)V;
  dague_arena_chunk_t *gV = NULL; (void)gV;
  data_repo_entry_t *eV = NULL; (void)eV;
  void *C = NULL; (void)C;
  dague_arena_chunk_t *gC = NULL; (void)gC;
  data_repo_entry_t *eC = NULL; (void)eC;
  void *H = NULL; (void)H;
  dague_arena_chunk_t *gH = NULL; (void)gH;
  data_repo_entry_t *eH = NULL; (void)eH;
  void *L = NULL; (void)L;
  dague_arena_chunk_t *gL = NULL; (void)gL;
  data_repo_entry_t *eL = NULL; (void)eL;
  void *P = NULL; (void)P;
  dague_arena_chunk_t *gP = NULL; (void)gP;
  data_repo_entry_t *eP = NULL; (void)eP;

  /** Lookup the input data, and store them in the context if any */
  eV = this_task->data[0].data_repo;
  gV = this_task->data[0].data;
  if( NULL == gV ) {
  if( (prevp == descA.mt) ) {
      tass[0].value = k;
      tass[1].value = ip;
      tass[2].value = n;
    eV = data_repo_lookup_entry( zgessm_repo, zgessm_hash( __dague_object, tass ));
    gV = eV->data[0];
  } else {
      tass[0].value = k;
      tass[1].value = prevp;
      tass[2].value = n;
    eV = data_repo_lookup_entry( zttmqr_repo, zttmqr_hash( __dague_object, tass ));
    gV = eV->data[0];
  }
    this_task->data[0].data = gV;
    this_task->data[0].data_repo = eV;
  }
  V = ADATA(gV);
#if defined(DAGUE_SIM)
  if( (NULL != eV) && (eV->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eV->sim_exec_date;
#endif
  eC = this_task->data[1].data_repo;
  gC = this_task->data[1].data;
  if( NULL == gC ) {
  if( ((type == 0) && (k == 0)) ) {
    gC = (dague_arena_chunk_t*) A(m, n);
  }
  else if( ((type == 0) && (k != 0)) ) {
      tass[0].value = (k - 1);
      tass[1].value = m;
      tass[2].value = n;
    eC = data_repo_lookup_entry( zttmqr_repo, zttmqr_hash( __dague_object, tass ));
    gC = eC->data[1];
  }
  else if( ((type != 0) && (prevm == descA.mt)) ) {
      tass[0].value = k;
      tass[1].value = im;
      tass[2].value = n;
    eC = data_repo_lookup_entry( zgessm_repo, zgessm_hash( __dague_object, tass ));
    gC = eC->data[0];
  }
  else if( ((type != 0) && (prevm != descA.mt)) ) {
      tass[0].value = k;
      tass[1].value = prevm;
      tass[2].value = n;
    eC = data_repo_lookup_entry( zttmqr_repo, zttmqr_hash( __dague_object, tass ));
    gC = eC->data[0];
  }
    this_task->data[1].data = gC;
    this_task->data[1].data_repo = eC;
  }
  C = ADATA(gC);
#if defined(DAGUE_SIM)
  if( (NULL != eC) && (eC->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eC->sim_exec_date;
#endif
  eH = this_task->data[2].data_repo;
  gH = this_task->data[2].data;
  if( NULL == gH ) {
  tass[0].value = k;
  tass[1].value = m;
  eH = data_repo_lookup_entry( zttqrt_repo, zttqrt_hash( __dague_object, tass ));
  gH = eH->data[1];
    this_task->data[2].data = gH;
    this_task->data[2].data_repo = eH;
  }
  H = ADATA(gH);
#if defined(DAGUE_SIM)
  if( (NULL != eH) && (eH->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eH->sim_exec_date;
#endif
  eL = this_task->data[3].data_repo;
  gL = this_task->data[3].data;
  if( NULL == gL ) {
  tass[0].value = k;
  tass[1].value = m;
  eL = data_repo_lookup_entry( zttqrt_repo, zttqrt_hash( __dague_object, tass ));
  gL = eL->data[2];
    this_task->data[3].data = gL;
    this_task->data[3].data_repo = eL;
  }
  L = ADATA(gL);
#if defined(DAGUE_SIM)
  if( (NULL != eL) && (eL->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eL->sim_exec_date;
#endif
  eP = this_task->data[4].data_repo;
  gP = this_task->data[4].data;
  if( NULL == gP ) {
  tass[0].value = k;
  tass[1].value = m;
  eP = data_repo_lookup_entry( zttqrt_repo, zttqrt_hash( __dague_object, tass ));
  gP = eP->data[3];
    this_task->data[4].data = gP;
    this_task->data[4].data_repo = eP;
  }
  P = ADATA(gP);
#if defined(DAGUE_SIM)
  if( (NULL != eP) && (eP->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eP->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( this_task->function->sim_cost_fct != NULL ) {
    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);
  } else {
    this_task->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < this_task->sim_exec_date )
    context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, V);
  cache_buf_referenced(context->closest_cache, C);
  cache_buf_referenced(context->closest_cache, H);
  cache_buf_referenced(context->closest_cache, L);
  cache_buf_referenced(context->closest_cache, P);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                                  zttmqr BODY                                  *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*this_task->function->function_id, zttmqr_hash( __dague_object, this_task->locals), __dague_object->super.A, ((dague_ddesc_t*)(__dague_object->super.A))->data_key((dague_ddesc_t*)__dague_object->super.A, m, n) );
#line 358 "zgetrf_param.jdf"
  DRYRUN(

         if ( type == 0 ) {
            int tempmm = ((m) ==(descA.mt-1)) ? (descA.m -( m*descA.mb)) : (descA.mb);
            int tempnn = ((n)==(descA.nt-1)) ? (descA.n -(n*descA.nb)) : (descA.nb);
            int ldak   = descA.mb; /*((k+(Ai/descA.mb))<Alm1) ? (descA.mb) : (Alm%descA.mb);*/
            int ldam   = descA.mb; /*((m+(Ai/descA.mb))<Alm1) ? (descA.mb) : (Alm%descA.mb);*/

            CORE_zssssm(descA.nb, tempnn, tempmm, tempnn, descA.nb, ib,
                     V /* A(k,n) */, ldak,
                     C /* A(m,n) */, ldam,
                     L /* L(m,k) */,  descL.mb,
                     H /* A(m,k) */,  ldam,
                     P /* IPIV(m,k) */ );
         } else {
            void *p_elem_A = dague_private_memory_pop( p_work );

            int tempnn = ((n)==((descA.nt)-1)) ? ((descA.n)-(n*(descA.nb))) : (descA.nb);
            int tempmm = ((m)==((descA.mt)-1)) ? ((descA.m)-(m*(descA.mb))) : (descA.mb);
            int ldam = BLKLDD( descA, m );
            int ldwork = ib;

            CORE_zttmqr(
                PlasmaLeft, PlasmaConjTrans,
                descA.mb, tempnn, tempmm, tempnn, descA.nb, ib,
                V /* A(p, n) */, descA.mb,
                C /* A(m, n) */, ldam,
                H  /* A(m, k) */, ldam,
                L  /* T(m, k) */, descL.mb,
                p_elem_A, ldwork );

            dague_private_memory_push( p_work, p_elem_A );
         }

         );

  printlog("thread %d CORE_zttmqr(%d, %d, %d)\n"
           "\t(PlasmaLeft, PlasmaConjTrans, descA.mb, tempnn, tempmm, tempnn, descA.nb, ib, \n"
           "\t A(%d,%d)[%p], A.mb, A(%d,%d)[%p], ldam, A(%d,%d)[%p], ldam, T(%d,%d)[%p], descL.mb, p_elem_A, ldwork)\n",
           context->eu_id, k, m, n, p, n, A1, m, n, A2, m, k, V, m, k, T);


#line 2815 "zgetrf_param.c"
/*--------------------------------------------------------------------------------*
 *                              END OF zttmqr BODY                              *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zgetrf_param_zttmqr(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  (void)context; (void)__dague_object;
  int k = this_task->locals[0].value;
  int m = this_task->locals[1].value;
  int n = this_task->locals[2].value;
  int p = this_task->locals[3].value;
  int nextp = this_task->locals[4].value;
  int prevp = this_task->locals[5].value;
  int prevm = this_task->locals[6].value;
  int type = this_task->locals[7].value;
  int type1 = this_task->locals[8].value;
  int ip = this_task->locals[9].value;
  int im = this_task->locals[10].value;
  int im1 = this_task->locals[11].value;
  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;

  TAKE_TIME(context,2*this_task->function->function_id+1, zttmqr_hash( __dague_object, this_task->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(this_task, context->eu_id, zttmqr_hash(__dague_object, this_task->locals));
  release_deps_of_zgetrf_param_zttmqr(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      DAGUE_ACTION_DEPS_MASK,
      NULL);
  return 0;
}

#if defined(DAGUE_SIM)
static int simulation_cost_of_zgetrf_param_zttmqr(const dague_execution_context_t *this_task)
{
  const dague_object_t *__dague_object = (const dague_object_t*)this_task->dague_object;
  int k = this_task->locals[0].value;
  int m = this_task->locals[1].value;
  int n = this_task->locals[2].value;
  int p = this_task->locals[3].value;
  int nextp = this_task->locals[4].value;
  int prevp = this_task->locals[5].value;
  int prevm = this_task->locals[6].value;
  int type = this_task->locals[7].value;
  int type1 = this_task->locals[8].value;
  int ip = this_task->locals[9].value;
  int im = this_task->locals[10].value;
  int im1 = this_task->locals[11].value;
  (void)__dague_object;
  (void)k;  (void)m;  (void)n;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)type1;  (void)ip;  (void)im;  (void)im1;
  return zgetrf_param_inline_c_expr2_line_330((const dague_object_t*)__dague_object, this_task->locals);
}
#endif

static int zgetrf_param_zttmqr_internal_init(__dague_zgetrf_param_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, m, n, p, nextp, prevp, prevm, type, type1, ip, im, im1;
  int32_t  k_min = 0x7fffffff, m_min = 0x7fffffff, n_min = 0x7fffffff;
  int32_t  k_max = 0, m_max = 0, n_max = 0;
  (void)__dague_object; (void)__foundone;
  int32_t k_start, k_end;  int32_t m_start, m_end;  int32_t n_start, n_end;  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
      k++) {
    assignments[0].value = k;
    for(m = (k + 1);
        m <= (descA.mt - 1);
        m++) {
      assignments[1].value = m;
      for(n = (k + 1);
          n <= (descA.nt - 1);
          n++) {
        assignments[2].value = n;
        p = zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, assignments);
        assignments[3].value = p;
        nextp = zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, assignments);
        assignments[4].value = nextp;
        prevp = zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, assignments);
        assignments[5].value = prevp;
        prevm = zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, assignments);
        assignments[6].value = prevm;
        type = zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, assignments);
        assignments[7].value = type;
        type1 = zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, assignments);
        assignments[8].value = type1;
        ip = zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, assignments);
        assignments[9].value = ip;
        im = zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, assignments);
        assignments[10].value = im;
        im1 = zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, assignments);
        assignments[11].value = im1;
        if( !zttmqr_pred(k, m, n, p, nextp, prevp, prevm, type, type1, ip, im, im1) ) continue;
        nb_tasks++;
        k_max = dague_imax(k_max, k);
        k_min = dague_imin(k_min, k);
        m_max = dague_imax(m_max, m);
        m_min = dague_imin(m_min, m);
        n_max = dague_imax(n_max, n);
        n_min = dague_imin(n_min, n);
      }
    }
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  dep = NULL;
  k_start = 0;
  k_end = ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
  for(k = dague_imax(k_start, k_min); k <= dague_imin(k_end, k_max); k++) {
    assignments[0].value = k;
    m_start = (k + 1);
    m_end = (descA.mt - 1);
    for(m = dague_imax(m_start, m_min); m <= dague_imin(m_end, m_max); m++) {
      assignments[1].value = m;
      n_start = (k + 1);
      n_end = (descA.nt - 1);
      for(n = dague_imax(n_start, n_min); n <= dague_imin(n_end, n_max); n++) {
        assignments[2].value = n;
        p = zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, assignments);
        assignments[3].value = p;
        nextp = zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, assignments);
        assignments[4].value = nextp;
        prevp = zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, assignments);
        assignments[5].value = prevp;
        prevm = zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, assignments);
        assignments[6].value = prevm;
        type = zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, assignments);
        assignments[7].value = type;
        type1 = zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, assignments);
        assignments[8].value = type1;
        ip = zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, assignments);
        assignments[9].value = ip;
        im = zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, assignments);
        assignments[10].value = im;
        __foundone = 0;
        im1 = zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, assignments);
        assignments[11].value = im1;
        if( zttmqr_pred(k, m, n, p, nextp, prevp, prevm, type, type1, ip, im, im1) ) {
          /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zgetrf_param_zttmqr_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-k_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-k_min], m_min, m_max, "m", &symb_zgetrf_param_zttmqr_m, dep, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-k_min]->u.next[m-m_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-k_min]->u.next[m-m_min], n_min, n_max, "n", &symb_zgetrf_param_zttmqr_n, dep->u.next[k-k_min], DAGUE_DEPENDENCIES_FLAG_FINAL);
        }
        }
      }
    }
  }
  (void)k_start; (void)k_end;  (void)m_start; (void)m_end;  (void)n_start; (void)n_end;  __dague_object->super.super.dependencies_array[0] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static const dague_function_t zgetrf_param_zttmqr = {
  .name = "zttmqr",
  .deps = 0,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 0,
  .dependencies_goal = 0x1f,
  .nb_parameters = 3,
  .nb_definitions = 12,
  .params = { &symb_zgetrf_param_zttmqr_k, &symb_zgetrf_param_zttmqr_m, &symb_zgetrf_param_zttmqr_n },
  .locals = { &symb_zgetrf_param_zttmqr_k, &symb_zgetrf_param_zttmqr_m, &symb_zgetrf_param_zttmqr_n, &symb_zgetrf_param_zttmqr_p, &symb_zgetrf_param_zttmqr_nextp, &symb_zgetrf_param_zttmqr_prevp, &symb_zgetrf_param_zttmqr_prevm, &symb_zgetrf_param_zttmqr_type, &symb_zgetrf_param_zttmqr_type1, &symb_zgetrf_param_zttmqr_ip, &symb_zgetrf_param_zttmqr_im, &symb_zgetrf_param_zttmqr_im1 },
  .pred = &pred_of_zgetrf_param_zttmqr_as_expr,
  .priority = &priority_of_zgetrf_param_zttmqr_as_expr,
  .in = { &flow_of_zgetrf_param_zttmqr_for_V, &flow_of_zgetrf_param_zttmqr_for_C, &flow_of_zgetrf_param_zttmqr_for_H, &flow_of_zgetrf_param_zttmqr_for_L, &flow_of_zgetrf_param_zttmqr_for_P },
  .out = { &flow_of_zgetrf_param_zttmqr_for_V, &flow_of_zgetrf_param_zttmqr_for_C },
  .iterate_successors = iterate_successors_of_zgetrf_param_zttmqr,
  .release_deps = release_deps_of_zgetrf_param_zttmqr,
  .hook = hook_of_zgetrf_param_zttmqr,
  .complete_execution = complete_hook_of_zgetrf_param_zttmqr,
#if defined(DAGUE_SIM)
  .sim_cost_fct = simulation_cost_of_zgetrf_param_zttmqr,
#endif
  .key = (dague_functionkey_fn_t*)zttmqr_hash,
};


/**********************************************************************************
 *                                  zttmqr_out                                  *
 **********************************************************************************/

static inline int minexpr_of_symb_zgetrf_param_zttmqr_out_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgetrf_param_zttmqr_out_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zttmqr_out_k_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zttmqr_out_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return ((descA.mt < descA.nt) ? (descA.mt - 2) : (descA.nt - 2));
}
static const expr_t maxexpr_of_symb_zgetrf_param_zttmqr_out_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zttmqr_out_k_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_out_k = {.min = &minexpr_of_symb_zgetrf_param_zttmqr_out_k, .max = &maxexpr_of_symb_zgetrf_param_zttmqr_out_k,  .flags = 0x0};

static inline int minexpr_of_symb_zgetrf_param_zttmqr_out_n_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t minexpr_of_symb_zgetrf_param_zttmqr_out_n = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zttmqr_out_n_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zttmqr_out_n_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return (descA.nt - 1);
}
static const expr_t maxexpr_of_symb_zgetrf_param_zttmqr_out_n = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zttmqr_out_n_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_out_n = {.min = &minexpr_of_symb_zgetrf_param_zttmqr_out_n, .max = &maxexpr_of_symb_zgetrf_param_zttmqr_out_n,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttmqr_out_prevp_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr12_line_305((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttmqr_out_prevp = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttmqr_out_prevp_fct
};
static const symbol_t symb_zgetrf_param_zttmqr_out_prevp = {.min = &expr_of_symb_zgetrf_param_zttmqr_out_prevp, .max = &expr_of_symb_zgetrf_param_zttmqr_out_prevp,  .flags = 0x0};

static inline int pred_of_zgetrf_param_zttmqr_out_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int n = assignments[1].value;
  int prevp = assignments[2].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)n;
  (void)prevp;
  /* Compute Predicate */
  return zttmqr_out_pred(k, n, prevp);
}
static const expr_t pred_of_zgetrf_param_zttmqr_out_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zgetrf_param_zttmqr_out_as_expr_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevp = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return prevp;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310 = {
  .cond = NULL,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_V,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310,
    &expr_of_p3_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310
  }
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep2_atline_313_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep2_atline_313 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep2_atline_313_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep2_atline_313_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep2_atline_313 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep2_atline_313_fct
};
static const dep_t flow_of_zgetrf_param_zttmqr_out_for_A_dep2_atline_313 = {
  .cond = NULL,
  .dague = &zgetrf_param_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep2_atline_313,
    &expr_of_p2_for_flow_of_zgetrf_param_zttmqr_out_for_A_dep2_atline_313
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttmqr_out_for_A = {
  .name = "A",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 0,
  .dep_in  = { &flow_of_zgetrf_param_zttmqr_out_for_A_dep1_atline_310 },
  .dep_out = { &flow_of_zgetrf_param_zttmqr_out_for_A_dep2_atline_313 }
};

static void
iterate_successors_of_zgetrf_param_zttmqr_out(dague_execution_unit_t *eu, dague_execution_context_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)this_task->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int __nb_elt = -1;
  int rank_src = 0, rank_dst = 0;
  int k = this_task->locals[0].value;
  int n = this_task->locals[1].value;
  int prevp = this_task->locals[2].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object; (void)__nb_elt;
  (void)k;  (void)n;  (void)prevp;
  nc.dague_object = this_task->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, k, n);
#endif
  /* Flow of data A has only OUTPUT dependencies to Memory */
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zgetrf_param_zttmqr_out(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, zttmqr_out_repo, zttmqr_out_hash(__dague_object, context->locals) );
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
  }
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zgetrf_param_zttmqr_out(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(zttmqr_out_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int n = context->locals[1].value;
    int prevp = context->locals[2].value;
    (void)k; (void)n; (void)prevp;

    data_repo_entry_used_once( eu, zttmqr_repo, context->data[0].data_repo->key );
    (void)AUNREF(context->data[0].data);
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zgetrf_param_zttmqr_out(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = this_task->locals[0].value;
  int n = this_task->locals[1].value;
  int prevp = this_task->locals[2].value;
  (void)k;  (void)n;  (void)prevp;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;

  /** Lookup the input data, and store them in the context if any */
  eA = this_task->data[0].data_repo;
  gA = this_task->data[0].data;
  if( NULL == gA ) {
  tass[0].value = k;
  tass[1].value = prevp;
  tass[2].value = n;
  eA = data_repo_lookup_entry( zttmqr_repo, zttmqr_hash( __dague_object, tass ));
  gA = eA->data[0];
    this_task->data[0].data = gA;
    this_task->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( this_task->function->sim_cost_fct != NULL ) {
    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);
  } else {
    this_task->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < this_task->sim_exec_date )
    context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                                zttmqr_out BODY                                *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*this_task->function->function_id, zttmqr_out_hash( __dague_object, this_task->locals), __dague_object->super.A, ((dague_ddesc_t*)(__dague_object->super.A))->data_key((dague_ddesc_t*)__dague_object->super.A, k, n) );
#line 311 "zgetrf_param.jdf"
/* nothing */

#line 3330 "zgetrf_param.c"
/*--------------------------------------------------------------------------------*
 *                            END OF zttmqr_out BODY                            *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zgetrf_param_zttmqr_out(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  (void)context; (void)__dague_object;
  int k = this_task->locals[0].value;
  int n = this_task->locals[1].value;
  int prevp = this_task->locals[2].value;
  (void)k;  (void)n;  (void)prevp;

  TAKE_TIME(context,2*this_task->function->function_id+1, zttmqr_out_hash( __dague_object, this_task->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( ADATA(this_task->data[0].data) != A(k, n) ) {
    int __arena_index = DAGUE_zgetrf_param_DEFAULT_ARENA;
    int __dtt_nb = 1;
    assert( (__arena_index>=0) && (__arena_index < __dague_object->super.arenas_size) );
    assert( __dtt_nb >= 0 );
    dague_remote_dep_memcpy( context, this_task->dague_object, A(k, n), this_task->data[0].data, 
                             __dague_object->super.arenas[__arena_index]->opaque_dtt,
                             __dtt_nb );
  }
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(this_task, context->eu_id, zttmqr_out_hash(__dague_object, this_task->locals));
  release_deps_of_zgetrf_param_zttmqr_out(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      DAGUE_ACTION_DEPS_MASK,
      NULL);
  return 0;
}

static int zgetrf_param_zttmqr_out_internal_init(__dague_zgetrf_param_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, n, prevp;
  int32_t  k_min = 0x7fffffff, n_min = 0x7fffffff;
  int32_t  k_max = 0, n_max = 0;
  (void)__dague_object; (void)__foundone;
  int32_t k_start, k_end;  int32_t n_start, n_end;  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= ((descA.mt < descA.nt) ? (descA.mt - 2) : (descA.nt - 2));
      k++) {
    assignments[0].value = k;
    for(n = (k + 1);
        n <= (descA.nt - 1);
        n++) {
      assignments[1].value = n;
      prevp = zgetrf_param_inline_c_expr12_line_305((const dague_object_t*)__dague_object, assignments);
      assignments[2].value = prevp;
      if( !zttmqr_out_pred(k, n, prevp) ) continue;
      nb_tasks++;
      k_max = dague_imax(k_max, k);
      k_min = dague_imin(k_min, k);
      n_max = dague_imax(n_max, n);
      n_min = dague_imin(n_min, n);
    }
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  dep = NULL;
  k_start = 0;
  k_end = ((descA.mt < descA.nt) ? (descA.mt - 2) : (descA.nt - 2));
  for(k = dague_imax(k_start, k_min); k <= dague_imin(k_end, k_max); k++) {
    assignments[0].value = k;
    n_start = (k + 1);
    n_end = (descA.nt - 1);
    for(n = dague_imax(n_start, n_min); n <= dague_imin(n_end, n_max); n++) {
      assignments[1].value = n;
      __foundone = 0;
      prevp = zgetrf_param_inline_c_expr12_line_305((const dague_object_t*)__dague_object, assignments);
      assignments[2].value = prevp;
      if( zttmqr_out_pred(k, n, prevp) ) {
        /* We did find one! Allocate the dependencies array. */
      if( dep == NULL ) {
        ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zgetrf_param_zttmqr_out_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
      }
      if( dep->u.next[k-k_min] == NULL ) {
        ALLOCATE_DEP_TRACKING(dep->u.next[k-k_min], n_min, n_max, "n", &symb_zgetrf_param_zttmqr_out_n, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
      }
      }
    }
  }
  (void)k_start; (void)k_end;  (void)n_start; (void)n_end;  __dague_object->super.super.dependencies_array[1] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static const dague_function_t zgetrf_param_zttmqr_out = {
  .name = "zttmqr_out",
  .deps = 1,
  .flags = 0x0,
  .function_id = 1,
  .dependencies_goal = 0x1,
  .nb_parameters = 2,
  .nb_definitions = 3,
  .params = { &symb_zgetrf_param_zttmqr_out_k, &symb_zgetrf_param_zttmqr_out_n },
  .locals = { &symb_zgetrf_param_zttmqr_out_k, &symb_zgetrf_param_zttmqr_out_n, &symb_zgetrf_param_zttmqr_out_prevp },
  .pred = &pred_of_zgetrf_param_zttmqr_out_as_expr,
  .priority = NULL,
  .in = { &flow_of_zgetrf_param_zttmqr_out_for_A },
  .out = { &flow_of_zgetrf_param_zttmqr_out_for_A },
  .iterate_successors = iterate_successors_of_zgetrf_param_zttmqr_out,
  .release_deps = release_deps_of_zgetrf_param_zttmqr_out,
  .hook = hook_of_zgetrf_param_zttmqr_out,
  .complete_execution = complete_hook_of_zgetrf_param_zttmqr_out,
#if defined(DAGUE_SIM)
  .sim_cost_fct = NULL,
#endif
  .key = (dague_functionkey_fn_t*)zttmqr_out_hash,
};


/**********************************************************************************
 *                                    zttqrt                                    *
 **********************************************************************************/

static inline int minexpr_of_symb_zgetrf_param_zttqrt_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgetrf_param_zttqrt_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zttqrt_k_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zttqrt_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
}
static const expr_t maxexpr_of_symb_zgetrf_param_zttqrt_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zttqrt_k_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_k = {.min = &minexpr_of_symb_zgetrf_param_zttqrt_k, .max = &maxexpr_of_symb_zgetrf_param_zttqrt_k,  .flags = 0x0};

static inline int minexpr_of_symb_zgetrf_param_zttqrt_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t minexpr_of_symb_zgetrf_param_zttqrt_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zttqrt_m_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zttqrt_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return (descA.mt - 1);
}
static const expr_t maxexpr_of_symb_zgetrf_param_zttqrt_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zttqrt_m_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_m = {.min = &minexpr_of_symb_zgetrf_param_zttqrt_m, .max = &maxexpr_of_symb_zgetrf_param_zttqrt_m,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttqrt_p_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr21_line_210((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttqrt_p = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttqrt_p_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_p = {.min = &expr_of_symb_zgetrf_param_zttqrt_p, .max = &expr_of_symb_zgetrf_param_zttqrt_p,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttqrt_nextp_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr20_line_211((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttqrt_nextp = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttqrt_nextp_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_nextp = {.min = &expr_of_symb_zgetrf_param_zttqrt_nextp, .max = &expr_of_symb_zgetrf_param_zttqrt_nextp,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttqrt_prevp_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr19_line_212((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttqrt_prevp = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttqrt_prevp_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_prevp = {.min = &expr_of_symb_zgetrf_param_zttqrt_prevp, .max = &expr_of_symb_zgetrf_param_zttqrt_prevp,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttqrt_prevm_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr18_line_213((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttqrt_prevm = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttqrt_prevm_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_prevm = {.min = &expr_of_symb_zgetrf_param_zttqrt_prevm, .max = &expr_of_symb_zgetrf_param_zttqrt_prevm,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttqrt_type_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr17_line_214((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttqrt_type = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttqrt_type_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_type = {.min = &expr_of_symb_zgetrf_param_zttqrt_type, .max = &expr_of_symb_zgetrf_param_zttqrt_type,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttqrt_ip_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr16_line_215((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttqrt_ip = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttqrt_ip_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_ip = {.min = &expr_of_symb_zgetrf_param_zttqrt_ip, .max = &expr_of_symb_zgetrf_param_zttqrt_ip,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttqrt_im_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr15_line_216((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttqrt_im = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttqrt_im_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_im = {.min = &expr_of_symb_zgetrf_param_zttqrt_im, .max = &expr_of_symb_zgetrf_param_zttqrt_im,  .flags = 0x0};

static inline int pred_of_zgetrf_param_zttqrt_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int m = assignments[1].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;
  int prevp = assignments[4].value;
  int prevm = assignments[5].value;
  int type = assignments[6].value;
  int ip = assignments[7].value;
  int im = assignments[8].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)m;
  (void)p;
  (void)nextp;
  (void)prevp;
  (void)prevm;
  (void)type;
  (void)ip;
  (void)im;
  /* Compute Predicate */
  return zttqrt_pred(k, m, p, nextp, prevp, prevm, type, ip, im);
}
static const expr_t pred_of_zgetrf_param_zttqrt_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zgetrf_param_zttqrt_as_expr_fct
};
static inline int priority_of_zgetrf_param_zttqrt_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr13_line_243((const dague_object_t*)__dague_object, assignments);
}
static const expr_t priority_of_zgetrf_param_zttqrt_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = priority_of_zgetrf_param_zttqrt_as_expr_fct
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevp = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return (prevp == descA.mt);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int ip = assignments[7].value;

  (void)__dague_object; (void)assignments;
  return ip;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226,
  .dague = &zgetrf_param_zgetrf_param,
  .flow = &flow_of_zgetrf_param_zgetrf_param_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_UPPER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevp = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return !(prevp == descA.mt);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevp = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return prevp;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_UPPER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int nextp = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return (nextp != descA.mt);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int nextp = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return nextp;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_UPPER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep3_atline_228_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return ((nextp == descA.mt) && (p == k));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep3_atline_228 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep3_atline_228_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep3_atline_228_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep3_atline_228 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep3_atline_228_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_A_dep3_atline_228 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep3_atline_228,
  .dague = &zgetrf_param_zttqrt_out_A1,
  .flow = &flow_of_zgetrf_param_zttqrt_out_A1_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_UPPER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep3_atline_228
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int p = assignments[2].value;
  int nextp = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return ((nextp == descA.mt) && (p != k));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int p = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return p;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_UPPER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttqrt_for_A = {
  .name = "A",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 0,
  .dep_in  = { &flow_of_zgetrf_param_zttqrt_for_A_dep1_iftrue_atline_226, &flow_of_zgetrf_param_zttqrt_for_A_dep1_iffalse_atline_226 },
  .dep_out = { &flow_of_zgetrf_param_zttqrt_for_A_dep2_atline_227, &flow_of_zgetrf_param_zttqrt_for_A_dep3_atline_228, &flow_of_zgetrf_param_zttqrt_for_A_dep4_atline_229 }
};

static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int type = assignments[6].value;

  (void)__dague_object; (void)assignments;
  return ((type == 0) && (k == 0));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231,
  .dague = &zgetrf_param_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int type = assignments[6].value;

  (void)__dague_object; (void)assignments;
  return ((type == 0) && (k != 0));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k - 1);
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232,
    &expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevm = assignments[5].value;
  int type = assignments[6].value;

  (void)__dague_object; (void)assignments;
  return ((type != 0) && (prevm == descA.mt));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int im = assignments[8].value;

  (void)__dague_object; (void)assignments;
  return im;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233,
  .dague = &zgetrf_param_zgetrf_param,
  .flow = &flow_of_zgetrf_param_zgetrf_param_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevm = assignments[5].value;
  int type = assignments[6].value;

  (void)__dague_object; (void)assignments;
  return ((type != 0) && (prevm != descA.mt));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevm = assignments[5].value;

  (void)__dague_object; (void)assignments;
  return prevm;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234
  }
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep5_atline_235_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep5_atline_235 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep5_atline_235_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep5_atline_235_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep5_atline_235 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep5_atline_235_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_C_dep5_atline_235 = {
  .cond = NULL,
  .dague = &zgetrf_param_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep5_atline_235,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep5_atline_235
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return ((descA.nt - 1) > k);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236_fct
};
static inline int rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236_fct
};
static inline int rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return (descA.nt - 1);
}
static const expr_t rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236_fct
};
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236 = {
  .op = EXPR_OP_BINARY_RANGE,
  .u_expr.binary = {
    .op1 = &rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236,
    .op2 = &rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236
  }
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_H,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236,
    &expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttqrt_for_C = {
  .name = "C",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 1,
  .dep_in  = { &flow_of_zgetrf_param_zttqrt_for_C_dep1_atline_231, &flow_of_zgetrf_param_zttqrt_for_C_dep2_atline_232, &flow_of_zgetrf_param_zttqrt_for_C_dep3_atline_233, &flow_of_zgetrf_param_zttqrt_for_C_dep4_atline_234 },
  .dep_out = { &flow_of_zgetrf_param_zttqrt_for_C_dep5_atline_235, &flow_of_zgetrf_param_zttqrt_for_C_dep6_atline_236 }
};

static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep1_atline_236_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep1_atline_236 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep1_atline_236_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep1_atline_236_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep1_atline_236 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep1_atline_236_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_L_dep1_atline_236 = {
  .cond = NULL,
  .dague = &zgetrf_param_L,
  .datatype = { .index = DAGUE_zgetrf_param_SMALL_L_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep1_atline_236,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep1_atline_236
  }
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep2_atline_237_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep2_atline_237 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep2_atline_237_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep2_atline_237_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep2_atline_237 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep2_atline_237_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_L_dep2_atline_237 = {
  .cond = NULL,
  .dague = &zgetrf_param_L,
  .datatype = { .index = DAGUE_zgetrf_param_SMALL_L_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep2_atline_237,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep2_atline_237
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return ((descA.nt - 1) > k);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238_fct
};
static inline int rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238_fct
};
static inline int rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return (descA.nt - 1);
}
static const expr_t rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238_fct
};
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238 = {
  .op = EXPR_OP_BINARY_RANGE,
  .u_expr.binary = {
    .op1 = &rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238,
    .op2 = &rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238
  }
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_L,
  .datatype = { .index = DAGUE_zgetrf_param_SMALL_L_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238,
    &expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttqrt_for_L = {
  .name = "L",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 2,
  .dep_in  = { &flow_of_zgetrf_param_zttqrt_for_L_dep1_atline_236 },
  .dep_out = { &flow_of_zgetrf_param_zttqrt_for_L_dep2_atline_237, &flow_of_zgetrf_param_zttqrt_for_L_dep3_atline_238 }
};

static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep1_atline_240_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep1_atline_240 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep1_atline_240_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep1_atline_240_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep1_atline_240 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep1_atline_240_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_P_dep1_atline_240 = {
  .cond = NULL,
  .dague = &zgetrf_param_IPIV,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep1_atline_240,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep1_atline_240
  }
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep2_atline_241_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep2_atline_241 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep2_atline_241_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep2_atline_241_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep2_atline_241 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep2_atline_241_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_P_dep2_atline_241 = {
  .cond = NULL,
  .dague = &zgetrf_param_IPIV,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep2_atline_241,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep2_atline_241
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return ((descA.nt - 1) > k);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243_fct
};
static inline int rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243_fct
};
static inline int rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return (descA.nt - 1);
}
static const expr_t rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243_fct
};
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243 = {
  .op = EXPR_OP_BINARY_RANGE,
  .u_expr.binary = {
    .op1 = &rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243,
    .op2 = &rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243
  }
};
static const dep_t flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_P,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243,
    &expr_of_p3_for_flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttqrt_for_P = {
  .name = "P",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 3,
  .dep_in  = { &flow_of_zgetrf_param_zttqrt_for_P_dep1_atline_240 },
  .dep_out = { &flow_of_zgetrf_param_zttqrt_for_P_dep2_atline_241, &flow_of_zgetrf_param_zttqrt_for_P_dep3_atline_243 }
};

static void
iterate_successors_of_zgetrf_param_zttqrt(dague_execution_unit_t *eu, dague_execution_context_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)this_task->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int __nb_elt = -1;
  int rank_src = 0, rank_dst = 0;
  int k = this_task->locals[0].value;
  int m = this_task->locals[1].value;
  int p = this_task->locals[2].value;
  int nextp = this_task->locals[3].value;
  int prevp = this_task->locals[4].value;
  int prevm = this_task->locals[5].value;
  int type = this_task->locals[6].value;
  int ip = this_task->locals[7].value;
  int im = this_task->locals[8].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object; (void)__nb_elt;
  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;
  nc.dague_object = this_task->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, m, k);
#endif
  /* Flow of Data A */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_UPPER_TILE_ARENA];
#endif
#if defined(DISTRIBUTED)
    __nb_elt = 1;
#endif  /* defined(DISTRIBUTED) */
    if( (nextp != descA.mt) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttqrt;
      {
        const int zttqrt_k = k;
        if( (zttqrt_k >= (0)) && (zttqrt_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttqrt_k;
          {
            const int zttqrt_m = nextp;
            if( (zttqrt_m >= ((zttqrt_k + 1))) && (zttqrt_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttqrt_m;
              const int zttqrt_p = zgetrf_param_inline_c_expr21_line_210((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[2].value = zttqrt_p;
              const int zttqrt_nextp = zgetrf_param_inline_c_expr20_line_211((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[3].value = zttqrt_nextp;
              const int zttqrt_prevp = zgetrf_param_inline_c_expr19_line_212((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[4].value = zttqrt_prevp;
              const int zttqrt_prevm = zgetrf_param_inline_c_expr18_line_213((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[5].value = zttqrt_prevm;
              const int zttqrt_type = zgetrf_param_inline_c_expr17_line_214((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[6].value = zttqrt_type;
              const int zttqrt_ip = zgetrf_param_inline_c_expr16_line_215((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[7].value = zttqrt_ip;
              const int zttqrt_im = zgetrf_param_inline_c_expr15_line_216((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[8].value = zttqrt_im;
#if defined(DISTRIBUTED)
                rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttqrt_m, zttqrt_k);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
              if( NULL != eu ) {
                char tmp[128], tmp1[128];
                DEBUG(("thread %d release deps of A:%s to A:%s (from node %d to %d)\n", eu->eu_id,
                       dague_service_to_string(this_task, tmp, 128),
                       dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
              }
#endif
                nc.priority = priority_of_zgetrf_param_zttqrt_as_expr_fct(this_task->dague_object, nc.locals);
                if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 0, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                  return;
      }
        }
          }
            }
    }
    if( ((nextp == descA.mt) && (p == k)) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttqrt_out_A1;
      {
        const int zttqrt_out_A1_k = k;
        if( (zttqrt_out_A1_k >= (0)) && (zttqrt_out_A1_k <= (((descA.mt <= descA.nt) ? (descA.mt - 2) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttqrt_out_A1_k;
          const int zttqrt_out_A1_prevp = zgetrf_param_inline_c_expr22_line_196((const dague_object_t*)__dague_object, nc.locals);
          nc.locals[1].value = zttqrt_out_A1_prevp;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttqrt_out_A1_k, zttqrt_out_A1_k);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
          if( NULL != eu ) {
            char tmp[128], tmp1[128];
            DEBUG(("thread %d release deps of A:%s to A:%s (from node %d to %d)\n", eu->eu_id,
                   dague_service_to_string(this_task, tmp, 128),
                   dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
          }
#endif
            nc.priority = 0;
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 1, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
              return;
      }
        }
    }
    if( ((nextp == descA.mt) && (p != k)) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttqrt;
      {
        const int zttqrt_k = k;
        if( (zttqrt_k >= (0)) && (zttqrt_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttqrt_k;
          {
            const int zttqrt_m = p;
            if( (zttqrt_m >= ((zttqrt_k + 1))) && (zttqrt_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttqrt_m;
              const int zttqrt_p = zgetrf_param_inline_c_expr21_line_210((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[2].value = zttqrt_p;
              const int zttqrt_nextp = zgetrf_param_inline_c_expr20_line_211((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[3].value = zttqrt_nextp;
              const int zttqrt_prevp = zgetrf_param_inline_c_expr19_line_212((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[4].value = zttqrt_prevp;
              const int zttqrt_prevm = zgetrf_param_inline_c_expr18_line_213((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[5].value = zttqrt_prevm;
              const int zttqrt_type = zgetrf_param_inline_c_expr17_line_214((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[6].value = zttqrt_type;
              const int zttqrt_ip = zgetrf_param_inline_c_expr16_line_215((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[7].value = zttqrt_ip;
              const int zttqrt_im = zgetrf_param_inline_c_expr15_line_216((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[8].value = zttqrt_im;
#if defined(DISTRIBUTED)
                rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttqrt_m, zttqrt_k);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
              if( NULL != eu ) {
                char tmp[128], tmp1[128];
                DEBUG(("thread %d release deps of A:%s to C:%s (from node %d to %d)\n", eu->eu_id,
                       dague_service_to_string(this_task, tmp, 128),
                       dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
              }
#endif
                nc.priority = priority_of_zgetrf_param_zttqrt_as_expr_fct(this_task->dague_object, nc.locals);
                if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 2, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                  return;
      }
        }
          }
            }
    }
  }
  /* Flow of Data C */
  if( action_mask & (1 << 1) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_DEFAULT_ARENA];
#endif
#if defined(DISTRIBUTED)
    __nb_elt = 1;
#endif  /* defined(DISTRIBUTED) */
    if( ((descA.nt - 1) > k) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttmqr;
      {
        const int zttmqr_k = k;
        if( (zttmqr_k >= (0)) && (zttmqr_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttmqr_k;
          {
            const int zttmqr_m = m;
            if( (zttmqr_m >= ((zttmqr_k + 1))) && (zttmqr_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttmqr_m;
              {
                int zttmqr_n;
                for( zttmqr_n = (k + 1);zttmqr_n <= (descA.nt - 1); zttmqr_n++ ) {
                  if( (zttmqr_n >= ((zttmqr_k + 1))) && (zttmqr_n <= ((descA.nt - 1))) ) {
                    nc.locals[2].value = zttmqr_n;
                    const int zttmqr_p = zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[3].value = zttmqr_p;
                    const int zttmqr_nextp = zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[4].value = zttmqr_nextp;
                    const int zttmqr_prevp = zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[5].value = zttmqr_prevp;
                    const int zttmqr_prevm = zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[6].value = zttmqr_prevm;
                    const int zttmqr_type = zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[7].value = zttmqr_type;
                    const int zttmqr_type1 = zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[8].value = zttmqr_type1;
                    const int zttmqr_ip = zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[9].value = zttmqr_ip;
                    const int zttmqr_im = zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[10].value = zttmqr_im;
                    const int zttmqr_im1 = zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[11].value = zttmqr_im1;
#if defined(DISTRIBUTED)
                      rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttmqr_m, zttmqr_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                    if( NULL != eu ) {
                      char tmp[128], tmp1[128];
                      DEBUG(("thread %d release deps of C:%s to H:%s (from node %d to %d)\n", eu->eu_id,
                             dague_service_to_string(this_task, tmp, 128),
                             dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                    }
#endif
                      nc.priority = priority_of_zgetrf_param_zttmqr_as_expr_fct(this_task->dague_object, nc.locals);
                      if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 1, 1, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                        return;
      }
        }
          }
            }
              }
                }
                  }
    }
  }
  /* Flow of Data L */
  if( action_mask & (1 << 2) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_SMALL_L_ARENA];
#endif
#if defined(DISTRIBUTED)
    __nb_elt = 1;
#endif  /* defined(DISTRIBUTED) */
    if( ((descA.nt - 1) > k) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttmqr;
      {
        const int zttmqr_k = k;
        if( (zttmqr_k >= (0)) && (zttmqr_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttmqr_k;
          {
            const int zttmqr_m = m;
            if( (zttmqr_m >= ((zttmqr_k + 1))) && (zttmqr_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttmqr_m;
              {
                int zttmqr_n;
                for( zttmqr_n = (k + 1);zttmqr_n <= (descA.nt - 1); zttmqr_n++ ) {
                  if( (zttmqr_n >= ((zttmqr_k + 1))) && (zttmqr_n <= ((descA.nt - 1))) ) {
                    nc.locals[2].value = zttmqr_n;
                    const int zttmqr_p = zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[3].value = zttmqr_p;
                    const int zttmqr_nextp = zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[4].value = zttmqr_nextp;
                    const int zttmqr_prevp = zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[5].value = zttmqr_prevp;
                    const int zttmqr_prevm = zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[6].value = zttmqr_prevm;
                    const int zttmqr_type = zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[7].value = zttmqr_type;
                    const int zttmqr_type1 = zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[8].value = zttmqr_type1;
                    const int zttmqr_ip = zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[9].value = zttmqr_ip;
                    const int zttmqr_im = zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[10].value = zttmqr_im;
                    const int zttmqr_im1 = zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[11].value = zttmqr_im1;
#if defined(DISTRIBUTED)
                      rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttmqr_m, zttmqr_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                    if( NULL != eu ) {
                      char tmp[128], tmp1[128];
                      DEBUG(("thread %d release deps of L:%s to L:%s (from node %d to %d)\n", eu->eu_id,
                             dague_service_to_string(this_task, tmp, 128),
                             dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                    }
#endif
                      nc.priority = priority_of_zgetrf_param_zttmqr_as_expr_fct(this_task->dague_object, nc.locals);
                      if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 2, 1, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                        return;
      }
        }
          }
            }
              }
                }
                  }
    }
  }
  /* Flow of Data P */
  if( action_mask & (1 << 3) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_DEFAULT_ARENA];
#endif
#if defined(DISTRIBUTED)
    __nb_elt = 1;
#endif  /* defined(DISTRIBUTED) */
    if( ((descA.nt - 1) > k) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttmqr;
      {
        const int zttmqr_k = k;
        if( (zttmqr_k >= (0)) && (zttmqr_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttmqr_k;
          {
            const int zttmqr_m = m;
            if( (zttmqr_m >= ((zttmqr_k + 1))) && (zttmqr_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttmqr_m;
              {
                int zttmqr_n;
                for( zttmqr_n = (k + 1);zttmqr_n <= (descA.nt - 1); zttmqr_n++ ) {
                  if( (zttmqr_n >= ((zttmqr_k + 1))) && (zttmqr_n <= ((descA.nt - 1))) ) {
                    nc.locals[2].value = zttmqr_n;
                    const int zttmqr_p = zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[3].value = zttmqr_p;
                    const int zttmqr_nextp = zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[4].value = zttmqr_nextp;
                    const int zttmqr_prevp = zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[5].value = zttmqr_prevp;
                    const int zttmqr_prevm = zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[6].value = zttmqr_prevm;
                    const int zttmqr_type = zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[7].value = zttmqr_type;
                    const int zttmqr_type1 = zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[8].value = zttmqr_type1;
                    const int zttmqr_ip = zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[9].value = zttmqr_ip;
                    const int zttmqr_im = zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[10].value = zttmqr_im;
                    const int zttmqr_im1 = zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[11].value = zttmqr_im1;
#if defined(DISTRIBUTED)
                      rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttmqr_m, zttmqr_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                    if( NULL != eu ) {
                      char tmp[128], tmp1[128];
                      DEBUG(("thread %d release deps of P:%s to P:%s (from node %d to %d)\n", eu->eu_id,
                             dague_service_to_string(this_task, tmp, 128),
                             dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                    }
#endif
                      nc.priority = priority_of_zgetrf_param_zttmqr_as_expr_fct(this_task->dague_object, nc.locals);
                      if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 3, 1, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                        return;
      }
        }
          }
            }
              }
                }
                  }
    }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zgetrf_param_zttqrt(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, zttqrt_repo, zttqrt_hash(__dague_object, context->locals) );
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
  }
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zgetrf_param_zttqrt(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(zttqrt_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int m = context->locals[1].value;
    int p = context->locals[2].value;
    int nextp = context->locals[3].value;
    int prevp = context->locals[4].value;
    int prevm = context->locals[5].value;
    int type = context->locals[6].value;
    int ip = context->locals[7].value;
    int im = context->locals[8].value;
    (void)k; (void)m; (void)p; (void)nextp; (void)prevp; (void)prevm; (void)type; (void)ip; (void)im;

    if( (prevp == descA.mt) ) {
      data_repo_entry_used_once( eu, zgetrf_param_repo, context->data[0].data_repo->key );
      (void)AUNREF(context->data[0].data);
    } else {
      data_repo_entry_used_once( eu, zttqrt_repo, context->data[0].data_repo->key );
      (void)AUNREF(context->data[0].data);
    }
    if( ((type == 0) && (k != 0)) ) {
      data_repo_entry_used_once( eu, zttmqr_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    }
    else if( ((type != 0) && (prevm == descA.mt)) ) {
      data_repo_entry_used_once( eu, zgetrf_param_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    }
    else if( ((type != 0) && (prevm != descA.mt)) ) {
      data_repo_entry_used_once( eu, zttqrt_repo, context->data[1].data_repo->key );
      (void)AUNREF(context->data[1].data);
    }
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zgetrf_param_zttqrt(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = this_task->locals[0].value;
  int m = this_task->locals[1].value;
  int p = this_task->locals[2].value;
  int nextp = this_task->locals[3].value;
  int prevp = this_task->locals[4].value;
  int prevm = this_task->locals[5].value;
  int type = this_task->locals[6].value;
  int ip = this_task->locals[7].value;
  int im = this_task->locals[8].value;
  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;
  void *C = NULL; (void)C;
  dague_arena_chunk_t *gC = NULL; (void)gC;
  data_repo_entry_t *eC = NULL; (void)eC;
  void *L = NULL; (void)L;
  dague_arena_chunk_t *gL = NULL; (void)gL;
  data_repo_entry_t *eL = NULL; (void)eL;
  void *P = NULL; (void)P;
  dague_arena_chunk_t *gP = NULL; (void)gP;
  data_repo_entry_t *eP = NULL; (void)eP;

  /** Lookup the input data, and store them in the context if any */
  eA = this_task->data[0].data_repo;
  gA = this_task->data[0].data;
  if( NULL == gA ) {
  if( (prevp == descA.mt) ) {
      tass[0].value = k;
      tass[1].value = ip;
    eA = data_repo_lookup_entry( zgetrf_param_repo, zgetrf_param_hash( __dague_object, tass ));
    gA = eA->data[0];
  } else {
      tass[0].value = k;
      tass[1].value = prevp;
    eA = data_repo_lookup_entry( zttqrt_repo, zttqrt_hash( __dague_object, tass ));
    gA = eA->data[0];
  }
    this_task->data[0].data = gA;
    this_task->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
  eC = this_task->data[1].data_repo;
  gC = this_task->data[1].data;
  if( NULL == gC ) {
  if( ((type == 0) && (k == 0)) ) {
    gC = (dague_arena_chunk_t*) A(m, k);
  }
  else if( ((type == 0) && (k != 0)) ) {
      tass[0].value = (k - 1);
      tass[1].value = m;
      tass[2].value = k;
    eC = data_repo_lookup_entry( zttmqr_repo, zttmqr_hash( __dague_object, tass ));
    gC = eC->data[1];
  }
  else if( ((type != 0) && (prevm == descA.mt)) ) {
      tass[0].value = k;
      tass[1].value = im;
    eC = data_repo_lookup_entry( zgetrf_param_repo, zgetrf_param_hash( __dague_object, tass ));
    gC = eC->data[0];
  }
  else if( ((type != 0) && (prevm != descA.mt)) ) {
      tass[0].value = k;
      tass[1].value = prevm;
    eC = data_repo_lookup_entry( zttqrt_repo, zttqrt_hash( __dague_object, tass ));
    gC = eC->data[0];
  }
    this_task->data[1].data = gC;
    this_task->data[1].data_repo = eC;
  }
  C = ADATA(gC);
#if defined(DAGUE_SIM)
  if( (NULL != eC) && (eC->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eC->sim_exec_date;
#endif
  eL = this_task->data[2].data_repo;
  gL = this_task->data[2].data;
  if( NULL == gL ) {
  gL = (dague_arena_chunk_t*) L(m, k);
    this_task->data[2].data = gL;
    this_task->data[2].data_repo = eL;
  }
  L = ADATA(gL);
#if defined(DAGUE_SIM)
  if( (NULL != eL) && (eL->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eL->sim_exec_date;
#endif
  eP = this_task->data[3].data_repo;
  gP = this_task->data[3].data;
  if( NULL == gP ) {
  gP = (dague_arena_chunk_t*) IPIV(m, k);
    this_task->data[3].data = gP;
    this_task->data[3].data_repo = eP;
  }
  P = ADATA(gP);
#if defined(DAGUE_SIM)
  if( (NULL != eP) && (eP->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eP->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( this_task->function->sim_cost_fct != NULL ) {
    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);
  } else {
    this_task->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < this_task->sim_exec_date )
    context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
  cache_buf_referenced(context->closest_cache, C);
  cache_buf_referenced(context->closest_cache, L);
  cache_buf_referenced(context->closest_cache, P);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                                  zttqrt BODY                                  *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*this_task->function->function_id, zttqrt_hash( __dague_object, this_task->locals), __dague_object->super.A, ((dague_ddesc_t*)(__dague_object->super.A))->data_key((dague_ddesc_t*)__dague_object->super.A, m, k) );
#line 245 "zgetrf_param.jdf"
  DRYRUN(
         if ( type == 0 ) {
           int tempmm = ((m)==(descA.mt-1)) ? (descA.m-(m*descA.mb)) : (descA.mb);
           int tempkn = ((k)==(descA.nt-1)) ? (descA.n-(k*descA.nb)) : (descA.nb);
           int ldak   = descA.mb; /*((k+(Ai/descA.mb))<Alm1) ? (descA.mb) : (Alm%descA.mb);*/
           int ldam   = descA.mb; /*((m+(Ai/descA.mb))<Alm1) ? (descA.mb) : (Alm%descA.mb);*/

           int iinfo;
           PLASMA_Complex64_t *work = dague_private_memory_pop(work_pool);

           memset(P, 0, min(tempkn, tempmm) * sizeof(int) );
           CORE_ztstrf(tempmm, tempkn, ib, descL.nb,
                       A  /* A(k,k) */, ldak,
                       C  /* A(m,k) */, ldam,
                       L  /* L(m,k) */, descL.mb,
                       P /* IPIV(m,k) */,
                       work, descL.nb, &iinfo );
           dague_private_memory_push(work_pool, work);

         if ( (iinfo != 0) && (m == descA.mt-1) ) {
             *INFO = k * descA.mb + iinfo; /* Should return if enter here */
             fprintf(stderr, "ztstrf(%d, %d) failed => %d\n", m, k, *INFO );
         }
         } else {
           void *p_elem_A = dague_private_memory_pop( p_tau  );
           void *p_elem_B = dague_private_memory_pop( p_work );

           int tempmm = ((m)==((descA.mt)-1)) ? ((descA.m)-(m*(descA.mb))) : (descA.mb);
           int tempkn = ((k)==((descA.nt)-1)) ? ((descA.n)-(k*(descA.nb))) : (descA.nb);
           int ldam = BLKLDD( descA, m );

           CORE_zttqrt(
                     tempmm, tempkn, ib,
                     A /* A(p, k) */, descA.mb,
                     C /* A(m, k) */, ldam,
                     L  /* T(m, k) */, descL.mb,
                     p_elem_A, p_elem_B );

           dague_private_memory_push( p_tau , p_elem_A );
           dague_private_memory_push( p_work, p_elem_B );
         }

         );

#if defined(DAGUE_SIM)
  ((PLASMA_Complex64_t*)C)[0] = (PLASMA_Complex64_t)(this_task->sim_exec_date);
  if ( ( ( nextp == descA.mt ) & (p == k) ) )
    ((PLASMA_Complex64_t*)H)[0] = (PLASMA_Complex64_t)(this_task->sim_exec_date);
#endif
  printlog("thread %d CORE_zttqrt(%d, %d)\n"
           "\t(tempmm, tempkn, ib, A(%d,%d)[%p], A.mb, A(%d,%d)[%p], ldam, T(%d,%d)[%p], descL.mb, p_elem_A, p_elem_B)\n",
           context->eu_id, k, m, p, k, V, m, k, C, m, k, L);


#line 5124 "zgetrf_param.c"
/*--------------------------------------------------------------------------------*
 *                              END OF zttqrt BODY                              *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zgetrf_param_zttqrt(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  (void)context; (void)__dague_object;
  int k = this_task->locals[0].value;
  int m = this_task->locals[1].value;
  int p = this_task->locals[2].value;
  int nextp = this_task->locals[3].value;
  int prevp = this_task->locals[4].value;
  int prevm = this_task->locals[5].value;
  int type = this_task->locals[6].value;
  int ip = this_task->locals[7].value;
  int im = this_task->locals[8].value;
  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;

  TAKE_TIME(context,2*this_task->function->function_id+1, zttqrt_hash( __dague_object, this_task->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( ADATA(this_task->data[1].data) != A(m, k) ) {
    int __arena_index = DAGUE_zgetrf_param_DEFAULT_ARENA;
    int __dtt_nb = 1;
    assert( (__arena_index>=0) && (__arena_index < __dague_object->super.arenas_size) );
    assert( __dtt_nb >= 0 );
    dague_remote_dep_memcpy( context, this_task->dague_object, A(m, k), this_task->data[1].data, 
                             __dague_object->super.arenas[__arena_index]->opaque_dtt,
                             __dtt_nb );
  }
  if( ADATA(this_task->data[2].data) != L(m, k) ) {
    int __arena_index = DAGUE_zgetrf_param_SMALL_L_ARENA;
    int __dtt_nb = 1;
    assert( (__arena_index>=0) && (__arena_index < __dague_object->super.arenas_size) );
    assert( __dtt_nb >= 0 );
    dague_remote_dep_memcpy( context, this_task->dague_object, L(m, k), this_task->data[2].data, 
                             __dague_object->super.arenas[__arena_index]->opaque_dtt,
                             __dtt_nb );
  }
  if( ADATA(this_task->data[3].data) != IPIV(m, k) ) {
    int __arena_index = DAGUE_zgetrf_param_DEFAULT_ARENA;
    int __dtt_nb = 1;
    assert( (__arena_index>=0) && (__arena_index < __dague_object->super.arenas_size) );
    assert( __dtt_nb >= 0 );
    dague_remote_dep_memcpy( context, this_task->dague_object, IPIV(m, k), this_task->data[3].data, 
                             __dague_object->super.arenas[__arena_index]->opaque_dtt,
                             __dtt_nb );
  }
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(this_task, context->eu_id, zttqrt_hash(__dague_object, this_task->locals));
  release_deps_of_zgetrf_param_zttqrt(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      DAGUE_ACTION_DEPS_MASK,
      NULL);
  return 0;
}

#if defined(DAGUE_SIM)
static int simulation_cost_of_zgetrf_param_zttqrt(const dague_execution_context_t *this_task)
{
  const dague_object_t *__dague_object = (const dague_object_t*)this_task->dague_object;
  int k = this_task->locals[0].value;
  int m = this_task->locals[1].value;
  int p = this_task->locals[2].value;
  int nextp = this_task->locals[3].value;
  int prevp = this_task->locals[4].value;
  int prevm = this_task->locals[5].value;
  int type = this_task->locals[6].value;
  int ip = this_task->locals[7].value;
  int im = this_task->locals[8].value;
  (void)__dague_object;
  (void)k;  (void)m;  (void)p;  (void)nextp;  (void)prevp;  (void)prevm;  (void)type;  (void)ip;  (void)im;
  return zgetrf_param_inline_c_expr14_line_218((const dague_object_t*)__dague_object, this_task->locals);
}
#endif

static int zgetrf_param_zttqrt_internal_init(__dague_zgetrf_param_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, m, p, nextp, prevp, prevm, type, ip, im;
  int32_t  k_min = 0x7fffffff, m_min = 0x7fffffff;
  int32_t  k_max = 0, m_max = 0;
  (void)__dague_object; (void)__foundone;
  int32_t k_start, k_end;  int32_t m_start, m_end;  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
      k++) {
    assignments[0].value = k;
    for(m = (k + 1);
        m <= (descA.mt - 1);
        m++) {
      assignments[1].value = m;
      p = zgetrf_param_inline_c_expr21_line_210((const dague_object_t*)__dague_object, assignments);
      assignments[2].value = p;
      nextp = zgetrf_param_inline_c_expr20_line_211((const dague_object_t*)__dague_object, assignments);
      assignments[3].value = nextp;
      prevp = zgetrf_param_inline_c_expr19_line_212((const dague_object_t*)__dague_object, assignments);
      assignments[4].value = prevp;
      prevm = zgetrf_param_inline_c_expr18_line_213((const dague_object_t*)__dague_object, assignments);
      assignments[5].value = prevm;
      type = zgetrf_param_inline_c_expr17_line_214((const dague_object_t*)__dague_object, assignments);
      assignments[6].value = type;
      ip = zgetrf_param_inline_c_expr16_line_215((const dague_object_t*)__dague_object, assignments);
      assignments[7].value = ip;
      im = zgetrf_param_inline_c_expr15_line_216((const dague_object_t*)__dague_object, assignments);
      assignments[8].value = im;
      if( !zttqrt_pred(k, m, p, nextp, prevp, prevm, type, ip, im) ) continue;
      nb_tasks++;
      k_max = dague_imax(k_max, k);
      k_min = dague_imin(k_min, k);
      m_max = dague_imax(m_max, m);
      m_min = dague_imin(m_min, m);
    }
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  dep = NULL;
  k_start = 0;
  k_end = ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
  for(k = dague_imax(k_start, k_min); k <= dague_imin(k_end, k_max); k++) {
    assignments[0].value = k;
    m_start = (k + 1);
    m_end = (descA.mt - 1);
    for(m = dague_imax(m_start, m_min); m <= dague_imin(m_end, m_max); m++) {
      assignments[1].value = m;
      p = zgetrf_param_inline_c_expr21_line_210((const dague_object_t*)__dague_object, assignments);
      assignments[2].value = p;
      nextp = zgetrf_param_inline_c_expr20_line_211((const dague_object_t*)__dague_object, assignments);
      assignments[3].value = nextp;
      prevp = zgetrf_param_inline_c_expr19_line_212((const dague_object_t*)__dague_object, assignments);
      assignments[4].value = prevp;
      prevm = zgetrf_param_inline_c_expr18_line_213((const dague_object_t*)__dague_object, assignments);
      assignments[5].value = prevm;
      type = zgetrf_param_inline_c_expr17_line_214((const dague_object_t*)__dague_object, assignments);
      assignments[6].value = type;
      ip = zgetrf_param_inline_c_expr16_line_215((const dague_object_t*)__dague_object, assignments);
      assignments[7].value = ip;
      __foundone = 0;
      im = zgetrf_param_inline_c_expr15_line_216((const dague_object_t*)__dague_object, assignments);
      assignments[8].value = im;
      if( zttqrt_pred(k, m, p, nextp, prevp, prevm, type, ip, im) ) {
        /* We did find one! Allocate the dependencies array. */
      if( dep == NULL ) {
        ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zgetrf_param_zttqrt_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
      }
      if( dep->u.next[k-k_min] == NULL ) {
        ALLOCATE_DEP_TRACKING(dep->u.next[k-k_min], m_min, m_max, "m", &symb_zgetrf_param_zttqrt_m, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
      }
      }
    }
  }
  (void)k_start; (void)k_end;  (void)m_start; (void)m_end;  __dague_object->super.super.dependencies_array[2] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static const dague_function_t zgetrf_param_zttqrt = {
  .name = "zttqrt",
  .deps = 2,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 2,
  .dependencies_goal = 0xf,
  .nb_parameters = 2,
  .nb_definitions = 9,
  .params = { &symb_zgetrf_param_zttqrt_k, &symb_zgetrf_param_zttqrt_m },
  .locals = { &symb_zgetrf_param_zttqrt_k, &symb_zgetrf_param_zttqrt_m, &symb_zgetrf_param_zttqrt_p, &symb_zgetrf_param_zttqrt_nextp, &symb_zgetrf_param_zttqrt_prevp, &symb_zgetrf_param_zttqrt_prevm, &symb_zgetrf_param_zttqrt_type, &symb_zgetrf_param_zttqrt_ip, &symb_zgetrf_param_zttqrt_im },
  .pred = &pred_of_zgetrf_param_zttqrt_as_expr,
  .priority = &priority_of_zgetrf_param_zttqrt_as_expr,
  .in = { &flow_of_zgetrf_param_zttqrt_for_A, &flow_of_zgetrf_param_zttqrt_for_C, &flow_of_zgetrf_param_zttqrt_for_L, &flow_of_zgetrf_param_zttqrt_for_P },
  .out = { &flow_of_zgetrf_param_zttqrt_for_A, &flow_of_zgetrf_param_zttqrt_for_C, &flow_of_zgetrf_param_zttqrt_for_L, &flow_of_zgetrf_param_zttqrt_for_P },
  .iterate_successors = iterate_successors_of_zgetrf_param_zttqrt,
  .release_deps = release_deps_of_zgetrf_param_zttqrt,
  .hook = hook_of_zgetrf_param_zttqrt,
  .complete_execution = complete_hook_of_zgetrf_param_zttqrt,
#if defined(DAGUE_SIM)
  .sim_cost_fct = simulation_cost_of_zgetrf_param_zttqrt,
#endif
  .key = (dague_functionkey_fn_t*)zttqrt_hash,
};


/**********************************************************************************
 *                                zttqrt_out_A1                                  *
 **********************************************************************************/

static inline int minexpr_of_symb_zgetrf_param_zttqrt_out_A1_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgetrf_param_zttqrt_out_A1_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zttqrt_out_A1_k_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zttqrt_out_A1_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return ((descA.mt <= descA.nt) ? (descA.mt - 2) : (descA.nt - 1));
}
static const expr_t maxexpr_of_symb_zgetrf_param_zttqrt_out_A1_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zttqrt_out_A1_k_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_out_A1_k = {.min = &minexpr_of_symb_zgetrf_param_zttqrt_out_A1_k, .max = &maxexpr_of_symb_zgetrf_param_zttqrt_out_A1_k,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zttqrt_out_A1_prevp_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr22_line_196((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zttqrt_out_A1_prevp = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zttqrt_out_A1_prevp_fct
};
static const symbol_t symb_zgetrf_param_zttqrt_out_A1_prevp = {.min = &expr_of_symb_zgetrf_param_zttqrt_out_A1_prevp, .max = &expr_of_symb_zgetrf_param_zttqrt_out_A1_prevp,  .flags = 0x0};

static inline int pred_of_zgetrf_param_zttqrt_out_A1_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int prevp = assignments[1].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)prevp;
  /* Compute Predicate */
  return zttqrt_out_A1_pred(k, prevp);
}
static const expr_t pred_of_zgetrf_param_zttqrt_out_A1_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zgetrf_param_zttqrt_out_A1_as_expr_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep1_atline_200_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep1_atline_200 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep1_atline_200_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep1_atline_200_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int prevp = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return prevp;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep1_atline_200 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep1_atline_200_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep1_atline_200 = {
  .cond = NULL,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_UPPER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep1_atline_200,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep1_atline_200
  }
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep2_atline_201_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep2_atline_201 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep2_atline_201_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep2_atline_201_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep2_atline_201 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep2_atline_201_fct
};
static const dep_t flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep2_atline_201 = {
  .cond = NULL,
  .dague = &zgetrf_param_A,
  .datatype = { .index = DAGUE_zgetrf_param_UPPER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep2_atline_201,
    &expr_of_p2_for_flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep2_atline_201
  }
};
static const dague_flow_t flow_of_zgetrf_param_zttqrt_out_A1_for_A = {
  .name = "A",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 0,
  .dep_in  = { &flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep1_atline_200 },
  .dep_out = { &flow_of_zgetrf_param_zttqrt_out_A1_for_A_dep2_atline_201 }
};

static void
iterate_successors_of_zgetrf_param_zttqrt_out_A1(dague_execution_unit_t *eu, dague_execution_context_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)this_task->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int __nb_elt = -1;
  int rank_src = 0, rank_dst = 0;
  int k = this_task->locals[0].value;
  int prevp = this_task->locals[1].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object; (void)__nb_elt;
  (void)k;  (void)prevp;
  nc.dague_object = this_task->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, k, k);
#endif
  /* Flow of data A has only OUTPUT dependencies to Memory */
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zgetrf_param_zttqrt_out_A1(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, zttqrt_out_A1_repo, zttqrt_out_A1_hash(__dague_object, context->locals) );
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
  }
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zgetrf_param_zttqrt_out_A1(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(zttqrt_out_A1_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int prevp = context->locals[1].value;
    (void)k; (void)prevp;

    data_repo_entry_used_once( eu, zttqrt_repo, context->data[0].data_repo->key );
    (void)AUNREF(context->data[0].data);
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zgetrf_param_zttqrt_out_A1(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = this_task->locals[0].value;
  int prevp = this_task->locals[1].value;
  (void)k;  (void)prevp;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;

  /** Lookup the input data, and store them in the context if any */
  eA = this_task->data[0].data_repo;
  gA = this_task->data[0].data;
  if( NULL == gA ) {
  tass[0].value = k;
  tass[1].value = prevp;
  eA = data_repo_lookup_entry( zttqrt_repo, zttqrt_hash( __dague_object, tass ));
  gA = eA->data[0];
    this_task->data[0].data = gA;
    this_task->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( this_task->function->sim_cost_fct != NULL ) {
    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);
  } else {
    this_task->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < this_task->sim_exec_date )
    context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                              zttqrt_out_A1 BODY                              *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*this_task->function->function_id, zttqrt_out_A1_hash( __dague_object, this_task->locals), __dague_object->super.A, ((dague_ddesc_t*)(__dague_object->super.A))->data_key((dague_ddesc_t*)__dague_object->super.A, k, k) );
#line 202 "zgetrf_param.jdf"
/* nothing */

#line 5592 "zgetrf_param.c"
/*--------------------------------------------------------------------------------*
 *                          END OF zttqrt_out_A1 BODY                            *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zgetrf_param_zttqrt_out_A1(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  (void)context; (void)__dague_object;
  int k = this_task->locals[0].value;
  int prevp = this_task->locals[1].value;
  (void)k;  (void)prevp;

  TAKE_TIME(context,2*this_task->function->function_id+1, zttqrt_out_A1_hash( __dague_object, this_task->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( ADATA(this_task->data[0].data) != A(k, k) ) {
    int __arena_index = DAGUE_zgetrf_param_UPPER_TILE_ARENA;
    int __dtt_nb = 1;
    assert( (__arena_index>=0) && (__arena_index < __dague_object->super.arenas_size) );
    assert( __dtt_nb >= 0 );
    dague_remote_dep_memcpy( context, this_task->dague_object, A(k, k), this_task->data[0].data, 
                             __dague_object->super.arenas[__arena_index]->opaque_dtt,
                             __dtt_nb );
  }
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(this_task, context->eu_id, zttqrt_out_A1_hash(__dague_object, this_task->locals));
  release_deps_of_zgetrf_param_zttqrt_out_A1(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      DAGUE_ACTION_DEPS_MASK,
      NULL);
  return 0;
}

static int zgetrf_param_zttqrt_out_A1_internal_init(__dague_zgetrf_param_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, prevp;
  int32_t  k_min = 0x7fffffff;
  int32_t  k_max = 0;
  (void)__dague_object; (void)__foundone;
  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= ((descA.mt <= descA.nt) ? (descA.mt - 2) : (descA.nt - 1));
      k++) {
    assignments[0].value = k;
    prevp = zgetrf_param_inline_c_expr22_line_196((const dague_object_t*)__dague_object, assignments);
    assignments[1].value = prevp;
    if( !zttqrt_out_A1_pred(k, prevp) ) continue;
    nb_tasks++;
    k_max = dague_imax(k_max, k);
    k_min = dague_imin(k_min, k);
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  if( 0 != nb_tasks ) {
    ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zgetrf_param_zttqrt_out_A1_k, NULL, DAGUE_DEPENDENCIES_FLAG_FINAL);
  }
  __dague_object->super.super.dependencies_array[3] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static const dague_function_t zgetrf_param_zttqrt_out_A1 = {
  .name = "zttqrt_out_A1",
  .deps = 3,
  .flags = 0x0,
  .function_id = 3,
  .dependencies_goal = 0x1,
  .nb_parameters = 1,
  .nb_definitions = 2,
  .params = { &symb_zgetrf_param_zttqrt_out_A1_k },
  .locals = { &symb_zgetrf_param_zttqrt_out_A1_k, &symb_zgetrf_param_zttqrt_out_A1_prevp },
  .pred = &pred_of_zgetrf_param_zttqrt_out_A1_as_expr,
  .priority = NULL,
  .in = { &flow_of_zgetrf_param_zttqrt_out_A1_for_A },
  .out = { &flow_of_zgetrf_param_zttqrt_out_A1_for_A },
  .iterate_successors = iterate_successors_of_zgetrf_param_zttqrt_out_A1,
  .release_deps = release_deps_of_zgetrf_param_zttqrt_out_A1,
  .hook = hook_of_zgetrf_param_zttqrt_out_A1,
  .complete_execution = complete_hook_of_zgetrf_param_zttqrt_out_A1,
#if defined(DAGUE_SIM)
  .sim_cost_fct = NULL,
#endif
  .key = (dague_functionkey_fn_t*)zttqrt_out_A1_hash,
};


/**********************************************************************************
 *                                    zgessm                                    *
 **********************************************************************************/

static inline int minexpr_of_symb_zgetrf_param_zgessm_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgetrf_param_zgessm_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zgessm_k_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zgessm_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
}
static const expr_t maxexpr_of_symb_zgetrf_param_zgessm_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zgessm_k_fct
};
static const symbol_t symb_zgetrf_param_zgessm_k = {.min = &minexpr_of_symb_zgetrf_param_zgessm_k, .max = &maxexpr_of_symb_zgetrf_param_zgessm_k,  .flags = 0x0};

static inline int minexpr_of_symb_zgetrf_param_zgessm_i_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgetrf_param_zgessm_i = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zgessm_i_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zgessm_i_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr26_line_151((const dague_object_t*)__dague_object, assignments);
}
static const expr_t maxexpr_of_symb_zgetrf_param_zgessm_i = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zgessm_i_fct
};
static const symbol_t symb_zgetrf_param_zgessm_i = {.min = &minexpr_of_symb_zgetrf_param_zgessm_i, .max = &maxexpr_of_symb_zgetrf_param_zgessm_i,  .flags = 0x0};

static inline int minexpr_of_symb_zgetrf_param_zgessm_n_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t minexpr_of_symb_zgetrf_param_zgessm_n = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zgessm_n_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zgessm_n_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return (descA.nt - 1);
}
static const expr_t maxexpr_of_symb_zgetrf_param_zgessm_n = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zgessm_n_fct
};
static const symbol_t symb_zgetrf_param_zgessm_n = {.min = &minexpr_of_symb_zgetrf_param_zgessm_n, .max = &maxexpr_of_symb_zgetrf_param_zgessm_n,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zgessm_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr25_line_153((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zgessm_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zgessm_m_fct
};
static const symbol_t symb_zgetrf_param_zgessm_m = {.min = &expr_of_symb_zgetrf_param_zgessm_m, .max = &expr_of_symb_zgetrf_param_zgessm_m,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zgessm_nextm_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr24_line_154((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zgessm_nextm = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zgessm_nextm_fct
};
static const symbol_t symb_zgetrf_param_zgessm_nextm = {.min = &expr_of_symb_zgetrf_param_zgessm_nextm, .max = &expr_of_symb_zgetrf_param_zgessm_nextm,  .flags = 0x0};

static inline int pred_of_zgetrf_param_zgessm_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int i = assignments[1].value;
  int n = assignments[2].value;
  int m = assignments[3].value;
  int nextm = assignments[4].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)i;
  (void)n;
  (void)m;
  (void)nextm;
  /* Compute Predicate */
  return zgessm_pred(k, i, n, m, nextm);
}
static const expr_t pred_of_zgetrf_param_zgessm_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zgetrf_param_zgessm_as_expr_fct
};
static inline int priority_of_zgetrf_param_zgessm_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr23_line_172((const dague_object_t*)__dague_object, assignments);
}
static const expr_t priority_of_zgetrf_param_zgessm_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = priority_of_zgetrf_param_zgessm_as_expr_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_A_dep1_atline_164_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_A_dep1_atline_164 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_A_dep1_atline_164_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_A_dep1_atline_164_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int i = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return i;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_A_dep1_atline_164 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_A_dep1_atline_164_fct
};
static const dep_t flow_of_zgetrf_param_zgessm_for_A_dep1_atline_164 = {
  .cond = NULL,
  .dague = &zgetrf_param_zgetrf_param_out,
  .flow = &flow_of_zgetrf_param_zgetrf_param_out_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_LOWER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_A_dep1_atline_164,
    &expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_A_dep1_atline_164
  }
};
static const dague_flow_t flow_of_zgetrf_param_zgessm_for_A = {
  .name = "A",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 0,
  .dep_in  = { &flow_of_zgetrf_param_zgessm_for_A_dep1_atline_164 },
  .dep_out = { NULL }
};

static inline int expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_P_dep1_atline_165_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_P_dep1_atline_165 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_P_dep1_atline_165_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_P_dep1_atline_165_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int i = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return i;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_P_dep1_atline_165 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_P_dep1_atline_165_fct
};
static const dep_t flow_of_zgetrf_param_zgessm_for_P_dep1_atline_165 = {
  .cond = NULL,
  .dague = &zgetrf_param_zgetrf_param,
  .flow = &flow_of_zgetrf_param_zgetrf_param_for_P,
  .datatype = { .index = DAGUE_zgetrf_param_PIVOT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_P_dep1_atline_165,
    &expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_P_dep1_atline_165
  }
};
static const dague_flow_t flow_of_zgetrf_param_zgessm_for_P = {
  .name = "P",
  .sym_type = SYM_IN,
  .access_type = ACCESS_READ,
  .flow_index = 1,
  .dep_in  = { &flow_of_zgetrf_param_zgessm_for_P_dep1_atline_165 },
  .dep_out = { NULL }
};

static inline int expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (0 == k);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167_fct
};
static const dep_t flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167,
  .dague = &zgetrf_param_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167,
    &expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k > 0);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k - 1);
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168_fct
};
static const dep_t flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168,
    &expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168,
    &expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k == (descA.mt - 1));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169_fct
};
static const dep_t flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169,
  .dague = &zgetrf_param_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169,
    &expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int nextm = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return ((k < (descA.mt - 1)) && (nextm != descA.mt));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int nextm = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return nextm;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170_fct
};
static const dep_t flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_V,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170,
    &expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170,
    &expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int nextm = assignments[4].value;

  (void)__dague_object; (void)assignments;
  return ((k < (descA.mt - 1)) && (nextm == descA.mt));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int n = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return n;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172_fct
};
static const dep_t flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172,
    &expr_of_p2_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172,
    &expr_of_p3_for_flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172
  }
};
static const dague_flow_t flow_of_zgetrf_param_zgessm_for_C = {
  .name = "C",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 2,
  .dep_in  = { &flow_of_zgetrf_param_zgessm_for_C_dep1_atline_167, &flow_of_zgetrf_param_zgessm_for_C_dep2_atline_168 },
  .dep_out = { &flow_of_zgetrf_param_zgessm_for_C_dep3_atline_169, &flow_of_zgetrf_param_zgessm_for_C_dep4_atline_170, &flow_of_zgetrf_param_zgessm_for_C_dep5_atline_172 }
};

static void
iterate_successors_of_zgetrf_param_zgessm(dague_execution_unit_t *eu, dague_execution_context_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)this_task->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int __nb_elt = -1;
  int rank_src = 0, rank_dst = 0;
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int n = this_task->locals[2].value;
  int m = this_task->locals[3].value;
  int nextm = this_task->locals[4].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object; (void)__nb_elt;
  (void)k;  (void)i;  (void)n;  (void)m;  (void)nextm;
  nc.dague_object = this_task->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, m, n);
#endif
  /* Flow of data A has only IN dependencies */
  /* Flow of data P has only IN dependencies */
  /* Flow of Data C */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_DEFAULT_ARENA];
#endif
#if defined(DISTRIBUTED)
    __nb_elt = 1;
#endif  /* defined(DISTRIBUTED) */
    if( ((k < (descA.mt - 1)) && (nextm != descA.mt)) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttmqr;
      {
        const int zttmqr_k = k;
        if( (zttmqr_k >= (0)) && (zttmqr_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttmqr_k;
          {
            const int zttmqr_m = nextm;
            if( (zttmqr_m >= ((zttmqr_k + 1))) && (zttmqr_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttmqr_m;
              {
                const int zttmqr_n = n;
                if( (zttmqr_n >= ((zttmqr_k + 1))) && (zttmqr_n <= ((descA.nt - 1))) ) {
                  nc.locals[2].value = zttmqr_n;
                  const int zttmqr_p = zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[3].value = zttmqr_p;
                  const int zttmqr_nextp = zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[4].value = zttmqr_nextp;
                  const int zttmqr_prevp = zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[5].value = zttmqr_prevp;
                  const int zttmqr_prevm = zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[6].value = zttmqr_prevm;
                  const int zttmqr_type = zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[7].value = zttmqr_type;
                  const int zttmqr_type1 = zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[8].value = zttmqr_type1;
                  const int zttmqr_ip = zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[9].value = zttmqr_ip;
                  const int zttmqr_im = zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[10].value = zttmqr_im;
                  const int zttmqr_im1 = zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[11].value = zttmqr_im1;
#if defined(DISTRIBUTED)
                    rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttmqr_m, zttmqr_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                  if( NULL != eu ) {
                    char tmp[128], tmp1[128];
                    DEBUG(("thread %d release deps of C:%s to V:%s (from node %d to %d)\n", eu->eu_id,
                           dague_service_to_string(this_task, tmp, 128),
                           dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                  }
#endif
                    nc.priority = priority_of_zgetrf_param_zttmqr_as_expr_fct(this_task->dague_object, nc.locals);
                    if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 1, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                      return;
      }
        }
          }
            }
              }
                }
    }
    if( ((k < (descA.mt - 1)) && (nextm == descA.mt)) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttmqr;
      {
        const int zttmqr_k = k;
        if( (zttmqr_k >= (0)) && (zttmqr_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttmqr_k;
          {
            const int zttmqr_m = m;
            if( (zttmqr_m >= ((zttmqr_k + 1))) && (zttmqr_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttmqr_m;
              {
                const int zttmqr_n = n;
                if( (zttmqr_n >= ((zttmqr_k + 1))) && (zttmqr_n <= ((descA.nt - 1))) ) {
                  nc.locals[2].value = zttmqr_n;
                  const int zttmqr_p = zgetrf_param_inline_c_expr11_line_320((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[3].value = zttmqr_p;
                  const int zttmqr_nextp = zgetrf_param_inline_c_expr10_line_321((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[4].value = zttmqr_nextp;
                  const int zttmqr_prevp = zgetrf_param_inline_c_expr9_line_322((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[5].value = zttmqr_prevp;
                  const int zttmqr_prevm = zgetrf_param_inline_c_expr8_line_323((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[6].value = zttmqr_prevm;
                  const int zttmqr_type = zgetrf_param_inline_c_expr7_line_324((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[7].value = zttmqr_type;
                  const int zttmqr_type1 = zgetrf_param_inline_c_expr6_line_325((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[8].value = zttmqr_type1;
                  const int zttmqr_ip = zgetrf_param_inline_c_expr5_line_326((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[9].value = zttmqr_ip;
                  const int zttmqr_im = zgetrf_param_inline_c_expr4_line_327((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[10].value = zttmqr_im;
                  const int zttmqr_im1 = zgetrf_param_inline_c_expr3_line_328((const dague_object_t*)__dague_object, nc.locals);
                  nc.locals[11].value = zttmqr_im1;
#if defined(DISTRIBUTED)
                    rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttmqr_m, zttmqr_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                  if( NULL != eu ) {
                    char tmp[128], tmp1[128];
                    DEBUG(("thread %d release deps of C:%s to C:%s (from node %d to %d)\n", eu->eu_id,
                           dague_service_to_string(this_task, tmp, 128),
                           dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                  }
#endif
                    nc.priority = priority_of_zgetrf_param_zttmqr_as_expr_fct(this_task->dague_object, nc.locals);
                    if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 2, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                      return;
      }
        }
          }
            }
              }
                }
    }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zgetrf_param_zgessm(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, zgessm_repo, zgessm_hash(__dague_object, context->locals) );
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
  }
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zgetrf_param_zgessm(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(zgessm_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int i = context->locals[1].value;
    int n = context->locals[2].value;
    int m = context->locals[3].value;
    int nextm = context->locals[4].value;
    (void)k; (void)i; (void)n; (void)m; (void)nextm;

    data_repo_entry_used_once( eu, zgetrf_param_out_repo, context->data[0].data_repo->key );
    (void)AUNREF(context->data[0].data);
    data_repo_entry_used_once( eu, zgetrf_param_repo, context->data[1].data_repo->key );
    (void)AUNREF(context->data[1].data);
    if( (k > 0) ) {
      data_repo_entry_used_once( eu, zttmqr_repo, context->data[2].data_repo->key );
      (void)AUNREF(context->data[2].data);
    }
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zgetrf_param_zgessm(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int n = this_task->locals[2].value;
  int m = this_task->locals[3].value;
  int nextm = this_task->locals[4].value;
  (void)k;  (void)i;  (void)n;  (void)m;  (void)nextm;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;
  void *P = NULL; (void)P;
  dague_arena_chunk_t *gP = NULL; (void)gP;
  data_repo_entry_t *eP = NULL; (void)eP;
  void *C = NULL; (void)C;
  dague_arena_chunk_t *gC = NULL; (void)gC;
  data_repo_entry_t *eC = NULL; (void)eC;

  /** Lookup the input data, and store them in the context if any */
  eA = this_task->data[0].data_repo;
  gA = this_task->data[0].data;
  if( NULL == gA ) {
  tass[0].value = k;
  tass[1].value = i;
  eA = data_repo_lookup_entry( zgetrf_param_out_repo, zgetrf_param_out_hash( __dague_object, tass ));
  gA = eA->data[0];
    this_task->data[0].data = gA;
    this_task->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
  eP = this_task->data[1].data_repo;
  gP = this_task->data[1].data;
  if( NULL == gP ) {
  tass[0].value = k;
  tass[1].value = i;
  eP = data_repo_lookup_entry( zgetrf_param_repo, zgetrf_param_hash( __dague_object, tass ));
  gP = eP->data[1];
    this_task->data[1].data = gP;
    this_task->data[1].data_repo = eP;
  }
  P = ADATA(gP);
#if defined(DAGUE_SIM)
  if( (NULL != eP) && (eP->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eP->sim_exec_date;
#endif
  eC = this_task->data[2].data_repo;
  gC = this_task->data[2].data;
  if( NULL == gC ) {
  if( (0 == k) ) {
    gC = (dague_arena_chunk_t*) A(m, n);
  }
  else if( (k > 0) ) {
      tass[0].value = (k - 1);
      tass[1].value = m;
      tass[2].value = n;
    eC = data_repo_lookup_entry( zttmqr_repo, zttmqr_hash( __dague_object, tass ));
    gC = eC->data[1];
  }
    this_task->data[2].data = gC;
    this_task->data[2].data_repo = eC;
  }
  C = ADATA(gC);
#if defined(DAGUE_SIM)
  if( (NULL != eC) && (eC->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eC->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( this_task->function->sim_cost_fct != NULL ) {
    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);
  } else {
    this_task->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < this_task->sim_exec_date )
    context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
  cache_buf_referenced(context->closest_cache, P);
  cache_buf_referenced(context->closest_cache, C);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                                  zgessm BODY                                  *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*this_task->function->function_id, zgessm_hash( __dague_object, this_task->locals), __dague_object->super.A, ((dague_ddesc_t*)(__dague_object->super.A))->data_key((dague_ddesc_t*)__dague_object->super.A, m, n) );
#line 174 "zgetrf_param.jdf"

  DRYRUN(
         int tempnn = ((n)==(descA.nt-1)) ? (descA.n-(n*descA.nb)) : (descA.nb);
         int tempkm = ((k)==(descA.mt-1)) ? (descA.m-(k*descA.mb)) : (descA.mb);
         int ldak   = descA.mb; /*((k+(Ai/descA.mb))<Alm1) ? (descA.mb) : (Alm%descA.mb);*/

         CORE_zgessm(tempkm, tempnn, tempkm, ib,
                     P /* IPIV(k,k) */,
                     A /* A(k,k) */, ldak,
                     C /* A(k,n) */, ldak );
         );
   printlog("thread %d   CORE_zgessm(%d, %d)\n"
            "\t(tempkm, tempnn, tempkm, ib, IPIV(%d,%d)[%p], \n"
            "\tA(%d,%d)[%p], ldak, A(%d,%d)[%p], ldak)\n",
            context->eu_id, k, n, k, k, P, k, k, A, k, n, C);


#line 6527 "zgetrf_param.c"
/*--------------------------------------------------------------------------------*
 *                              END OF zgessm BODY                              *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zgetrf_param_zgessm(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  (void)context; (void)__dague_object;
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int n = this_task->locals[2].value;
  int m = this_task->locals[3].value;
  int nextm = this_task->locals[4].value;
  (void)k;  (void)i;  (void)n;  (void)m;  (void)nextm;

  TAKE_TIME(context,2*this_task->function->function_id+1, zgessm_hash( __dague_object, this_task->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( (k == (descA.mt - 1)) ) {
    if( ADATA(this_task->data[2].data) != A(m, n) ) {
      int __arena_index = DAGUE_zgetrf_param_DEFAULT_ARENA;
      int __dtt_nb = 1;
      assert( (__arena_index>=0) && (__arena_index < __dague_object->super.arenas_size) );
      assert( __dtt_nb >= 0 );
      dague_remote_dep_memcpy( context, this_task->dague_object, A(m, n), this_task->data[2].data, 
                               __dague_object->super.arenas[__arena_index]->opaque_dtt,
                               __dtt_nb );
    }
  }
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(this_task, context->eu_id, zgessm_hash(__dague_object, this_task->locals));
  release_deps_of_zgetrf_param_zgessm(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      DAGUE_ACTION_DEPS_MASK,
      NULL);
  return 0;
}

#if defined(DAGUE_SIM)
static int simulation_cost_of_zgetrf_param_zgessm(const dague_execution_context_t *this_task)
{
  const dague_object_t *__dague_object = (const dague_object_t*)this_task->dague_object;
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int n = this_task->locals[2].value;
  int m = this_task->locals[3].value;
  int nextm = this_task->locals[4].value;
  (void)__dague_object;
  (void)k;  (void)i;  (void)n;  (void)m;  (void)nextm;
  return 6;
}
#endif

static int zgetrf_param_zgessm_internal_init(__dague_zgetrf_param_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, i, n, m, nextm;
  int32_t  k_min = 0x7fffffff, i_min = 0x7fffffff, n_min = 0x7fffffff;
  int32_t  k_max = 0, i_max = 0, n_max = 0;
  (void)__dague_object; (void)__foundone;
  int32_t k_start, k_end;  int32_t i_start, i_end;  int32_t n_start, n_end;  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
      k++) {
    assignments[0].value = k;
    for(i = 0;
        i <= zgetrf_param_inline_c_expr26_line_151((const dague_object_t*)__dague_object, assignments);
        i++) {
      assignments[1].value = i;
      for(n = (k + 1);
          n <= (descA.nt - 1);
          n++) {
        assignments[2].value = n;
        m = zgetrf_param_inline_c_expr25_line_153((const dague_object_t*)__dague_object, assignments);
        assignments[3].value = m;
        nextm = zgetrf_param_inline_c_expr24_line_154((const dague_object_t*)__dague_object, assignments);
        assignments[4].value = nextm;
        if( !zgessm_pred(k, i, n, m, nextm) ) continue;
        nb_tasks++;
        k_max = dague_imax(k_max, k);
        k_min = dague_imin(k_min, k);
        i_max = dague_imax(i_max, i);
        i_min = dague_imin(i_min, i);
        n_max = dague_imax(n_max, n);
        n_min = dague_imin(n_min, n);
      }
    }
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  dep = NULL;
  k_start = 0;
  k_end = ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
  for(k = dague_imax(k_start, k_min); k <= dague_imin(k_end, k_max); k++) {
    assignments[0].value = k;
    i_start = 0;
    i_end = zgetrf_param_inline_c_expr26_line_151((const dague_object_t*)__dague_object, assignments);
    for(i = dague_imax(i_start, i_min); i <= dague_imin(i_end, i_max); i++) {
      assignments[1].value = i;
      n_start = (k + 1);
      n_end = (descA.nt - 1);
      for(n = dague_imax(n_start, n_min); n <= dague_imin(n_end, n_max); n++) {
        assignments[2].value = n;
        m = zgetrf_param_inline_c_expr25_line_153((const dague_object_t*)__dague_object, assignments);
        assignments[3].value = m;
        __foundone = 0;
        nextm = zgetrf_param_inline_c_expr24_line_154((const dague_object_t*)__dague_object, assignments);
        assignments[4].value = nextm;
        if( zgessm_pred(k, i, n, m, nextm) ) {
          /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zgetrf_param_zgessm_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-k_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-k_min], i_min, i_max, "i", &symb_zgetrf_param_zgessm_i, dep, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-k_min]->u.next[i-i_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-k_min]->u.next[i-i_min], n_min, n_max, "n", &symb_zgetrf_param_zgessm_n, dep->u.next[k-k_min], DAGUE_DEPENDENCIES_FLAG_FINAL);
        }
        }
      }
    }
  }
  (void)k_start; (void)k_end;  (void)i_start; (void)i_end;  (void)n_start; (void)n_end;  __dague_object->super.super.dependencies_array[4] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static const dague_function_t zgetrf_param_zgessm = {
  .name = "zgessm",
  .deps = 4,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 4,
  .dependencies_goal = 0x7,
  .nb_parameters = 3,
  .nb_definitions = 5,
  .params = { &symb_zgetrf_param_zgessm_k, &symb_zgetrf_param_zgessm_i, &symb_zgetrf_param_zgessm_n },
  .locals = { &symb_zgetrf_param_zgessm_k, &symb_zgetrf_param_zgessm_i, &symb_zgetrf_param_zgessm_n, &symb_zgetrf_param_zgessm_m, &symb_zgetrf_param_zgessm_nextm },
  .pred = &pred_of_zgetrf_param_zgessm_as_expr,
  .priority = &priority_of_zgetrf_param_zgessm_as_expr,
  .in = { &flow_of_zgetrf_param_zgessm_for_A, &flow_of_zgetrf_param_zgessm_for_P, &flow_of_zgetrf_param_zgessm_for_C },
  .out = { &flow_of_zgetrf_param_zgessm_for_C },
  .iterate_successors = iterate_successors_of_zgetrf_param_zgessm,
  .release_deps = release_deps_of_zgetrf_param_zgessm,
  .hook = hook_of_zgetrf_param_zgessm,
  .complete_execution = complete_hook_of_zgetrf_param_zgessm,
#if defined(DAGUE_SIM)
  .sim_cost_fct = simulation_cost_of_zgetrf_param_zgessm,
#endif
  .key = (dague_functionkey_fn_t*)zgessm_hash,
};


/**********************************************************************************
 *                                  zgetrf_param                                  *
 **********************************************************************************/

static inline int minexpr_of_symb_zgetrf_param_zgetrf_param_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgetrf_param_zgetrf_param_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zgetrf_param_k_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zgetrf_param_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
}
static const expr_t maxexpr_of_symb_zgetrf_param_zgetrf_param_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zgetrf_param_k_fct
};
static const symbol_t symb_zgetrf_param_zgetrf_param_k = {.min = &minexpr_of_symb_zgetrf_param_zgetrf_param_k, .max = &maxexpr_of_symb_zgetrf_param_zgetrf_param_k,  .flags = 0x0};

static inline int minexpr_of_symb_zgetrf_param_zgetrf_param_i_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgetrf_param_zgetrf_param_i = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zgetrf_param_i_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zgetrf_param_i_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr30_line_94((const dague_object_t*)__dague_object, assignments);
}
static const expr_t maxexpr_of_symb_zgetrf_param_zgetrf_param_i = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zgetrf_param_i_fct
};
static const symbol_t symb_zgetrf_param_zgetrf_param_i = {.min = &minexpr_of_symb_zgetrf_param_zgetrf_param_i, .max = &maxexpr_of_symb_zgetrf_param_zgetrf_param_i,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zgetrf_param_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr29_line_95((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zgetrf_param_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zgetrf_param_m_fct
};
static const symbol_t symb_zgetrf_param_zgetrf_param_m = {.min = &expr_of_symb_zgetrf_param_zgetrf_param_m, .max = &expr_of_symb_zgetrf_param_zgetrf_param_m,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zgetrf_param_nextm_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr28_line_96((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zgetrf_param_nextm = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zgetrf_param_nextm_fct
};
static const symbol_t symb_zgetrf_param_zgetrf_param_nextm = {.min = &expr_of_symb_zgetrf_param_zgetrf_param_nextm, .max = &expr_of_symb_zgetrf_param_zgetrf_param_nextm,  .flags = 0x0};

static inline int pred_of_zgetrf_param_zgetrf_param_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int i = assignments[1].value;
  int m = assignments[2].value;
  int nextm = assignments[3].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)i;
  (void)m;
  (void)nextm;
  /* Compute Predicate */
  return zgetrf_param_pred(k, i, m, nextm);
}
static const expr_t pred_of_zgetrf_param_zgetrf_param_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zgetrf_param_zgetrf_param_as_expr_fct
};
static inline int priority_of_zgetrf_param_zgetrf_param_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr27_line_116((const dague_object_t*)__dague_object, assignments);
}
static const expr_t priority_of_zgetrf_param_zgetrf_param_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = priority_of_zgetrf_param_zgetrf_param_as_expr_fct
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (0 == k);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106_fct
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106,
  .dague = &zgetrf_param_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k > 0);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k - 1);
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107_fct
};
static inline int expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107_fct
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107,
  .dague = &zgetrf_param_zttmqr,
  .flow = &flow_of_zgetrf_param_zttmqr_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107,
    &expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107
  }
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep3_atline_109_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep3_atline_109 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep3_atline_109_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep3_atline_109_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int i = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return i;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep3_atline_109 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep3_atline_109_fct
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_for_A_dep3_atline_109 = {
  .cond = NULL,
  .dague = &zgetrf_param_zgetrf_param_out,
  .flow = &flow_of_zgetrf_param_zgetrf_param_out_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep3_atline_109,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep3_atline_109
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k == (descA.mt - 1));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109_fct
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109,
  .dague = &zgetrf_param_A,
  .datatype = { .index = DAGUE_zgetrf_param_UPPER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int nextm = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return ((k < (descA.mt - 1)) && (nextm != descA.mt));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int nextm = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return nextm;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110_fct
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_UPPER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int nextm = assignments[3].value;

  (void)__dague_object; (void)assignments;
  return ((k < (descA.mt - 1)) && (nextm == descA.mt));
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111_fct
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111,
  .dague = &zgetrf_param_zttqrt,
  .flow = &flow_of_zgetrf_param_zttqrt_for_C,
  .datatype = { .index = DAGUE_zgetrf_param_UPPER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111
  }
};
static const dague_flow_t flow_of_zgetrf_param_zgetrf_param_for_A = {
  .name = "A",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 0,
  .dep_in  = { &flow_of_zgetrf_param_zgetrf_param_for_A_dep1_atline_106, &flow_of_zgetrf_param_zgetrf_param_for_A_dep2_atline_107 },
  .dep_out = { &flow_of_zgetrf_param_zgetrf_param_for_A_dep3_atline_109, &flow_of_zgetrf_param_zgetrf_param_for_A_dep4_atline_109, &flow_of_zgetrf_param_zgetrf_param_for_A_dep5_atline_110, &flow_of_zgetrf_param_zgetrf_param_for_A_dep6_atline_111 }
};

static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep1_atline_112_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep1_atline_112 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep1_atline_112_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep1_atline_112_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep1_atline_112 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep1_atline_112_fct
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_for_P_dep1_atline_112 = {
  .cond = NULL,
  .dague = &zgetrf_param_IPIV,
  .datatype = { .index = DAGUE_zgetrf_param_PIVOT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep1_atline_112,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep1_atline_112
  }
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep2_atline_113_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep2_atline_113 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep2_atline_113_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep2_atline_113_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep2_atline_113 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep2_atline_113_fct
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_for_P_dep2_atline_113 = {
  .cond = NULL,
  .dague = &zgetrf_param_IPIV,
  .datatype = { .index = DAGUE_zgetrf_param_PIVOT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep2_atline_113,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep2_atline_113
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return ((descA.nt - 1) > k);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int i = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return i;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114_fct
};
static inline int rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114_fct
};
static inline int rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return (descA.nt - 1);
}
static const expr_t rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114_fct
};
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114 = {
  .op = EXPR_OP_BINARY_RANGE,
  .u_expr.binary = {
    .op1 = &rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114,
    .op2 = &rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114
  }
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114,
  .dague = &zgetrf_param_zgessm,
  .flow = &flow_of_zgetrf_param_zgessm_for_P,
  .datatype = { .index = DAGUE_zgetrf_param_PIVOT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114,
    &expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114
  }
};
static const dague_flow_t flow_of_zgetrf_param_zgetrf_param_for_P = {
  .name = "P",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 1,
  .dep_in  = { &flow_of_zgetrf_param_zgetrf_param_for_P_dep1_atline_112 },
  .dep_out = { &flow_of_zgetrf_param_zgetrf_param_for_P_dep2_atline_113, &flow_of_zgetrf_param_zgetrf_param_for_P_dep3_atline_114 }
};

static void
iterate_successors_of_zgetrf_param_zgetrf_param(dague_execution_unit_t *eu, dague_execution_context_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)this_task->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int __nb_elt = -1;
  int rank_src = 0, rank_dst = 0;
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int m = this_task->locals[2].value;
  int nextm = this_task->locals[3].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object; (void)__nb_elt;
  (void)k;  (void)i;  (void)m;  (void)nextm;
  nc.dague_object = this_task->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, m, k);
#endif
  /* Flow of Data A */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_DEFAULT_ARENA];
#endif
#if defined(DISTRIBUTED)
    __nb_elt = 1;
#endif  /* defined(DISTRIBUTED) */
    nc.function = (const dague_function_t*)&zgetrf_param_zgetrf_param_out;
    {
      const int zgetrf_param_out_k = k;
      if( (zgetrf_param_out_k >= (0)) && (zgetrf_param_out_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
        nc.locals[0].value = zgetrf_param_out_k;
        {
          const int zgetrf_param_out_i = i;
          if( (zgetrf_param_out_i >= (0)) && (zgetrf_param_out_i <= (zgetrf_param_inline_c_expr32_line_77((const dague_object_t*)__dague_object, nc.locals))) ) {
            nc.locals[1].value = zgetrf_param_out_i;
            const int zgetrf_param_out_m = zgetrf_param_inline_c_expr31_line_78((const dague_object_t*)__dague_object, nc.locals);
            nc.locals[2].value = zgetrf_param_out_m;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zgetrf_param_out_m, zgetrf_param_out_k);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
            if( NULL != eu ) {
              char tmp[128], tmp1[128];
              DEBUG(("thread %d release deps of A:%s to A:%s (from node %d to %d)\n", eu->eu_id,
                     dague_service_to_string(this_task, tmp, 128),
                     dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
            }
#endif
              nc.priority = 0;
              if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 0, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                return;
    }
      }
        }
          }
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_UPPER_TILE_ARENA];
#endif
    if( ((k < (descA.mt - 1)) && (nextm != descA.mt)) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttqrt;
      {
        const int zttqrt_k = k;
        if( (zttqrt_k >= (0)) && (zttqrt_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttqrt_k;
          {
            const int zttqrt_m = nextm;
            if( (zttqrt_m >= ((zttqrt_k + 1))) && (zttqrt_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttqrt_m;
              const int zttqrt_p = zgetrf_param_inline_c_expr21_line_210((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[2].value = zttqrt_p;
              const int zttqrt_nextp = zgetrf_param_inline_c_expr20_line_211((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[3].value = zttqrt_nextp;
              const int zttqrt_prevp = zgetrf_param_inline_c_expr19_line_212((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[4].value = zttqrt_prevp;
              const int zttqrt_prevm = zgetrf_param_inline_c_expr18_line_213((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[5].value = zttqrt_prevm;
              const int zttqrt_type = zgetrf_param_inline_c_expr17_line_214((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[6].value = zttqrt_type;
              const int zttqrt_ip = zgetrf_param_inline_c_expr16_line_215((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[7].value = zttqrt_ip;
              const int zttqrt_im = zgetrf_param_inline_c_expr15_line_216((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[8].value = zttqrt_im;
#if defined(DISTRIBUTED)
                rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttqrt_m, zttqrt_k);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
              if( NULL != eu ) {
                char tmp[128], tmp1[128];
                DEBUG(("thread %d release deps of A:%s to A:%s (from node %d to %d)\n", eu->eu_id,
                       dague_service_to_string(this_task, tmp, 128),
                       dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
              }
#endif
                nc.priority = priority_of_zgetrf_param_zttqrt_as_expr_fct(this_task->dague_object, nc.locals);
                if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 2, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                  return;
      }
        }
          }
            }
    }
    if( ((k < (descA.mt - 1)) && (nextm == descA.mt)) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zttqrt;
      {
        const int zttqrt_k = k;
        if( (zttqrt_k >= (0)) && (zttqrt_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zttqrt_k;
          {
            const int zttqrt_m = m;
            if( (zttqrt_m >= ((zttqrt_k + 1))) && (zttqrt_m <= ((descA.mt - 1))) ) {
              nc.locals[1].value = zttqrt_m;
              const int zttqrt_p = zgetrf_param_inline_c_expr21_line_210((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[2].value = zttqrt_p;
              const int zttqrt_nextp = zgetrf_param_inline_c_expr20_line_211((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[3].value = zttqrt_nextp;
              const int zttqrt_prevp = zgetrf_param_inline_c_expr19_line_212((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[4].value = zttqrt_prevp;
              const int zttqrt_prevm = zgetrf_param_inline_c_expr18_line_213((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[5].value = zttqrt_prevm;
              const int zttqrt_type = zgetrf_param_inline_c_expr17_line_214((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[6].value = zttqrt_type;
              const int zttqrt_ip = zgetrf_param_inline_c_expr16_line_215((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[7].value = zttqrt_ip;
              const int zttqrt_im = zgetrf_param_inline_c_expr15_line_216((const dague_object_t*)__dague_object, nc.locals);
              nc.locals[8].value = zttqrt_im;
#if defined(DISTRIBUTED)
                rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zttqrt_m, zttqrt_k);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
              if( NULL != eu ) {
                char tmp[128], tmp1[128];
                DEBUG(("thread %d release deps of A:%s to C:%s (from node %d to %d)\n", eu->eu_id,
                       dague_service_to_string(this_task, tmp, 128),
                       dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
              }
#endif
                nc.priority = priority_of_zgetrf_param_zttqrt_as_expr_fct(this_task->dague_object, nc.locals);
                if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 3, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                  return;
      }
        }
          }
            }
    }
  }
  /* Flow of Data P */
  if( action_mask & (1 << 1) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_PIVOT_ARENA];
#endif
#if defined(DISTRIBUTED)
    __nb_elt = 1;
#endif  /* defined(DISTRIBUTED) */
    if( ((descA.nt - 1) > k) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zgessm;
      {
        const int zgessm_k = k;
        if( (zgessm_k >= (0)) && (zgessm_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zgessm_k;
          {
            const int zgessm_i = i;
            if( (zgessm_i >= (0)) && (zgessm_i <= (zgetrf_param_inline_c_expr26_line_151((const dague_object_t*)__dague_object, nc.locals))) ) {
              nc.locals[1].value = zgessm_i;
              {
                int zgessm_n;
                for( zgessm_n = (k + 1);zgessm_n <= (descA.nt - 1); zgessm_n++ ) {
                  if( (zgessm_n >= ((zgessm_k + 1))) && (zgessm_n <= ((descA.nt - 1))) ) {
                    nc.locals[2].value = zgessm_n;
                    const int zgessm_m = zgetrf_param_inline_c_expr25_line_153((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[3].value = zgessm_m;
                    const int zgessm_nextm = zgetrf_param_inline_c_expr24_line_154((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[4].value = zgessm_nextm;
#if defined(DISTRIBUTED)
                      rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zgessm_m, zgessm_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                    if( NULL != eu ) {
                      char tmp[128], tmp1[128];
                      DEBUG(("thread %d release deps of P:%s to P:%s (from node %d to %d)\n", eu->eu_id,
                             dague_service_to_string(this_task, tmp, 128),
                             dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                    }
#endif
                      nc.priority = priority_of_zgetrf_param_zgessm_as_expr_fct(this_task->dague_object, nc.locals);
                      if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 1, 1, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                        return;
      }
        }
          }
            }
              }
                }
                  }
    }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zgetrf_param_zgetrf_param(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, zgetrf_param_repo, zgetrf_param_hash(__dague_object, context->locals) );
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
  }
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zgetrf_param_zgetrf_param(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(zgetrf_param_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int i = context->locals[1].value;
    int m = context->locals[2].value;
    int nextm = context->locals[3].value;
    (void)k; (void)i; (void)m; (void)nextm;

    if( (k > 0) ) {
      data_repo_entry_used_once( eu, zttmqr_repo, context->data[0].data_repo->key );
      (void)AUNREF(context->data[0].data);
    }
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zgetrf_param_zgetrf_param(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int m = this_task->locals[2].value;
  int nextm = this_task->locals[3].value;
  (void)k;  (void)i;  (void)m;  (void)nextm;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;
  void *P = NULL; (void)P;
  dague_arena_chunk_t *gP = NULL; (void)gP;
  data_repo_entry_t *eP = NULL; (void)eP;

  /** Lookup the input data, and store them in the context if any */
  eA = this_task->data[0].data_repo;
  gA = this_task->data[0].data;
  if( NULL == gA ) {
  if( (0 == k) ) {
    gA = (dague_arena_chunk_t*) A(m, k);
  }
  else if( (k > 0) ) {
      tass[0].value = (k - 1);
      tass[1].value = m;
      tass[2].value = k;
    eA = data_repo_lookup_entry( zttmqr_repo, zttmqr_hash( __dague_object, tass ));
    gA = eA->data[1];
  }
    this_task->data[0].data = gA;
    this_task->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
  eP = this_task->data[1].data_repo;
  gP = this_task->data[1].data;
  if( NULL == gP ) {
  gP = (dague_arena_chunk_t*) IPIV(m, k);
    this_task->data[1].data = gP;
    this_task->data[1].data_repo = eP;
  }
  P = ADATA(gP);
#if defined(DAGUE_SIM)
  if( (NULL != eP) && (eP->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eP->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( this_task->function->sim_cost_fct != NULL ) {
    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);
  } else {
    this_task->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < this_task->sim_exec_date )
    context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
  cache_buf_referenced(context->closest_cache, P);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                              zgetrf_param BODY                                *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*this_task->function->function_id, zgetrf_param_hash( __dague_object, this_task->locals), __dague_object->super.A, ((dague_ddesc_t*)(__dague_object->super.A))->data_key((dague_ddesc_t*)__dague_object->super.A, m, k) );
#line 118 "zgetrf_param.jdf"

  int iinfo = 0;
  DRYRUN(
         int tempkm = ((k)==(descA.mt-1)) ? (descA.m-(k*descA.mb)) : (descA.mb);
         int tempkn = ((k)==(descA.nt-1)) ? (descA.n-(k*descA.nb)) : (descA.nb);
         int ldak   = descA.mb; /*((k+(Ai/descA.mb))<Alm1) ? (descA.mb) : (Alm%Bmb);*/

         /* Set local IPIV to 0 before generation
          * Better here than a global initialization for locality
          * and it's also done in parallel */
         memset(P, 0, min(tempkn, tempkm) * sizeof(int) );

         CORE_zgetrf_incpiv(tempkm, tempkn, ib,
                     A /* A(k,k)    */, ldak,
                     P /* IPIV(k,k) */, &iinfo );

         if ( (iinfo != 0) && (k == descA.mt-1) ) {
             *INFO = k * descA.mb + iinfo; /* Should return if enter here */
             fprintf(stderr, "zgetrf(%d) failed => %d\n", k, *INFO );
         }
         );

   printlog("thread %d   CORE_zgetrf(%d)\n"
            "\t(tempkm, tempkn, ib, A(%d,%d)[%p], ldak, IPIV(%d,%d)[%p]) => info = %d\n",
            context->eu_id, k, k, k, A, k, k, P, k * descA.mb + iinfo);


#line 7616 "zgetrf_param.c"
/*--------------------------------------------------------------------------------*
 *                            END OF zgetrf_param BODY                            *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zgetrf_param_zgetrf_param(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  (void)context; (void)__dague_object;
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int m = this_task->locals[2].value;
  int nextm = this_task->locals[3].value;
  (void)k;  (void)i;  (void)m;  (void)nextm;

  TAKE_TIME(context,2*this_task->function->function_id+1, zgetrf_param_hash( __dague_object, this_task->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( (k == (descA.mt - 1)) ) {
    if( ADATA(this_task->data[0].data) != A(m, k) ) {
      int __arena_index = DAGUE_zgetrf_param_UPPER_TILE_ARENA;
      int __dtt_nb = 1;
      assert( (__arena_index>=0) && (__arena_index < __dague_object->super.arenas_size) );
      assert( __dtt_nb >= 0 );
      dague_remote_dep_memcpy( context, this_task->dague_object, A(m, k), this_task->data[0].data, 
                               __dague_object->super.arenas[__arena_index]->opaque_dtt,
                               __dtt_nb );
    }
  }
  if( ADATA(this_task->data[1].data) != IPIV(m, k) ) {
    int __arena_index = DAGUE_zgetrf_param_PIVOT_ARENA;
    int __dtt_nb = 1;
    assert( (__arena_index>=0) && (__arena_index < __dague_object->super.arenas_size) );
    assert( __dtt_nb >= 0 );
    dague_remote_dep_memcpy( context, this_task->dague_object, IPIV(m, k), this_task->data[1].data, 
                             __dague_object->super.arenas[__arena_index]->opaque_dtt,
                             __dtt_nb );
  }
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(this_task, context->eu_id, zgetrf_param_hash(__dague_object, this_task->locals));
  release_deps_of_zgetrf_param_zgetrf_param(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      DAGUE_ACTION_DEPS_MASK,
      NULL);
  return 0;
}

#if defined(DAGUE_SIM)
static int simulation_cost_of_zgetrf_param_zgetrf_param(const dague_execution_context_t *this_task)
{
  const dague_object_t *__dague_object = (const dague_object_t*)this_task->dague_object;
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int m = this_task->locals[2].value;
  int nextm = this_task->locals[3].value;
  (void)__dague_object;
  (void)k;  (void)i;  (void)m;  (void)nextm;
  return 4;
}
#endif

static int zgetrf_param_zgetrf_param_internal_init(__dague_zgetrf_param_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, i, m, nextm;
  int32_t  k_min = 0x7fffffff, i_min = 0x7fffffff;
  int32_t  k_max = 0, i_max = 0;
  (void)__dague_object; (void)__foundone;
  int32_t k_start, k_end;  int32_t i_start, i_end;  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
      k++) {
    assignments[0].value = k;
    for(i = 0;
        i <= zgetrf_param_inline_c_expr30_line_94((const dague_object_t*)__dague_object, assignments);
        i++) {
      assignments[1].value = i;
      m = zgetrf_param_inline_c_expr29_line_95((const dague_object_t*)__dague_object, assignments);
      assignments[2].value = m;
      nextm = zgetrf_param_inline_c_expr28_line_96((const dague_object_t*)__dague_object, assignments);
      assignments[3].value = nextm;
      if( !zgetrf_param_pred(k, i, m, nextm) ) continue;
      nb_tasks++;
      k_max = dague_imax(k_max, k);
      k_min = dague_imin(k_min, k);
      i_max = dague_imax(i_max, i);
      i_min = dague_imin(i_min, i);
    }
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  dep = NULL;
  k_start = 0;
  k_end = ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
  for(k = dague_imax(k_start, k_min); k <= dague_imin(k_end, k_max); k++) {
    assignments[0].value = k;
    i_start = 0;
    i_end = zgetrf_param_inline_c_expr30_line_94((const dague_object_t*)__dague_object, assignments);
    for(i = dague_imax(i_start, i_min); i <= dague_imin(i_end, i_max); i++) {
      assignments[1].value = i;
      m = zgetrf_param_inline_c_expr29_line_95((const dague_object_t*)__dague_object, assignments);
      assignments[2].value = m;
      __foundone = 0;
      nextm = zgetrf_param_inline_c_expr28_line_96((const dague_object_t*)__dague_object, assignments);
      assignments[3].value = nextm;
      if( zgetrf_param_pred(k, i, m, nextm) ) {
        /* We did find one! Allocate the dependencies array. */
      if( dep == NULL ) {
        ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zgetrf_param_zgetrf_param_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
      }
      if( dep->u.next[k-k_min] == NULL ) {
        ALLOCATE_DEP_TRACKING(dep->u.next[k-k_min], i_min, i_max, "i", &symb_zgetrf_param_zgetrf_param_i, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
      }
      }
    }
  }
  (void)k_start; (void)k_end;  (void)i_start; (void)i_end;  __dague_object->super.super.dependencies_array[5] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static int zgetrf_param_zgetrf_param_startup_tasks(dague_context_t *context, const __dague_zgetrf_param_internal_object_t *__dague_object, dague_execution_context_t** pready_list)
{
  dague_execution_context_t* new_context;
  assignment_t *assignments = NULL;
  int32_t  k = -1, i = -1, m = -1, nextm = -1;
  (void)k; (void)i; (void)m; (void)nextm;
  new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( context->execution_units[0]->context_mempool );
  assignments = new_context->locals;
  /* Parse all the inputs and generate the ready execution tasks */
  for(k = 0;
      k <= ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
      k++) {
    assignments[0].value = k;
    for(i = 0;
        i <= zgetrf_param_inline_c_expr30_line_94((const dague_object_t*)__dague_object, assignments);
        i++) {
      assignments[1].value = i;
      assignments[2].value = m = zgetrf_param_inline_c_expr29_line_95((const dague_object_t*)__dague_object, assignments);
      assignments[3].value = nextm = zgetrf_param_inline_c_expr28_line_96((const dague_object_t*)__dague_object, assignments);
      if( !zgetrf_param_pred(k, i, m, nextm) ) continue;
      if( !(((0 == k))) ) continue;
      DAGUE_STAT_INCREASE(mem_contexts, sizeof(dague_execution_context_t) + STAT_MALLOC_OVERHEAD);
      DAGUE_LIST_ITEM_SINGLETON( new_context );
      new_context->dague_object = (dague_object_t*)__dague_object;
      new_context->function = (const dague_function_t*)&zgetrf_param_zgetrf_param;
      new_context->priority = priority_of_zgetrf_param_zgetrf_param_as_expr_fct(new_context->dague_object, new_context->locals);
    new_context->data[0].data_repo = NULL;
    new_context->data[0].data      = NULL;
    new_context->data[1].data_repo = NULL;
    new_context->data[1].data      = NULL;
#if defined(DAGUE_DEBUG_VERBOSE2)
      {
        char tmp[128];
        DEBUG2(("Add startup task %s\n",
               dague_service_to_string(new_context, tmp, 128)));
      }
#endif
      dague_list_add_single_elem_by_priority( pready_list, new_context );
      new_context = (dague_execution_context_t*)dague_thread_mempool_allocate( context->execution_units[0]->context_mempool );
      assignments = new_context->locals;
      assignments[0].value = k;
      assignments[1].value = i;
      assignments[2].value = m;
    }
  }
  dague_thread_mempool_free( context->execution_units[0]->context_mempool, new_context );
  return 0;
}

static const dague_function_t zgetrf_param_zgetrf_param = {
  .name = "zgetrf_param",
  .deps = 5,
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES,
  .function_id = 5,
  .dependencies_goal = 0x3,
  .nb_parameters = 2,
  .nb_definitions = 4,
  .params = { &symb_zgetrf_param_zgetrf_param_k, &symb_zgetrf_param_zgetrf_param_i },
  .locals = { &symb_zgetrf_param_zgetrf_param_k, &symb_zgetrf_param_zgetrf_param_i, &symb_zgetrf_param_zgetrf_param_m, &symb_zgetrf_param_zgetrf_param_nextm },
  .pred = &pred_of_zgetrf_param_zgetrf_param_as_expr,
  .priority = &priority_of_zgetrf_param_zgetrf_param_as_expr,
  .in = { &flow_of_zgetrf_param_zgetrf_param_for_A, &flow_of_zgetrf_param_zgetrf_param_for_P },
  .out = { &flow_of_zgetrf_param_zgetrf_param_for_A, &flow_of_zgetrf_param_zgetrf_param_for_P },
  .iterate_successors = iterate_successors_of_zgetrf_param_zgetrf_param,
  .release_deps = release_deps_of_zgetrf_param_zgetrf_param,
  .hook = hook_of_zgetrf_param_zgetrf_param,
  .complete_execution = complete_hook_of_zgetrf_param_zgetrf_param,
#if defined(DAGUE_SIM)
  .sim_cost_fct = simulation_cost_of_zgetrf_param_zgetrf_param,
#endif
  .key = (dague_functionkey_fn_t*)zgetrf_param_hash,
};


/**********************************************************************************
 *                                zgetrf_param_out                                *
 **********************************************************************************/

static inline int minexpr_of_symb_zgetrf_param_zgetrf_param_out_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgetrf_param_zgetrf_param_out_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zgetrf_param_out_k_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zgetrf_param_out_k_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
}
static const expr_t maxexpr_of_symb_zgetrf_param_zgetrf_param_out_k = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zgetrf_param_out_k_fct
};
static const symbol_t symb_zgetrf_param_zgetrf_param_out_k = {.min = &minexpr_of_symb_zgetrf_param_zgetrf_param_out_k, .max = &maxexpr_of_symb_zgetrf_param_zgetrf_param_out_k,  .flags = 0x0};

static inline int minexpr_of_symb_zgetrf_param_zgetrf_param_out_i_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return 0;
}
static const expr_t minexpr_of_symb_zgetrf_param_zgetrf_param_out_i = {
  .op = EXPR_OP_INLINE,
  .inline_func = minexpr_of_symb_zgetrf_param_zgetrf_param_out_i_fct
};
static inline int maxexpr_of_symb_zgetrf_param_zgetrf_param_out_i_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr32_line_77((const dague_object_t*)__dague_object, assignments);
}
static const expr_t maxexpr_of_symb_zgetrf_param_zgetrf_param_out_i = {
  .op = EXPR_OP_INLINE,
  .inline_func = maxexpr_of_symb_zgetrf_param_zgetrf_param_out_i_fct
};
static const symbol_t symb_zgetrf_param_zgetrf_param_out_i = {.min = &minexpr_of_symb_zgetrf_param_zgetrf_param_out_i, .max = &maxexpr_of_symb_zgetrf_param_zgetrf_param_out_i,  .flags = 0x0};

static inline int expr_of_symb_zgetrf_param_zgetrf_param_out_m_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return zgetrf_param_inline_c_expr31_line_78((const dague_object_t*)__dague_object, assignments);
}
static const expr_t expr_of_symb_zgetrf_param_zgetrf_param_out_m = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_symb_zgetrf_param_zgetrf_param_out_m_fct
};
static const symbol_t symb_zgetrf_param_zgetrf_param_out_m = {.min = &expr_of_symb_zgetrf_param_zgetrf_param_out_m, .max = &expr_of_symb_zgetrf_param_zgetrf_param_out_m,  .flags = 0x0};

static inline int pred_of_zgetrf_param_zgetrf_param_out_as_expr_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;
  int i = assignments[1].value;
  int m = assignments[2].value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_object;
  (void)k;
  (void)i;
  (void)m;
  /* Compute Predicate */
  return zgetrf_param_out_pred(k, i, m);
}
static const expr_t pred_of_zgetrf_param_zgetrf_param_out_as_expr = {
  .op = EXPR_OP_INLINE,
  .inline_func = pred_of_zgetrf_param_zgetrf_param_out_as_expr_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep1_atline_83_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep1_atline_83 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep1_atline_83_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep1_atline_83_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int i = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return i;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep1_atline_83 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep1_atline_83_fct
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_out_for_A_dep1_atline_83 = {
  .cond = NULL,
  .dague = &zgetrf_param_zgetrf_param,
  .flow = &flow_of_zgetrf_param_zgetrf_param_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_DEFAULT_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep1_atline_83,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep1_atline_83
  }
};
static inline int expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return ((descA.nt - 1) > k);
}
static const expr_t expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83_fct
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int i = assignments[1].value;

  (void)__dague_object; (void)assignments;
  return i;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83_fct
};
static inline int rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return (k + 1);
}
static const expr_t rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83_fct
};
static inline int rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
;

  (void)__dague_object; (void)assignments;
  return (descA.nt - 1);
}
static const expr_t rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83 = {
  .op = EXPR_OP_INLINE,
  .inline_func = rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83_fct
};
static const expr_t expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83 = {
  .op = EXPR_OP_BINARY_RANGE,
  .u_expr.binary = {
    .op1 = &rangemin_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83,
    .op2 = &rangemax_of_expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83
  }
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83 = {
  .cond = &expr_of_cond_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83,
  .dague = &zgetrf_param_zgessm,
  .flow = &flow_of_zgetrf_param_zgessm_for_A,
  .datatype = { .index = DAGUE_zgetrf_param_LOWER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83,
    &expr_of_p3_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83
  }
};
static inline int expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep3_atline_84_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int m = assignments[2].value;

  (void)__dague_object; (void)assignments;
  return m;
}
static const expr_t expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep3_atline_84 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep3_atline_84_fct
};
static inline int expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep3_atline_84_fct(const dague_object_t *__dague_object_parent, const assignment_t *assignments)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)__dague_object_parent;
  int k = assignments[0].value;

  (void)__dague_object; (void)assignments;
  return k;
}
static const expr_t expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep3_atline_84 = {
  .op = EXPR_OP_INLINE,
  .inline_func = expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep3_atline_84_fct
};
static const dep_t flow_of_zgetrf_param_zgetrf_param_out_for_A_dep3_atline_84 = {
  .cond = NULL,
  .dague = &zgetrf_param_A,
  .datatype = { .index = DAGUE_zgetrf_param_LOWER_TILE_ARENA, .index_fct = NULL,.nb_elt = 1, .nb_elt_fct = NULL },
  .call_params = {
    &expr_of_p1_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep3_atline_84,
    &expr_of_p2_for_flow_of_zgetrf_param_zgetrf_param_out_for_A_dep3_atline_84
  }
};
static const dague_flow_t flow_of_zgetrf_param_zgetrf_param_out_for_A = {
  .name = "A",
  .sym_type = SYM_INOUT,
  .access_type = ACCESS_RW,
  .flow_index = 0,
  .dep_in  = { &flow_of_zgetrf_param_zgetrf_param_out_for_A_dep1_atline_83 },
  .dep_out = { &flow_of_zgetrf_param_zgetrf_param_out_for_A_dep2_atline_83, &flow_of_zgetrf_param_zgetrf_param_out_for_A_dep3_atline_84 }
};

static void
iterate_successors_of_zgetrf_param_zgetrf_param_out(dague_execution_unit_t *eu, dague_execution_context_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t*)this_task->dague_object;
  dague_execution_context_t nc;
  dague_arena_t* arena = NULL;
  int __nb_elt = -1;
  int rank_src = 0, rank_dst = 0;
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int m = this_task->locals[2].value;
  (void)rank_src; (void)rank_dst; (void)__dague_object; (void)__nb_elt;
  (void)k;  (void)i;  (void)m;
  nc.dague_object = this_task->dague_object;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, m, k);
#endif
  /* Flow of Data A */
  if( action_mask & (1 << 0) ) {
#if defined(DISTRIBUTED)
    arena = __dague_object->super.arenas[DAGUE_zgetrf_param_LOWER_TILE_ARENA];
#endif
#if defined(DISTRIBUTED)
    __nb_elt = 1;
#endif  /* defined(DISTRIBUTED) */
    if( ((descA.nt - 1) > k) ) {
      nc.function = (const dague_function_t*)&zgetrf_param_zgessm;
      {
        const int zgessm_k = k;
        if( (zgessm_k >= (0)) && (zgessm_k <= (((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1)))) ) {
          nc.locals[0].value = zgessm_k;
          {
            const int zgessm_i = i;
            if( (zgessm_i >= (0)) && (zgessm_i <= (zgetrf_param_inline_c_expr26_line_151((const dague_object_t*)__dague_object, nc.locals))) ) {
              nc.locals[1].value = zgessm_i;
              {
                int zgessm_n;
                for( zgessm_n = (k + 1);zgessm_n <= (descA.nt - 1); zgessm_n++ ) {
                  if( (zgessm_n >= ((zgessm_k + 1))) && (zgessm_n <= ((descA.nt - 1))) ) {
                    nc.locals[2].value = zgessm_n;
                    const int zgessm_m = zgetrf_param_inline_c_expr25_line_153((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[3].value = zgessm_m;
                    const int zgessm_nextm = zgetrf_param_inline_c_expr24_line_154((const dague_object_t*)__dague_object, nc.locals);
                    nc.locals[4].value = zgessm_nextm;
#if defined(DISTRIBUTED)
                      rank_dst = ((dague_ddesc_t*)__dague_object->super.A)->rank_of((dague_ddesc_t*)__dague_object->super.A, zgessm_m, zgessm_n);
#endif
#if defined(DAGUE_DEBUG_VERBOSE1)
                    if( NULL != eu ) {
                      char tmp[128], tmp1[128];
                      DEBUG(("thread %d release deps of A:%s to A:%s (from node %d to %d)\n", eu->eu_id,
                             dague_service_to_string(this_task, tmp, 128),
                             dague_service_to_string(&nc, tmp1, 128), rank_src, rank_dst));
                    }
#endif
                      nc.priority = priority_of_zgetrf_param_zgessm_as_expr_fct(this_task->dague_object, nc.locals);
                      if( DAGUE_ITERATE_STOP == ontask(eu, &nc, this_task, 0, 0, rank_src, rank_dst, arena, __nb_elt, ontask_arg) )
                        return;
      }
        }
          }
            }
              }
                }
                  }
    }
  }
  (void)nc;(void)arena;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_zgetrf_param_zgetrf_param_out(dague_execution_unit_t *eu, dague_execution_context_t *context, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (const __dague_zgetrf_param_internal_object_t *)context->dague_object;
  dague_release_dep_fct_arg_t arg;
  arg.nb_released = 0;
  arg.output_usage = 0;
  arg.action_mask = action_mask;
  arg.deps = deps;
  arg.ready_list = NULL;
  (void)__dague_object;
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, zgetrf_param_out_repo, zgetrf_param_out_hash(__dague_object, context->locals) );
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = context->sim_exec_date;
#endif
  }
#if defined(DISTRIBUTED)
  arg.remote_deps_count = 0;
  arg.remote_deps = NULL;
#endif
  iterate_successors_of_zgetrf_param_zgetrf_param_out(eu, context, action_mask, dague_release_dep_fct, &arg);

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(zgetrf_param_out_repo, arg.output_entry->key, arg.output_usage);
    if( NULL != arg.ready_list ) {
      __dague_schedule(eu, arg.ready_list);
      arg.ready_list = NULL;
    }
  }
#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && arg.remote_deps_count ) {
    arg.nb_released += dague_remote_dep_activate(eu, context, arg.remote_deps, arg.remote_deps_count);
  }
#endif
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    int k = context->locals[0].value;
    int i = context->locals[1].value;
    int m = context->locals[2].value;
    (void)k; (void)i; (void)m;

    data_repo_entry_used_once( eu, zgetrf_param_repo, context->data[0].data_repo->key );
    (void)AUNREF(context->data[0].data);
  }
  assert( NULL == arg.ready_list );
  return arg.nb_released;
}

static int hook_of_zgetrf_param_zgetrf_param_out(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  assignment_t tass[MAX_PARAM_COUNT];
  (void)context; (void)__dague_object; (void)tass;
#if defined(DAGUE_SIM)
  int __dague_simulation_date = 0;
#endif
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int m = this_task->locals[2].value;
  (void)k;  (void)i;  (void)m;

  /** Declare the variables that will hold the data, and all the accounting for each */
  void *A = NULL; (void)A;
  dague_arena_chunk_t *gA = NULL; (void)gA;
  data_repo_entry_t *eA = NULL; (void)eA;

  /** Lookup the input data, and store them in the context if any */
  eA = this_task->data[0].data_repo;
  gA = this_task->data[0].data;
  if( NULL == gA ) {
  tass[0].value = k;
  tass[1].value = i;
  eA = data_repo_lookup_entry( zgetrf_param_repo, zgetrf_param_hash( __dague_object, tass ));
  gA = eA->data[0];
    this_task->data[0].data = gA;
    this_task->data[0].data_repo = eA;
  }
  A = ADATA(gA);
#if defined(DAGUE_SIM)
  if( (NULL != eA) && (eA->sim_exec_date > __dague_simulation_date) )
    __dague_simulation_date =  eA->sim_exec_date;
#endif
#if defined(DAGUE_SIM)
  if( this_task->function->sim_cost_fct != NULL ) {
    this_task->sim_exec_date = __dague_simulation_date + this_task->function->sim_cost_fct(this_task);
  } else {
    this_task->sim_exec_date = __dague_simulation_date;
  }
  if( context->largest_simulation_date < this_task->sim_exec_date )
    context->largest_simulation_date = this_task->sim_exec_date;
#endif
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_start_thread_counters();
#endif
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*--------------------------------------------------------------------------------*
 *                            zgetrf_param_out BODY                              *
 *--------------------------------------------------------------------------------*/

  TAKE_TIME(context, 2*this_task->function->function_id, zgetrf_param_out_hash( __dague_object, this_task->locals), __dague_object->super.A, ((dague_ddesc_t*)(__dague_object->super.A))->data_key((dague_ddesc_t*)__dague_object->super.A, m, k) );
#line 85 "zgetrf_param.jdf"
 /* Nothing */

#line 8254 "zgetrf_param.c"
/*--------------------------------------------------------------------------------*
 *                          END OF zgetrf_param_out BODY                          *
 *--------------------------------------------------------------------------------*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return 0;
}
static int complete_hook_of_zgetrf_param_zgetrf_param_out(dague_execution_unit_t *context, dague_execution_context_t *this_task)
{
  const __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)this_task->dague_object;
  (void)context; (void)__dague_object;
  int k = this_task->locals[0].value;
  int i = this_task->locals[1].value;
  int m = this_task->locals[2].value;
  (void)k;  (void)i;  (void)m;

  TAKE_TIME(context,2*this_task->function->function_id+1, zgetrf_param_out_hash( __dague_object, this_task->locals ), NULL, 0);
  /** PAPI events */
#if defined(HAVE_PAPI)
  papime_stop_thread_counters();
#endif
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( ADATA(this_task->data[0].data) != A(m, k) ) {
    int __arena_index = DAGUE_zgetrf_param_LOWER_TILE_ARENA;
    int __dtt_nb = 1;
    assert( (__arena_index>=0) && (__arena_index < __dague_object->super.arenas_size) );
    assert( __dtt_nb >= 0 );
    dague_remote_dep_memcpy( context, this_task->dague_object, A(m, k), this_task->data[0].data, 
                             __dague_object->super.arenas[__arena_index]->opaque_dtt,
                             __dtt_nb );
  }
#endif /* DISTRIBUTED */
  dague_prof_grapher_task(this_task, context->eu_id, zgetrf_param_out_hash(__dague_object, this_task->locals));
  release_deps_of_zgetrf_param_zgetrf_param_out(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      DAGUE_ACTION_DEPS_MASK,
      NULL);
  return 0;
}

static int zgetrf_param_zgetrf_param_out_internal_init(__dague_zgetrf_param_internal_object_t *__dague_object)
{
  dague_dependencies_t *dep = NULL;
  assignment_t assignments[MAX_LOCAL_COUNT];(void) assignments;
  int nb_tasks = 0, __foundone = 0;
  int32_t  k, i, m;
  int32_t  k_min = 0x7fffffff, i_min = 0x7fffffff;
  int32_t  k_max = 0, i_max = 0;
  (void)__dague_object; (void)__foundone;
  int32_t k_start, k_end;  int32_t i_start, i_end;  /* First, find the min and max value for each of the dimensions */
  for(k = 0;
      k <= ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
      k++) {
    assignments[0].value = k;
    for(i = 0;
        i <= zgetrf_param_inline_c_expr32_line_77((const dague_object_t*)__dague_object, assignments);
        i++) {
      assignments[1].value = i;
      m = zgetrf_param_inline_c_expr31_line_78((const dague_object_t*)__dague_object, assignments);
      assignments[2].value = m;
      if( !zgetrf_param_out_pred(k, i, m) ) continue;
      nb_tasks++;
      k_max = dague_imax(k_max, k);
      k_min = dague_imin(k_min, k);
      i_max = dague_imax(i_max, i);
      i_min = dague_imin(i_min, i);
    }
  }

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
  dep = NULL;
  k_start = 0;
  k_end = ((descA.mt < descA.nt) ? (descA.mt - 1) : (descA.nt - 1));
  for(k = dague_imax(k_start, k_min); k <= dague_imin(k_end, k_max); k++) {
    assignments[0].value = k;
    i_start = 0;
    i_end = zgetrf_param_inline_c_expr32_line_77((const dague_object_t*)__dague_object, assignments);
    for(i = dague_imax(i_start, i_min); i <= dague_imin(i_end, i_max); i++) {
      assignments[1].value = i;
      __foundone = 0;
      m = zgetrf_param_inline_c_expr31_line_78((const dague_object_t*)__dague_object, assignments);
      assignments[2].value = m;
      if( zgetrf_param_out_pred(k, i, m) ) {
        /* We did find one! Allocate the dependencies array. */
      if( dep == NULL ) {
        ALLOCATE_DEP_TRACKING(dep, k_min, k_max, "k", &symb_zgetrf_param_zgetrf_param_out_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
      }
      if( dep->u.next[k-k_min] == NULL ) {
        ALLOCATE_DEP_TRACKING(dep->u.next[k-k_min], i_min, i_max, "i", &symb_zgetrf_param_zgetrf_param_out_i, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
      }
      }
    }
  }
  (void)k_start; (void)k_end;  (void)i_start; (void)i_end;  __dague_object->super.super.dependencies_array[6] = dep;
  __dague_object->super.super.nb_local_tasks += nb_tasks;
  return nb_tasks;
}

static const dague_function_t zgetrf_param_zgetrf_param_out = {
  .name = "zgetrf_param_out",
  .deps = 6,
  .flags = 0x0,
  .function_id = 6,
  .dependencies_goal = 0x1,
  .nb_parameters = 2,
  .nb_definitions = 3,
  .params = { &symb_zgetrf_param_zgetrf_param_out_k, &symb_zgetrf_param_zgetrf_param_out_i },
  .locals = { &symb_zgetrf_param_zgetrf_param_out_k, &symb_zgetrf_param_zgetrf_param_out_i, &symb_zgetrf_param_zgetrf_param_out_m },
  .pred = &pred_of_zgetrf_param_zgetrf_param_out_as_expr,
  .priority = NULL,
  .in = { &flow_of_zgetrf_param_zgetrf_param_out_for_A },
  .out = { &flow_of_zgetrf_param_zgetrf_param_out_for_A },
  .iterate_successors = iterate_successors_of_zgetrf_param_zgetrf_param_out,
  .release_deps = release_deps_of_zgetrf_param_zgetrf_param_out,
  .hook = hook_of_zgetrf_param_zgetrf_param_out,
  .complete_execution = complete_hook_of_zgetrf_param_zgetrf_param_out,
#if defined(DAGUE_SIM)
  .sim_cost_fct = NULL,
#endif
  .key = (dague_functionkey_fn_t*)zgetrf_param_out_hash,
};


static const dague_function_t *zgetrf_param_functions[] = {
  &zgetrf_param_zttmqr,
  &zgetrf_param_zttmqr_out,
  &zgetrf_param_zttqrt,
  &zgetrf_param_zttqrt_out_A1,
  &zgetrf_param_zgessm,
  &zgetrf_param_zgetrf_param,
  &zgetrf_param_zgetrf_param_out
};

static void zgetrf_param_startup(dague_context_t *context, dague_object_t *dague_object, dague_execution_context_t** pready_list)
{
  zgetrf_param_zgetrf_param_startup_tasks(context, (__dague_zgetrf_param_internal_object_t*)dague_object, pready_list);
}
#undef descA
#undef A
#undef descL
#undef L
#undef descL2
#undef L2
#undef pivfct
#undef ib
#undef p_work
#undef p_tau
#undef param_p
#undef param_a
#undef param_d
#undef IPIV
#undef INFO
#undef work_pool

dague_zgetrf_param_object_t *dague_zgetrf_param_new(tiled_matrix_desc_t descA, dague_ddesc_t * A /* data A */, tiled_matrix_desc_t descL, dague_ddesc_t * L /* data L */, tiled_matrix_desc_t descL2, dague_ddesc_t * L2, qr_piv_t* pivfct, int ib, dague_memory_pool_t * p_work, dague_memory_pool_t * p_tau, dague_ddesc_t * IPIV /* data IPIV */, int* INFO, dague_memory_pool_t* work_pool)
{
  __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t *)calloc(1, sizeof(__dague_zgetrf_param_internal_object_t));
  int i;
  int zttmqr_nblocal_tasks;
  int zttmqr_out_nblocal_tasks;
  int zttqrt_nblocal_tasks;
  int zttqrt_out_A1_nblocal_tasks;
  int zgessm_nblocal_tasks;
  int zgetrf_param_nblocal_tasks;
  int zgetrf_param_out_nblocal_tasks;

  __dague_object->super.super.nb_functions    = DAGUE_zgetrf_param_NB_FUNCTIONS;
  __dague_object->super.super.functions_array = (const dague_function_t**)malloc(DAGUE_zgetrf_param_NB_FUNCTIONS * sizeof(dague_function_t*));
  __dague_object->super.super.dependencies_array = (dague_dependencies_t **)
              calloc(DAGUE_zgetrf_param_NB_FUNCTIONS, sizeof(dague_dependencies_t *));
  memcpy(__dague_object->super.super.functions_array, zgetrf_param_functions, DAGUE_zgetrf_param_NB_FUNCTIONS * sizeof(dague_function_t*));
  /* Compute the amount of arenas: */
  /*   DAGUE_zgetrf_param_DEFAULT_ARENA  ->  0 */
  /*   DAGUE_zgetrf_param_LOWER_TILE_ARENA  ->  1 */
  /*   DAGUE_zgetrf_param_UPPER_TILE_ARENA  ->  2 */
  /*   DAGUE_zgetrf_param_PIVOT_ARENA  ->  3 */
  /*   DAGUE_zgetrf_param_SMALL_L_ARENA  ->  4 */
  __dague_object->super.arenas_size = 5;
  __dague_object->super.arenas = (dague_arena_t **)malloc(__dague_object->super.arenas_size * sizeof(dague_arena_t*));
  for(i = 0; i < __dague_object->super.arenas_size; i++) {
    __dague_object->super.arenas[i] = (dague_arena_t*)calloc(1, sizeof(dague_arena_t));
  }
  /* Now the Parameter-dependent structures: */
  __dague_object->super.descA = descA;
  __dague_object->super.A = A;
  __dague_object->super.descL = descL;
  __dague_object->super.L = L;
  __dague_object->super.descL2 = descL2;
  __dague_object->super.L2 = L2;
  __dague_object->super.pivfct = pivfct;
  __dague_object->super.ib = ib;
  __dague_object->super.p_work = p_work;
  __dague_object->super.p_tau = p_tau;
  __dague_object->super.param_p = pivfct->p;
  __dague_object->super.param_a = pivfct->a;
  __dague_object->super.param_d = pivfct->domino;
  __dague_object->super.IPIV = IPIV;
  __dague_object->super.INFO = INFO;
  __dague_object->super.work_pool = work_pool;
  /* If profiling is enabled, the keys for profiling */
#  if defined(DAGUE_PROF_TRACE)
  __dague_object->super.super.profiling_array = zgetrf_param_profiling_array;
  if( -1 == zgetrf_param_profiling_array[0] ) {
    dague_profiling_add_dictionary_keyword("zttmqr", "fill:CC2828",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&__dague_object->super.super.profiling_array[0 + 2 * zgetrf_param_zttmqr.function_id /* zttmqr start key */],
                                       (int*)&__dague_object->super.super.profiling_array[1 + 2 * zgetrf_param_zttmqr.function_id /* zttmqr end key */]);
    dague_profiling_add_dictionary_keyword("zttmqr_out", "fill:CCB428",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&__dague_object->super.super.profiling_array[0 + 2 * zgetrf_param_zttmqr_out.function_id /* zttmqr_out start key */],
                                       (int*)&__dague_object->super.super.profiling_array[1 + 2 * zgetrf_param_zttmqr_out.function_id /* zttmqr_out end key */]);
    dague_profiling_add_dictionary_keyword("zttqrt", "fill:57CC28",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&__dague_object->super.super.profiling_array[0 + 2 * zgetrf_param_zttqrt.function_id /* zttqrt start key */],
                                       (int*)&__dague_object->super.super.profiling_array[1 + 2 * zgetrf_param_zttqrt.function_id /* zttqrt end key */]);
    dague_profiling_add_dictionary_keyword("zttqrt_out_A1", "fill:28CC86",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&__dague_object->super.super.profiling_array[0 + 2 * zgetrf_param_zttqrt_out_A1.function_id /* zttqrt_out_A1 start key */],
                                       (int*)&__dague_object->super.super.profiling_array[1 + 2 * zgetrf_param_zttqrt_out_A1.function_id /* zttqrt_out_A1 end key */]);
    dague_profiling_add_dictionary_keyword("zgessm", "fill:2886CC",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&__dague_object->super.super.profiling_array[0 + 2 * zgetrf_param_zgessm.function_id /* zgessm start key */],
                                       (int*)&__dague_object->super.super.profiling_array[1 + 2 * zgetrf_param_zgessm.function_id /* zgessm end key */]);
    dague_profiling_add_dictionary_keyword("zgetrf_param", "fill:5728CC",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&__dague_object->super.super.profiling_array[0 + 2 * zgetrf_param_zgetrf_param.function_id /* zgetrf_param start key */],
                                       (int*)&__dague_object->super.super.profiling_array[1 + 2 * zgetrf_param_zgetrf_param.function_id /* zgetrf_param end key */]);
    dague_profiling_add_dictionary_keyword("zgetrf_param_out", "fill:CC28B4",
                                       sizeof(dague_profile_ddesc_info_t), dague_profile_ddesc_key_to_string,
                                       (int*)&__dague_object->super.super.profiling_array[0 + 2 * zgetrf_param_zgetrf_param_out.function_id /* zgetrf_param_out start key */],
                                       (int*)&__dague_object->super.super.profiling_array[1 + 2 * zgetrf_param_zgetrf_param_out.function_id /* zgetrf_param_out end key */]);
  }
#  endif /* defined(DAGUE_PROF_TRACE) */
  /* Create the data repositories for this object */
  zttmqr_nblocal_tasks = zgetrf_param_zttmqr_internal_init(__dague_object);
  if( 0 == zttmqr_nblocal_tasks ) zttmqr_nblocal_tasks = 10;
  __dague_object->zttmqr_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(zttmqr_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(zttmqr_nblocal_tasks * 1.5)), 5);

  zttmqr_out_nblocal_tasks = zgetrf_param_zttmqr_out_internal_init(__dague_object);
  if( 0 == zttmqr_out_nblocal_tasks ) zttmqr_out_nblocal_tasks = 10;
  __dague_object->zttmqr_out_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(zttmqr_out_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(zttmqr_out_nblocal_tasks * 1.5)), 1);

  zttqrt_nblocal_tasks = zgetrf_param_zttqrt_internal_init(__dague_object);
  if( 0 == zttqrt_nblocal_tasks ) zttqrt_nblocal_tasks = 10;
  __dague_object->zttqrt_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(zttqrt_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(zttqrt_nblocal_tasks * 1.5)), 4);

  zttqrt_out_A1_nblocal_tasks = zgetrf_param_zttqrt_out_A1_internal_init(__dague_object);
  if( 0 == zttqrt_out_A1_nblocal_tasks ) zttqrt_out_A1_nblocal_tasks = 10;
  __dague_object->zttqrt_out_A1_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(zttqrt_out_A1_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(zttqrt_out_A1_nblocal_tasks * 1.5)), 1);

  zgessm_nblocal_tasks = zgetrf_param_zgessm_internal_init(__dague_object);
  if( 0 == zgessm_nblocal_tasks ) zgessm_nblocal_tasks = 10;
  __dague_object->zgessm_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(zgessm_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(zgessm_nblocal_tasks * 1.5)), 3);

  zgetrf_param_nblocal_tasks = zgetrf_param_zgetrf_param_internal_init(__dague_object);
  if( 0 == zgetrf_param_nblocal_tasks ) zgetrf_param_nblocal_tasks = 10;
  __dague_object->zgetrf_param_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(zgetrf_param_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(zgetrf_param_nblocal_tasks * 1.5)), 2);

  zgetrf_param_out_nblocal_tasks = zgetrf_param_zgetrf_param_out_internal_init(__dague_object);
  if( 0 == zgetrf_param_out_nblocal_tasks ) zgetrf_param_out_nblocal_tasks = 10;
  __dague_object->zgetrf_param_out_repository = data_repo_create_nothreadsafe(
          ((unsigned int)(zgetrf_param_out_nblocal_tasks * 1.5)) > MAX_DATAREPO_HASH ?
          MAX_DATAREPO_HASH :
          ((unsigned int)(zgetrf_param_out_nblocal_tasks * 1.5)), 1);

  __dague_object->super.super.startup_hook = zgetrf_param_startup;
  (void)dague_object_register((dague_object_t*)__dague_object);
  return (dague_zgetrf_param_object_t*)__dague_object;
}

void dague_zgetrf_param_destroy( dague_zgetrf_param_object_t *o )
{
  dague_object_t *d = (dague_object_t *)o;
  __dague_zgetrf_param_internal_object_t *__dague_object = (__dague_zgetrf_param_internal_object_t*)o; (void)__dague_object;
  int i;
  free(d->functions_array);
  d->functions_array = NULL;
  d->nb_functions = 0;
  for(i =0; i < o->arenas_size; i++) {
    if( o->arenas[i] != NULL ) {
      dague_arena_destruct(o->arenas[i]);
      free(o->arenas[i]);
    }
  }
  free( o->arenas );
  o->arenas = NULL;
  o->arenas_size = 0;
  /* Destroy the data repositories for this object */
   data_repo_destroy_nothreadsafe(__dague_object->zttmqr_repository);
   data_repo_destroy_nothreadsafe(__dague_object->zttmqr_out_repository);
   data_repo_destroy_nothreadsafe(__dague_object->zttqrt_repository);
   data_repo_destroy_nothreadsafe(__dague_object->zttqrt_out_A1_repository);
   data_repo_destroy_nothreadsafe(__dague_object->zgessm_repository);
   data_repo_destroy_nothreadsafe(__dague_object->zgetrf_param_repository);
   data_repo_destroy_nothreadsafe(__dague_object->zgetrf_param_out_repository);
  for(i = 0; i < DAGUE_zgetrf_param_NB_FUNCTIONS; i++)
    dague_destruct_dependencies( d->dependencies_array[i] );
  free( d->dependencies_array );
  free(o);
}

