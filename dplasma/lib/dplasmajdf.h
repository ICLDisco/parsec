#ifndef _DPLASMAJDF_H_
#define _DPLASMAJDF_H_

#include <core_blas.h>
#include "dague.h"
#include "dplasma.h"
#include "dague/private_mempool.h"

/* Check for LU recursive kernel version */
#if (PLASMA_VERSION_MAJOR < 2) || ((PLASMA_VERSION_MAJOR == 2) && (PLASMA_VERSION_MINOR < 8))
#warning "Please update your PLASMA library to 2.8.0 or higher"
#define CORE_GETRF_270
typedef void * CORE_zgetrf_data_t;
typedef void * CORE_cgetrf_data_t;
typedef void * CORE_dgetrf_data_t;
typedef void * CORE_sgetrf_data_t;
#endif

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

#define plasma_const( x )  plasma_lapack_constants[x]

#ifdef DAGUE_CALL_TRACE
#   include <stdlib.h>
#   include <stdio.h>
#   define printlog(str, ...) fprintf(stderr, "thread %d VP %d " str "\n", \
                                      context->th_id, context->virtual_process->vp_id, __VA_ARGS__)
#   define OUTPUT(ARG)  printf ARG
#else
#   define printlog(...) do {} while(0)
#   define OUTPUT(ARG)
#endif

#ifdef DAGUE_DRY_RUN
#define DRYRUN( body )
#else
#define DRYRUN( body ) body
#endif

#ifndef HAVE_MPI
#define TEMP_TYPE MPITYPE
#undef MPITYPE
#define MPITYPE ((dague_datatype_t)QUOTEME(TEMP_TYPE))
#undef TEMP_TYPE
#endif  /* HAVE_MPI */


#if defined(HAVE_CUDA)
#include <cublas.h>

typedef void (*cublas_zgemm_t) ( char TRANSA, char TRANSB, int m, int n, int k,
                                 cuDoubleComplex alpha, cuDoubleComplex *d_A, int lda,
                                 cuDoubleComplex *d_B, int ldb,
                                 cuDoubleComplex beta,  cuDoubleComplex *d_C, int ldc );
typedef void (*cublas_cgemm_t) ( char TRANSA, char TRANSB, int m, int n, int k,
                                 cuComplex alpha, cuComplex *d_A, int lda,
                                 cuComplex *d_B, int ldb,
                                 cuComplex beta,  cuComplex *d_C, int ldc );
typedef void (*cublas_dgemm_t) ( char TRANSA, char TRANSB, int m, int n, int k,
                                 double alpha, double *d_A, int lda,
                                 double *d_B, int ldb,
                                 double beta,  double *d_C, int ldc );
typedef void (*cublas_sgemm_t) ( char TRANSA, char TRANSB, int m, int n, int k,
                                 float alpha, float *d_A, int lda,
                                              float *d_B, int ldb,
                                 float beta,  float *d_C, int ldc );
#endif

#endif /* _DPLASMAJDF_H_ */

