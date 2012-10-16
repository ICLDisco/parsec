/*
  -- MAGMA (version 1.1) --
  Univ. of Tennessee, Knoxville
  Univ. of California, Berkeley
  Univ. of Colorado, Denver
  November 2011


  @precisions normal z -> z c d s
       
*/

#if (CUDA_SM_VERSION == 11) || (CUDA_SM_VERSION == 12) || (CUDA_SM_VERSION == 13)

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cublas.h>

#include "dague.h"
#include "data_dist/matrix/precision.h"

#define PRECISION_z

#if defined(PRECISION_z) || defined(PRECISION_c) 
#include <cuComplex.h>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

#define GENERATE_SM_VERSION_NAME_I(func, version)  magmablas_##func##_SM##version
#define GENERATE_SM_VERSION_NAME_I2(func, version) GENERATE_SM_VERSION_NAME_I(func, version)
#define GENERATE_SM_VERSION_NAME(func)             GENERATE_SM_VERSION_NAME_I2(func, CUDA_SM_VERSION)

///////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" void
GENERATE_SM_VERSION_NAME(zgemm)( char TRANSA, char TRANSB, int m, int n, int k,
                                 dague_complex64_t alpha, dague_complex64_t *d_A, int lda,
                                                          dague_complex64_t *d_B, int ldb,
                                 dague_complex64_t beta,  dague_complex64_t *d_C, int ldc,
                                 CUstream stream )
{
#if defined(PRECISION_z) || defined(PRECISION_c)    
    cuDoubleComplex lalpha = make_cuDoubleComplex( creal(alpha), cimag(alpha) );
    cuDoubleComplex lbeta  = make_cuDoubleComplex( creal(beta),  cimag(beta)  );
#else
    double lalpha = alpha;
    double lbeta  = beta;
#endif

#if (__CUDA_API_VERSION < 4000)
    
    cublasSetKernelStream( stream );

    cublasZgemm(TRANSA, TRANSB, m, n, k, 
                lalpha, (cuDoubleComplex*)d_A, lda,
                        (cuDoubleComplex*)d_B, ldb,
                lbeta,  (cuDoubleComplex*)d_C, ldc); 
    assert( CUBLAS_STATUS_SUCCESS == cublasGetError() );

#else
    cudaStream_t current_stream;
    cublasHandle_t handle = cublasGetCurrentCtx();
    
    cublasGetStream_v2 ( handle, &saved_stream );
    cublasSetStream_v2 ( handle, &stream );

    cublasZgemm_v2(handle, convertToOp(TRANSA), convertToOp(TRANSB),
                   m, n, k, 
                   &lalpha, (cuDoubleComplex*)d_A, lda,
                            (cuDoubleComplex*)d_B, ldb,
                   &lbeta,  (cuDoubleComplex*)d_C, ldc); 
    assert( CUBLAS_STATUS_SUCCESS == cublasGetError() );

    cublasSetStream_v2 ( handle, &saved_stream );
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////
#endif /* (CUDA_SM_VERSION == 11) || (CUDA_SM_VERSION == 12) || (CUDA_SM_VERSION == 13) */

