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
                                 cuDoubleComplex alpha, cuDoubleComplex *d_A, int lda,
                                                        cuDoubleComplex *d_B, int ldb,
                                 cuDoubleComplex beta,  cuDoubleComplex *d_C, int ldc,
                                 CUstream stream )
{
#if (__CUDA_API_VERSION < 4000)
    
    cublasSetKernelStream( stream );

    cublasZgemm(TRANSA, TRANSB, m, n, k, 
                alpha, d_A, lda,
                        d_B, ldb,
                beta,  d_C, ldc); 

#else
    cudaStream_t current_stream;
    cublasHandle_t handle = cublasGetCurrentCtx();
    
    cublasGetStream_v2 ( handle, &saved_stream );
    cublasSetStream_v2 ( handle, &stream );

    cublasZgemm_v2(handle, convertToOp(TRANSA), convertToOp(TRANSB),
                   m, n, k, 
                   &alpha, d_A, lda,
                           d_B, ldb,
                   &beta,  d_C, ldc); 

    cublasSetStream_v2 ( handle, &saved_stream );
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////
#endif /* (CUDA_SM_VERSION == 11) || (CUDA_SM_VERSION == 12) || (CUDA_SM_VERSION == 13) */

