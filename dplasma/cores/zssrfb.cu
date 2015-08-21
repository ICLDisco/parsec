/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
       @precisions normal z -> s d c
*/

#include <cublas.h>
#include <cuda.h>
#include <stdio.h>
#include "data_dist/matrix/precision.h"

#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#include <cuComplex.h>
#endif  /* defined(PRECISION_z) || defined(PRECISION_c) */

///////////////////////////////////////////////////////////////////////////////////////////////////

#define GENERATE_SM_VERSION_KERNEL_NAME_I(func, version)  zgemm_##func##_SM##version
#define GENERATE_SM_VERSION_KERNEL_NAME_I2(func, version) GENERATE_SM_VERSION_KERNEL_NAME_I(func, version)
#define GENERATE_SM_VERSION_KERNEL_NAME(func)             GENERATE_SM_VERSION_KERNEL_NAME_I2(func, CUDA_SM_VERSION)

#define GENERATE_SM_VERSION_NAME_I(func, version)  magmablas_##func##_SM##version
#define GENERATE_SM_VERSION_NAME_I2(func, version) GENERATE_SM_VERSION_NAME_I(func, version)
#define GENERATE_SM_VERSION_NAME(func)             GENERATE_SM_VERSION_NAME_I2(func, CUDA_SM_VERSION)

///////////////////////////////////////////////////////////////////////////////////////////////////


extern "C" void
magmablas_zgemm_kernel_T_N_32_32_8_8_8_ssrfb2(int k, int n, int m,
					       dague_complex64_t *dt, int ldt,
					       dague_complex64_t *dwork, int ldwork,
					       dague_complex64_t *dwork1, int ldwork1,
					       dague_complex64_t *da1  , int lda1, CUstream stream);
extern "C" void
magmablas_zgemm_kernel_T_N_32_32_8_8_8_ssrfb(int k, int n, int m,
					      dague_complex64_t *v_ref, int ldv,
					      dague_complex64_t *da2_ref, int lda2,
					      dague_complex64_t *dwork, int ldwork,
					      dague_complex64_t *da1  , int lda1, CUstream stream);

extern "C" int
GENERATE_SM_VERSION_NAME(ZSSRFB)(int m, int n, int *k, dague_complex64_t *dv, int *ldv,
		dague_complex64_t *dt, int *ldt,
		dague_complex64_t *da1, int *lda1, dague_complex64_t *da2, int *lda2,
		dague_complex64_t *dwork, int *ldwork, CUstream stream)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Univ. of California Berkeley
       November 2009

    Purpose
    =======

    SSSRFB applies a real block reflector H or its transpose H' to a
    k+m by n matrix /A1\ , from the left.
		    \A2/
    A1 is k by n, A2 m by n, and H =   I - / I \  T  / I \' .
					   \ V /     \ V /

    Arguments
    =========

    M       (input) INTEGER
	    The number of rows of the matrix C.

    N       (input) INTEGER
	    The number of columns of the matrix C.

    K       (input) INTEGER
	    The order of the matrix T (= the number of elementary
	    reflectors whose product defines the block reflector).

    V       (input) DOUBLE REAL array, dimension (LDV,K)
	    The matrix V. See further details.

    LDV     (input) INTEGER
	    The leading dimension of the array V. LDV >= max(1,M);

    T       (input) DOUBLE REAL array, dimension (LDT,K)
	    The triangular k by k matrix T in the representation of the
	    block reflector.

    LDT     (input) INTEGER
	    The leading dimension of the array T. LDT >= K.

    DA1     (input/output) DOUBLE REAL array, dimension (LDA1,N)
	    On entry, the k by n matrix A1.
	    On exit, /A1\ is overwritten by H /A1\
		     \A2/                     \A2/.

    LDA2    (input) INTEGER
	    The leading dimension of the array DA1. LDA1 >= max(1,K).

    DA2     (input/output) DOUBLE REAL array, dimension (LDA2,N)
	    On entry, the m by n matrix A2.
	    On exit, /A1\ is overwritten by H /A1\
		     \A2/                     \A2/.

    LDA2    (input) INTEGER
	    The leading dimension of the array A2. LDA2 >= max(1,M).

    WORK    (workspace) DOUBLE REAL array, dimension (LDWORK,N)

    LDWORK  (input) INTEGER
	    The leading dimension of the array WORK. LDWORK >= max(1,2*K);

    ===================================================================      */
#if defined(PRECISION_z) || defined(PRECISION_c)
  cuDoubleComplex mone = make_cuDoubleComplex(-1., 0.),
                  one = make_cuDoubleComplex(1., 0.);
#else
  double mone = -1.,
         one = 1.;
#endif

  #define dwork_ref(a_1,a_2) (dwork+(a_2)*(*ldwork) + a_1)
  #define da2_ref(a_1,a_2)   (da2+(a_2)*(*lda2) + a_1)
  #define dv_ref(a_1,a_2)    (dv+(a_2)*(*ldv) + a_1)

  /* Function Body */
  if (m <= 0 || n <= 0) {
    return 0;
  }

  /* 1. dwork = A1 where A1 is of dimension k by n */
  //cudaMemcpy2D(dwork, (*ldwork) * sizeof(double),
  //	       da1  , (*lda1)   * sizeof(double),
  //	       sizeof(double)*(*k), n,
  //	       cudaMemcpyDeviceToDevice);

  /* 2. dwork = dwork + V' A2.                     */
  //cublasDgemm('t', 'n', *k, n, m, 1.f, dv_ref(0, 0), *ldv,
  //            da2_ref(0,0), *lda2, 1.f, dwork, *ldwork);


  // TimeStruct start, end;
  // start = get_current_time();
  magmablas_zgemm_kernel_T_N_32_32_8_8_8_ssrfb(*k, n, m,
						dv_ref(0, 0), *ldv,
						da2_ref(0,0), *lda2,
						dwork, *ldwork,
						da1  , *lda1, stream);
  // end = get_current_time();
  // printf("%5.2f ", 2.(*k)*n*m/(1000000.*GetTimerValue(start,end)));

  /* 3. (dwork+k) = T dwork
	T is triangular, assumed to have 0s in the unused part */
  //cublasDgemm('t', 'n', *k, n, *k, 1.f, dt, *ldt, dwork, *ldwork,
  //	      0.f, dwork+(*k), *ldwork);

  /* 4. A1 = A1 - (dwork+k)                        */
  //for(int i=0; i<n; i++)
  //  cublasDaxpy(*k, -1.f, dwork+(*k) + i*(*ldwork), 1, da1+i*(*lda1), 1);

  // start = get_current_time();
  magmablas_zgemm_kernel_T_N_32_32_8_8_8_ssrfb2(*k, n, *k,
						 dt, *ldt,
						 dwork, *ldwork,
						 dwork+(*k), *ldwork,
						 da1  , *lda1, stream);
  // end = get_current_time();
  // printf("%5.2f ", 2.*(*k)*(*k)*n/(1000000.*GetTimerValue(start,end)));


  /* 5. A2 = A2 - V (dwork+k) */
  // start = get_current_time();
  cublasSetKernelStream( stream );
  cublasZgemm('n', 'n', m, n, *k, mone, (cuDoubleComplex*)dv_ref(0, 0), *ldv,
	      (cuDoubleComplex*)dwork+(*k), *ldwork, one, (cuDoubleComplex*)da2_ref(0,0), *lda2);
  // end = get_current_time();
  // printf("%5.2f \n", 2.*m*n*(*k)/(1000000.*GetTimerValue(start,end)));

  return 0;

} /* magma_dssrfb */

#undef dv_ref
#undef da2_ref
#undef dwork_ref
