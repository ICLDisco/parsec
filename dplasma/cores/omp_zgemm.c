/*
 * @precisions normal z -> c d s
 */
#include "dplasma.h"
#include "dplasma_cores.h"
#include "dplasma_zcores.h"
#include <cblas.h>
#include <omp.h>

void CORE_omp_zgemm(PLASMA_enum transA, int transB,
                int M, int N, int K,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int LDA,
                                          const PLASMA_Complex64_t *B, int LDB,
                PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int LDC, int device_index, int *device_stream) {
//map(to: transA, transB, M, N, K, alpha, beta, LDA, LDB, LDC)
#pragma omp target nowait device(device_index) depend(out: device_stream[0])  is_device_ptr(A, B, C) 
{
#pragma omp declare target
void cblas_zgemm(const enum CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);
#pragma omp end declare target
    
    printf("Gemm %d %d %d A=%p, B=%p, C=%p device=%s event=%p (%d)\n", M, N, K, A, B, C, omp_is_initial_device()? "host": "offloaded", device_stream, device_stream[0]);
    cblas_zgemm(
        CblasColMajor,
        (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
        M, N, K,
        &alpha, A, LDA,
        B, LDB,
        &beta, C, LDC);
}
//#pragma omp taskwait
}

