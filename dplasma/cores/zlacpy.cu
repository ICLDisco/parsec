/*
       @precisions normal z -> s d c
*/
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include "data_dist/matrix/precision.h"

#if defined(PRECISION_z) || defined(PRECISION_c)
#include <cuComplex.h>
#endif  /* defined(PRECISION_z) || defined(PRECISION_c) */


#define     max(a, b)   ((a) > (b) ? (a) : (b))
#define NB 64

/* =====================================================================
    Matrix is m x n, and is divided into block rows, each NB x n.
    Each CUDA block has NB threads to handle one block row.
    Each thread copies one row, iterating across all columns.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
*/
__global__ void
zlacpy_kernel(
    int m, int n,
    const dague_complex64_t *dA, int ldda, dague_complex64_t alpha,
    dague_complex64_t       *dB, int lddb)
{
    // dA and dB iterate across row i
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i < m ) {
        dA += i;
        dB += i;
        const dague_complex64_t *dAend = dA + n*ldda;
        while( dA < dAend ) {
            *dB = (*dA)*alpha + *dB;
            dA += ldda;
            dB += lddb;
        }
    }
}


/* ===================================================================== */
extern "C" void
magmablas_zlacpy(
    char uplo, int m, int n,
    const dague_complex64_t *dA, int ldda, double alpha,
    dague_complex64_t       *dB, int lddb, CUstream stream )
{
/*  
    Purpose
    =======
    ZLACPY copies all or part of a two-dimensional matrix dA to another
    matrix dB.
    
    Arguments
    =========
    
    M       (input) INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    N       (input) INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    dA      (input) COMPLEX DOUBLE PRECISION array, dimension (LDDA,N)
            The m by n matrix dA.
            If UPLO = 'U', only the upper triangle or trapezoid is accessed;
            if UPLO = 'L', only the lower triangle or trapezoid is accessed.
    
    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    dB      (output) COMPLEX DOUBLE PRECISION array, dimension (LDDB,N)
            The m by n matrix dB.
            On exit, dB = dA in the locations specified by UPLO.
    
    LDDB    (input) INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).
    
    =====================================================================   */

    int info = 0;
    if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;
    
    if ( info != 0 ) {
        return;
    }
    
    if ( m == 0 || n == 0 )
        return;
    
    dim3 threads( NB );
    dim3 grid( (m + NB - 1)/NB );
    
    if ( (uplo == 'U') || (uplo == 'u') ) {
        fprintf(stderr, "lacpy upper is not implemented\n");
    }
    else if ( (uplo == 'L') || (uplo == 'l') ) {
        fprintf(stderr, "lacpy lower is not implemented\n");
    }
    else {
        zlacpy_kernel<<< grid, threads, 0, stream >>>(
            m, n, dA, ldda, alpha, dB, lddb );
    }
}
