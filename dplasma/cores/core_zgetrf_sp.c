/**
 *
 * @file core_zgetrf_sp.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 1.0.0
 * @author Mathieu Faverge
 * @author Pierre Ramet
 * @author Xavier Lacoste
 * @author Omar Zenati
 * @date 2011-11-11
 * @precisions normal z -> s d c
 *
 **/
#include <plasma.h>
#include <core_blas.h>
#include <math.h>

#define min( _a, _b ) ( (_a) < (_b) ? (_a) : (_b) )

/*
  Constant: MAXSIZEOFBLOCKS
  Maximum size of blocks given to blas in factorisation
*/

#define MAXSIZEOFBLOCKS 16 /*64 in LAPACK*/

static PLASMA_Complex64_t zone  = 1.;
static PLASMA_Complex64_t mzone = -1.;

/* 
   Function: FactorizationLU
   
   LU Factorisation of one (diagonal) block 
   $A = LU$

   For each column : 
   - Divide the column by the diagonal element.
   - Substract the product of the subdiagonal part by
   the line after the diagonal element from the 
   matrix under the diagonal element.

   Parameters: 
   A       - Matrix to factorize
   m       - number of rows of the Matrix A
   n       - number of cols of the Matrix A
   stride  - Stide between 2 columns of the matrix
   nbpivot - IN/OUT pivot number.
   critere - Pivoting threshold.
*/
static void CORE_zgetf2_sp(int  m, 
                    int  n, 
                    PLASMA_Complex64_t * A, 
                    int  stride, 
                    double criteria, 
                    int *nbpivot )
{
  int k, minMN;
  PLASMA_Complex64_t *Akk, *Aik, alpha;

  minMN = min( m, n );

  Akk = A;
  for (k=0; k<minMN; k++) {
    Aik = Akk + 1;

    if ( fabs(*Akk) < criteria ) {
      (*Akk) = (PLASMA_Complex64_t)criteria;
      (*nbpivot)++;
    }

    /* A_ik = A_ik / A_kk, i = k+1 .. n */
    alpha = 1. / (*Akk);
    cblas_zscal(m-k-1, CBLAS_SADDR( alpha ), Aik, 1 );

    if ( k+1 < minMN ) {

      /* A_ij = A_ij - A_ik * A_kj, i,j = k+1..n */
      cblas_zgeru(CblasColMajor, m-k-1, n-k-1, 
		  CBLAS_SADDR(mzone), 
		  Aik,        1, 
		  Akk+stride, stride, 
		  Aik+stride, stride);
    }

    Akk += stride+1;
  }
}

/* 
   Function: FactorizationLU_block
   
   Block LU Factorisation of one (diagonal) big block 
   > A = LU

   Parameters: 
   A       - Matrix to factorize
   n       - Size of A
   stride  - Stride between 2 columns of the matrix
   nbpivot - IN/OUT pivot number.
   critere - Pivoting threshold.
*/
void CORE_zgetrf_sp(int m, int  n, 
                    PLASMA_Complex64_t *A, 
                    int  stride, 
                    double  criteria,
                    int *nbpivot)
{
  int k, blocknbr, blocksize, u_size, l_size, tempm, tempn;
  PLASMA_Complex64_t *Akk, *Lik, *Ukj, *Aij;

  blocknbr = (int) ceil( (double)min(m,n)/(double)MAXSIZEOFBLOCKS );

  Akk = A; /* Lk,k     */

  for (k=0; k<blocknbr; k++) {
      
    tempm = m - k * MAXSIZEOFBLOCKS;
    tempn = n - k * MAXSIZEOFBLOCKS;

    blocksize = min(tempm, tempn);
    blocksize = min(MAXSIZEOFBLOCKS, blocksize);

    Lik = Akk + blocksize;
    Ukj = Akk + blocksize*stride;
    Aij = Ukj + blocksize;
    
    /* Factorize the diagonal block Akk*/
    CORE_zgetf2_sp( blocksize, blocksize, Akk, stride, criteria, nbpivot );
    
    u_size = tempn - blocksize;
    l_size = tempm - blocksize;
    if ( u_size > 0 && l_size > 0) {

      /* Compute the column Ukk+1 */
      cblas_ztrsm(CblasColMajor,
      		  (CBLAS_SIDE)CblasLeft, (CBLAS_UPLO)CblasLower,
      		  (CBLAS_TRANSPOSE)CblasNoTrans, (CBLAS_DIAG)CblasUnit,
      		  blocksize, u_size,
      		  CBLAS_SADDR(zone), Akk, stride,
      		  Ukj, stride);

     /* Compute the column Lk+1k */ 
      cblas_ztrsm(CblasColMajor,
      		  (CBLAS_SIDE)CblasRight, (CBLAS_UPLO)CblasUpper,
      		  (CBLAS_TRANSPOSE)CblasNoTrans, (CBLAS_DIAG)CblasNonUnit,
      		  l_size, blocksize, 
      		  CBLAS_SADDR(zone), Akk, stride,
      		  Lik, stride);

      /* Update Ak+1,k+1 = Ak+1,k+1 - Lk+1,k*Uk,k+1 */
      cblas_zgemm(CblasColMajor,
      		  (CBLAS_TRANSPOSE)CblasNoTrans, (CBLAS_TRANSPOSE)CblasNoTrans,
      		  l_size, u_size, blocksize,
      		  CBLAS_SADDR(mzone), Lik, stride,
      		  Ukj, stride,
      		  CBLAS_SADDR(zone),  Aij, stride);

    }

    Akk += blocksize * (stride+1);
  }
}

void CORE_zgetrf_sp_rec(int m, int  n, 
                        PLASMA_Complex64_t *A, 
                        int  stride,
                        double criteria,
                        int *nbpivot)
{
  if(m > 0)
    {
      if(n == 1)
        {
          if ( fabs(*A) < criteria ) 
            {
              (*A) = (PLASMA_Complex64_t)criteria;
              (*nbpivot)++;
            }
          PLASMA_Complex64_t alpha = 1. / (*A);
          cblas_zscal(m-1, CBLAS_SADDR( alpha ), A+1, 1);
        }
      else
        {
          CORE_zgetrf_sp_rec(m, n/2, A, stride, criteria, nbpivot);

          cblas_ztrsm(CblasColMajor,
                      (CBLAS_SIDE)CblasLeft, (CBLAS_UPLO)CblasLower,
                      (CBLAS_TRANSPOSE)CblasNoTrans, (CBLAS_DIAG)CblasUnit,
                      n/2, n/2+n%2,
                      CBLAS_SADDR(zone), A, stride,
                      A+(n/2)*stride, stride);

          cblas_zgemm(CblasColMajor,
                      (CBLAS_TRANSPOSE)CblasNoTrans, (CBLAS_TRANSPOSE)CblasNoTrans,
                      m-n/2, n/2+n%2, n/2,
                      CBLAS_SADDR(mzone), A+n/2,              stride,
                                          A+(n/2)*stride,     stride,
                      CBLAS_SADDR(zone),  A+(n/2)*stride+n/2, stride);

          CORE_zgetrf_sp_rec(m-n/2, n/2+n%2, A+((n/2)*stride)+(n/2), stride, criteria, nbpivot);
        }
    }
}

