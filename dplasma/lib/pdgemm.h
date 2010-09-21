/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


/**
 *  Purpose
 *    http://www.netlib.org/blas/dgemm.f
 *  =======
 *
 *  DGEMM  performs one of the matrix-matrix operations
 *
 *     C := alpha*op( A )*op( B ) + beta*C,
 *
 *  where  op( X ) is one of
 *
 *     op( X ) = X   or   op( X ) = X',
 *
 *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 *
 *  Arguments
 *  ==========
 *
 *  TRANSA - CHARACTER*1.
 *           On entry, TRANSA specifies the form of op( A ) to be used in
 *           the matrix multiplication as follows:
 *
 *              TRANSA = 'N' or 'n',  op( A ) = A.
 *
 *              TRANSA = 'T' or 't',  op( A ) = A'.
 *
 *              TRANSA = 'C' or 'c',  op( A ) = A'.
 *
 *           Unchanged on exit.
 *
 *  TRANSB - CHARACTER*1.
 *           On entry, TRANSB specifies the form of op( B ) to be used in
 *           the matrix multiplication as follows:
 *
 *              TRANSB = 'N' or 'n',  op( B ) = B.
 *
 *              TRANSB = 'T' or 't',  op( B ) = B'.
 *
 *              TRANSB = 'C' or 'c',  op( B ) = B'.
 *
 *           Unchanged on exit.
 *
 *  M      - INTEGER.
 *           On entry,  M  specifies  the number  of rows  of the  matrix
 *           op( A )  and of the  matrix  C.  M  must  be at least  zero.
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry,  N  specifies the number  of columns of the matrix
 *           op( B ) and the number of columns of the matrix C. N must be
 *           at least zero.
 *           Unchanged on exit.
 *
 *  K      - INTEGER.
 *           On entry,  K  specifies  the number of columns of the matrix
 *           op( A ) and the number of rows of the matrix op( B ). K must
 *           be at least  zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
 *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
 *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
 *           part of the array  A  must contain the matrix  A,  otherwise
 *           the leading  k by m  part of the array  A  must contain  the
 *           matrix A.
 *           Unchanged on exit.
 *
 *  B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
 *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
 *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
 *           part of the array  B  must contain the matrix  B,  otherwise
 *           the leading  n by k  part of the array  B  must contain  the
 *           matrix B.
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
 *           supplied as zero then C need not be set on input.
 *           Unchanged on exit.
 *
 *  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
 *           Before entry, the leading  m by n  part of the array  C must
 *           contain the matrix  C,  except when  beta  is zero, in which
 *           case C need not be set on entry.
 *           On exit, the array  C  is overwritten by the  m by n  matrix
 *           ( alpha*op( A )*op( B ) + beta*C ).
 *
 *  Level 3 Blas routine.
 */

extern dague_object_t*
dague_pdgemm_new( int TRANSA, int TRANSB,
                  int M, int N, int K,
                  float ALPHA, const tiled_matrix_desc_t* A,
                  const tiled_matrix_desc_t* B,
                  float BETA, tiled_matrix_desc_t* C );

