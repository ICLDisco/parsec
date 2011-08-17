/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

static __device__ void saxpy( float a, float *b, float *c )
{
	c[0] += a*b[0];
	c[1] += a*b[1];
	c[2] += a*b[2];
	c[3] += a*b[3];
	c[4] += a*b[4];
	c[5] += a*b[5];
	c[6] += a*b[6];
	c[7] += a*b[7];
	c[8] += a*b[8];
	c[9] += a*b[9];
	c[10] += a*b[10];
	c[11] += a*b[11];
	c[12] += a*b[12];
	c[13] += a*b[13];
	c[14] += a*b[14];
	c[15] += a*b[15];
}

#if (CUDA_SM_VERSION != 11) && (CUDA_SM_VERSION != 12) && (CUDA_SM_VERSION != 13)
  #error "CUDA_SM_VERSION must be defined to 11, 12 or 13 for this kernel"
#endif
#define GENERATE_SM_VERSION_NAME_I(func, version) func##_SM##version
#define GENERATE_SM_VERSION_NAME_I2(func, version) GENERATE_SM_VERSION_NAME_I(func, version)
#define GENERATE_SM_VERSION_NAME(func) GENERATE_SM_VERSION_NAME_I2(func, CUDA_SM_VERSION)

extern "C" __global__ void GENERATE_SM_VERSION_NAME(sgemmNT) ( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
	const int inx = threadIdx.x;
	const int iny = threadIdx.y;
	const int ibx = blockIdx.x * 64;
	const int iby = blockIdx.y * 16;
	const int id  = inx + iny*16;

	A += ibx + id;
	B += iby + inx + __mul24( iny, ldb );
	C += ibx + id  + __mul24( iby, ldc );
	
	float a[4] = {A[0], A[lda], A[2*lda], A[3*lda]};
	float b = B[0];
	
	const float *Blast = B + k*ldb;

	A += 4*lda;
	B += 4*ldb;
    
	__shared__ float bs[4][16];
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
	do
        {
            float as[4] = {a[0], a[1], a[2], a[3]};
		
            bs[iny][inx] = b;
            __syncthreads();
		
            a[0] = A[0*lda];
            a[1] = A[1*lda];
            a[2] = A[2*lda];
            a[3] = A[3*lda];
            b    = B[0];
		
            saxpy( as[0], &bs[0][0], c );
            saxpy( as[1], &bs[1][0], c );
            saxpy( as[2], &bs[2][0], c );
            saxpy( as[3], &bs[3][0], c );
		
            A += 4*lda;
            B += 4*ldb;
            __syncthreads();
		
        } while( B < Blast );
	
	bs[iny][inx] = b;
	__syncthreads();
	
	saxpy( a[0], &bs[0][0], c );
	saxpy( a[1], &bs[1][0], c );
	saxpy( a[2], &bs[2][0], c );
	saxpy( a[3], &bs[3][0], c );

	for( int i = 0; i < 16; i++, C += ldc )
		C[0] = alpha*c[i] + beta*C[0];
}	

extern "C" __global__ void GENERATE_SM_VERSION_NAME(sgemmNN) ( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
	const int inx = threadIdx.x;
	const int iny = threadIdx.y;
	const int ibx = blockIdx.x * 64;
	const int iby = blockIdx.y * 16;
	const int id = inx + iny*16;
	
	A += ibx + id;
	B += inx + __mul24( iby + iny, ldb );
	C += ibx + id  + __mul24( iby, ldc );
	
	const float *Blast = B + k;
	
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
	do
        {
            float a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            __shared__ float bs[16][17];
            bs[inx][iny]    = B[0*ldb];
            bs[inx][iny+4]  = B[4*ldb];
            bs[inx][iny+8]  = B[8*ldb];
            bs[inx][iny+12] = B[12*ldb];
            __syncthreads();

            A += 4*lda;
            saxpy( a[0], &bs[0][0], c );		a[0] = A[0*lda];
            saxpy( a[1], &bs[1][0], c );		a[1] = A[1*lda];
            saxpy( a[2], &bs[2][0], c );		a[2] = A[2*lda];
            saxpy( a[3], &bs[3][0], c );		a[3] = A[3*lda];	

            A += 4*lda;
            saxpy( a[0], &bs[4][0], c );		a[0] = A[0*lda];
            saxpy( a[1], &bs[5][0], c );		a[1] = A[1*lda];
            saxpy( a[2], &bs[6][0], c );		a[2] = A[2*lda];
            saxpy( a[3], &bs[7][0], c );		a[3] = A[3*lda];
		
            A += 4*lda;
            saxpy( a[0], &bs[8][0], c );		a[0] = A[0*lda];
            saxpy( a[1], &bs[9][0], c );		a[1] = A[1*lda];
            saxpy( a[2], &bs[10][0], c );		a[2] = A[2*lda];
            saxpy( a[3], &bs[11][0], c );		a[3] = A[3*lda];
		
            A += 4*lda;
            saxpy( a[0], &bs[12][0], c );
            saxpy( a[1], &bs[13][0], c );
            saxpy( a[2], &bs[14][0], c );
            saxpy( a[3], &bs[15][0], c );
		
            B += 16;
            __syncthreads();
        } while( B < Blast );
	
	for( int i = 0; i < 16; i++, C += ldc )
		C[0] = alpha*c[i] + beta*C[0]; 
}	

