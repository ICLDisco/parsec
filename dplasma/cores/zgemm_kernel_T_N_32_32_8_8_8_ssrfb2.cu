/*
       @precisions normal z -> s d c
*/
#include <cuda.h>

__device__ static void zaxpy(double a,double *b, double *c) {
	c[0] += a * b[0];
	c[1] += a * b[1];
	c[2] += a * b[2];
	c[3] += a * b[3];
	c[4] += a * b[4];
	c[5] += a * b[5];
	c[6] += a * b[6];
	c[7] += a * b[7];
	c[8] += a * b[8];
	c[9] += a * b[9];
	c[10] += a * b[10];
	c[11] += a * b[11];
	c[12] += a * b[12];
	c[13] += a * b[13];
	c[14] += a * b[14];
	c[15] += a * b[15];
}

/*
dgemm C = A^T*B;
     A1 = A1 - C;  
Both A and B is put into shared memory 
Parameters Used:
	blk_M=32 blk_N=32 blk_K=8 nthd_x=8 nthd_y=8
*/

extern "C" __global__ void 
zgemm_kernel_T_N_32_32_8_8_8_ssrfb2(
	double *C, const double *A, const double *B, 
        int m, int n, int k, int lda, int ldb, int ldc, 
        double *A1, int lda1) 
{
	const int ibx = blockIdx.x *32;
	const int iby = blockIdx.y *32;

	const int tx =  threadIdx.y ;
	const int ty =  threadIdx.x ;


	int idt = tx * 8 + ty;

	if( ty >=k ) 		
		A += __mul24(ibx ,lda)+0;
	else
		A += __mul24(ibx ,lda)+ty;
	
	if( (ibx + tx ) >= m ) 
		A += __mul24(0,lda);
	else 
		A += __mul24(tx,lda);

	if( (iby+tx) >=n ) 
		B+= __mul24(iby+0,ldb);
	else
		B+= __mul24(iby+tx,ldb) ;
	if( ty>=k)
		B+=0;
	else
		B+= ty;

	C += ibx +idt%32 +__mul24( iby+16*(idt/32),ldc);
	A1+= ibx +idt%32 +__mul24( iby+16*(idt/32),lda1);

	lda = lda *8 ;
	ldb = ldb *8 ;


	int as1=0, as2=lda, as3=2*lda , as4 =3*lda; 
	int bs1=0 , bs2=ldb , bs3=2*ldb , bs4=3*ldb ;


	switch(k){
		case 1: as2=0   ; as3 = 0*lda;as4=0; bs2=0   ; bs3 = 0*ldb; bs4=0; break;
		case 2: as2=lda ; as3 = 0*lda;as4=0; bs2=ldb ; bs3 = 0*ldb; bs4=0; break;
		case 3: as2=lda ; as3 = 2*lda;as4=0; bs2=ldb ; bs3 = 2*ldb; bs4=0; break;
	}

	if( (ibx + tx    ) >=m ) { as1=0;   as2=0*lda; as3=0*lda ; as4 =0*lda; } else
	if( (ibx + tx +8 ) >=m ) { as1=0;   as2=0*lda; as3=0*lda ; as4 =0*lda; } else
	if( (ibx + tx +16) >=m ) { as1=0;   as2=1*lda; as3=0*lda ; as4 =0*lda; } else
	if( (ibx + tx +24) >=m ) { as1=0;   as2=1*lda; as3=2*lda ; as4 =0*lda; } 
		

	if( (iby + tx    ) >=n ) { bs1=0;   bs2=0*ldb; bs3=0*ldb ; bs4 =0*ldb; } else
	if( (iby + tx +8 ) >=n ) { bs1=0;   bs2=0*ldb; bs3=0*ldb ; bs4 =0*ldb; } else
	if( (iby + tx +16) >=n ) { bs1=0;   bs2=1*ldb; bs3=0*ldb ; bs4 =0*ldb; } else
	if( (iby + tx +24) >=n ) { bs1=0;   bs2=1*ldb; bs3=2*ldb ; bs4 =0*ldb; } 

 
	double b= B[bs1];
	double b1=B[bs2];
	double b2=B[bs3];
	double b3=B[bs4];


	double Ap[4]={A[as1], A[as2], A[as3],A[as4]};

	const double *Bend = B + (k-k%8);

	B+=8;
	A+=8;

	__shared__ double Bb[8][33];
	__shared__ double ABb[32][9];
	
	double Cb[16] = {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
        	

	const int l = 17*(idt/32) ;
	int idt1 = idt ;	
	idt = idt % 32 ; 
	if(k>15){
	do {
		

		Bb[ty][tx   ] = b;
		Bb[ty][tx+8 ] = b1;
		Bb[ty][tx+17] = b2;
		Bb[ty][tx+25] = b3;

		ABb[tx   ][ty] = Ap[0];
		ABb[tx+8 ][ty] = Ap[1];
		ABb[tx+16][ty] = Ap[2];
		ABb[tx+24][ty] = Ap[3];
	

		__syncthreads();
		
		
		zaxpy(ABb[idt][0], &Bb[0][l], Cb);Ap[0]=A[as1]; 
		zaxpy(ABb[idt][1], &Bb[1][l], Cb);Ap[1]=A[as2];
		zaxpy(ABb[idt][2], &Bb[2][l], Cb);Ap[2]=A[as3];
		zaxpy(ABb[idt][3], &Bb[3][l], Cb);Ap[3]=A[as4]; 
	
		zaxpy(ABb[idt][4], &Bb[4][l], Cb);
		b=B[bs1];
		zaxpy(ABb[idt][5], &Bb[5][l], Cb);
		b1=B[bs2];
		zaxpy(ABb[idt][6], &Bb[6][l], Cb); 
		b2=B[bs3];
		zaxpy(ABb[idt][7], &Bb[7][l], Cb); 
		b3=B[bs4];

		B += 8;
		A += 8;

		__syncthreads();

	} while (B < Bend);
	}
	 if(k>7){

		Bb[ty][tx   ] = b;
		Bb[ty][tx+8 ] = b1;
		Bb[ty][tx+17] = b2;
		Bb[ty][tx+25] = b3;

		ABb[tx   ][ty] = Ap[0];
		ABb[tx+8 ][ty] = Ap[1];
		ABb[tx+16][ty] = Ap[2];
		ABb[tx+24][ty] = Ap[3];

		__syncthreads();
		as1 = k-k%8;
		
		if(as1+ty>=k){ bs1=0*ldb;bs2=0*ldb;bs3=0*ldb;bs4=0*ldb;B-=8;}		
		if(as1+ty>=k){ as1=0*lda;as2=0*lda;as3=0*lda;as4=0*lda;A-=8;}		

		as1=0;
		zaxpy(ABb[idt][0], &Bb[0][l], Cb);
		Ap[0]=A[as1]; 
		zaxpy(ABb[idt][1], &Bb[1][l], Cb);
		Ap[1]=A[as2];
		zaxpy(ABb[idt][2], &Bb[2][l], Cb);
		Ap[2]=A[as3];
		zaxpy(ABb[idt][3], &Bb[3][l], Cb);
		Ap[3]=A[as4]; 
	
		zaxpy(ABb[idt][4], &Bb[4][l], Cb);
		b=B[bs1];
		zaxpy(ABb[idt][5], &Bb[5][l], Cb);
		b1=B[bs2];
		zaxpy(ABb[idt][6], &Bb[6][l], Cb); 
		b2=B[bs3];
		zaxpy(ABb[idt][7], &Bb[7][l], Cb); 
		b3=B[bs4];
	}
	k=k%8;
	if ( k!=0){
		__syncthreads();

		Bb[ty][tx]= b;
		Bb[ty][tx+8] = b1;
		Bb[ty][tx+17] = b2;
		Bb[ty][tx+25] = b3;

		ABb[tx][ty]= Ap[0];
		ABb[tx+8][ty] = Ap[1];
		ABb[tx+16][ty] = Ap[2];
		ABb[tx+24][ty] = Ap[3];
		__syncthreads();

		for(int i=0;i<k;i++){
			zaxpy(ABb[idt][i],&Bb[i][l], Cb);
		}
	}

	if( (iby+16*(idt1/32+1))>=n) { 
		lda = n-iby-16*(idt1/32);
	}
	else    {
		lda = 16;
	}
	if( (ibx+idt) >= m )
		lda = 0 ;
	else lda = lda ;
	
	switch(lda){
		case 16:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			C[6*ldc] = Cb[6] ; A1[6*ldc]-=Cb[ 6];
			C[7*ldc] = Cb[7] ; A1[7*ldc]-=Cb[ 7];
			C[8*ldc] = Cb[8] ; A1[8*ldc]-=Cb[ 8];
			C[9*ldc] = Cb[9] ; A1[9*ldc]-=Cb[ 9];
			C[10*ldc] = Cb[10] ; A1[10*ldc]-=Cb[ 10];
			C[11*ldc] = Cb[11] ; A1[11*ldc]-=Cb[ 11];
			C[12*ldc] = Cb[12] ; A1[12*ldc]-=Cb[ 12];
			C[13*ldc] = Cb[13] ; A1[13*ldc]-=Cb[ 13];
			C[14*ldc] = Cb[14] ; A1[14*ldc]-=Cb[ 14];
			C[15*ldc] = Cb[15] ; A1[15*ldc]-=Cb[ 15];
			break;
		case 15:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			C[6*ldc] = Cb[6] ; A1[6*ldc]-=Cb[ 6];
			C[7*ldc] = Cb[7] ; A1[7*ldc]-=Cb[ 7];
			C[8*ldc] = Cb[8] ; A1[8*ldc]-=Cb[ 8];
			C[9*ldc] = Cb[9] ; A1[9*ldc]-=Cb[ 9];
			C[10*ldc] = Cb[10] ; A1[10*ldc]-=Cb[ 10];
			C[11*ldc] = Cb[11] ; A1[11*ldc]-=Cb[ 11];
			C[12*ldc] = Cb[12] ; A1[12*ldc]-=Cb[ 12];
			C[13*ldc] = Cb[13] ; A1[13*ldc]-=Cb[ 13];
			C[14*ldc] = Cb[14] ; A1[14*ldc]-=Cb[ 14];
			break;
		case 14:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			C[6*ldc] = Cb[6] ; A1[6*ldc]-=Cb[ 6];
			C[7*ldc] = Cb[7] ; A1[7*ldc]-=Cb[ 7];
			C[8*ldc] = Cb[8] ; A1[8*ldc]-=Cb[ 8];
			C[9*ldc] = Cb[9] ; A1[9*ldc]-=Cb[ 9];
			C[10*ldc] = Cb[10] ; A1[10*ldc]-=Cb[ 10];
			C[11*ldc] = Cb[11] ; A1[11*ldc]-=Cb[ 11];
			C[12*ldc] = Cb[12] ; A1[12*ldc]-=Cb[ 12];
			C[13*ldc] = Cb[13] ; A1[13*ldc]-=Cb[ 13];
			break;
		case 13:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			C[6*ldc] = Cb[6] ; A1[6*ldc]-=Cb[ 6];
			C[7*ldc] = Cb[7] ; A1[7*ldc]-=Cb[ 7];
			C[8*ldc] = Cb[8] ; A1[8*ldc]-=Cb[ 8];
			C[9*ldc] = Cb[9] ; A1[9*ldc]-=Cb[ 9];
			C[10*ldc] = Cb[10] ; A1[10*ldc]-=Cb[ 10];
			C[11*ldc] = Cb[11] ; A1[11*ldc]-=Cb[ 11];
			C[12*ldc] = Cb[12] ; A1[12*ldc]-=Cb[ 12];
			break; 
		case 12:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			C[6*ldc] = Cb[6] ; A1[6*ldc]-=Cb[ 6];
			C[7*ldc] = Cb[7] ; A1[7*ldc]-=Cb[ 7];
			C[8*ldc] = Cb[8] ; A1[8*ldc]-=Cb[ 8];
			C[9*ldc] = Cb[9] ; A1[9*ldc]-=Cb[ 9];
			C[10*ldc] = Cb[10] ; A1[10*ldc]-=Cb[ 10];
			C[11*ldc] = Cb[11] ; A1[11*ldc]-=Cb[ 11];
			break;
		case 11:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			C[6*ldc] = Cb[6] ; A1[6*ldc]-=Cb[ 6];
			C[7*ldc] = Cb[7] ; A1[7*ldc]-=Cb[ 7];
			C[8*ldc] = Cb[8] ; A1[8*ldc]-=Cb[ 8];
			C[9*ldc] = Cb[9] ; A1[9*ldc]-=Cb[ 9];
			C[10*ldc]= Cb[10] ; A1[10*ldc]-=Cb[ 10];
			break;
		case 10:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			C[6*ldc] = Cb[6] ; A1[6*ldc]-=Cb[ 6];
			C[7*ldc] = Cb[7] ; A1[7*ldc]-=Cb[ 7];
			C[8*ldc] = Cb[8] ; A1[8*ldc]-=Cb[ 8];
			C[9*ldc] = Cb[9] ; A1[9*ldc]-=Cb[ 9];
			break;
		case 9:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			C[6*ldc] = Cb[6] ; A1[6*ldc]-=Cb[ 6];
			C[7*ldc] = Cb[7] ; A1[7*ldc]-=Cb[ 7];
			C[8*ldc] = Cb[8] ; A1[8*ldc]-=Cb[ 8];
			break;
		case 8:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			C[6*ldc] = Cb[6] ; A1[6*ldc]-=Cb[ 6];
			C[7*ldc] = Cb[7] ; A1[7*ldc]-=Cb[ 7];
			break;
		case 7:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			C[6*ldc] = Cb[6] ; A1[6*ldc]-=Cb[ 6];
			break;
		case 6:
			C[0] = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			C[5*ldc] = Cb[5] ; A1[5*ldc]-=Cb[ 5];
			break;
		case 5:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			C[4*ldc] = Cb[4] ; A1[4*ldc]-=Cb[ 4];
			break;
		case 4:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			C[3*ldc] = Cb[3] ; A1[3*ldc]-=Cb[ 3];
			break;
		case 3:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			C[2*ldc] = Cb[2] ; A1[2*ldc]-=Cb[ 2];
			break;
		case 2:
			C[0]     = Cb[0] ; A1[0]-=Cb[ 0];
			C[1*ldc] = Cb[1] ; A1[1*ldc]-=Cb[ 1];
			break;
		case 1:
			C[0] = Cb[0] ; A1[0]-=Cb[ 0];
			break;
		case 0:
			break;
	}

}
 
extern "C" void
magmablas_zgemm_kernel_T_N_32_32_8_8_8_ssrfb2( 
	int m, int n, int k,
	const double *A, int lda, 
	const double *B, int ldb, 
	double *C, int ldc, 
        double *A1, int lda1, CUstream stream)
{	
        dim3 threads( 8, 8 );
        dim3 grid(m/32+(m%32!=0),n/32+(n%32!=0));
        zgemm_kernel_T_N_32_32_8_8_8_ssrfb2<<< grid, threads, 0, stream >>>(C, A, B, 
               m, n, k, lda , ldb , ldc, A1, lda1) ;
}
