/*
    -- MAGMA (version 0.3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2010

    -- Stan Tomov
    -- Rajib Nath
*/


/*
    This version is used with testing_stream_sgemm.cpp to add streams.
    Streams work with a code that does not use textures. The textures currently
    are given as a global variables so they can not be used concurrently.

    blk_M=64 blk_N=64 blk_K=16 nthd_x=64 nthd_y=4
*/

#define  blks 4
#define bx 4
#define by 4

#define  blk_M (16*bx)
#define  blk_N (16*by)

#define  blk_K 16
#define  nth_x 64

#define nth_x_1 16

#define tot_row blks

#include <cuda_runtime.h>
#include <cuda.h>

#define fermiSgemm_v2_kernel_NN sgemmNN
#define fermiSgemm_v2_kernel_TN sgemmTN
#define fermiSgemm_v2_kernel_NT sgemmNT
#define fermiSgemm_v2_kernel_TT sgemmTT

extern "C" __global__ 
void fermiSgemm_v2_kernel_NN( const float *A, int lda,
                              const float *B, int ldb,
                              float* C, int ldc,
                              int k,
                              float alpha, float beta)
{
    const  int tx = threadIdx.x;
    const  int ty = threadIdx.y;

    const int iby = blockIdx.y * blk_N;
    const int ibx = blockIdx.x * blk_M;
    const int idt = ty * nth_x + tx;

    // TTT - this is to reorganize the threads form 64x4 (in tx x ty)
    //       to 16x16 (in res x qot)
    const int res= idt%nth_x_1;
    const int qot= idt/nth_x_1;

    // ===== TTT - shared memory =================================================
    __shared__ float Bb[blk_K][blk_N+1];
    __shared__ float Abs[blk_M][blk_K+1];

    // ===== TTT - registers for blks x blks register blocking ===================
    float xxA[bx];
    float xxB[by];
    
    B+= res + __mul24(iby + qot * by, ldb );
    A+= ibx + __mul24( qot, lda) + res ; 

    int trackA =  ibx + __mul24( qot, lda) + res ;
    int trackB =  res + __mul24(iby + qot * by, ldb );

    // ===== TTT =================================================================
    // 1. Read blocks A and B in shared memory (pre-fetch)
    #pragma unroll
    for(int y=0; y<by; y++)
        Bb[res][qot*by+y] = B[y*ldb]; // fetch_x_B( trackB + y*ldb, B) ;
	

    // TTT - this reads 16 x 16 blocks per line 
    #pragma unroll
    for(int y=0; y<bx; y++)
        Abs[res+y*16][qot] = A[y*16];//fetch_x_A(trackA +  y*16 , A);

    __syncthreads();

    const float *Bend = B + k-16;
   
    float Axs[bx];
    float Bxp[by];

    //float Cb[blks*blks] = {0,0,0,0,   ... };

    //float Cb[9] = {0,0,0, 0,0,0, 0,0,0};
    //float Cb[16] = {0,0,0,0,    0,0,0,0, 0,0,0,0, 0,0,0,0};
    //float Cb[25] = {0,0,0,0,0, 0,0,0,0,0,  0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0};
    float Cb[36] = {0,0,0,0,0,0, 0,0,0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0,0,
                    0,0,0,0,0,0, 0,0,0,0,0,0};
    //float Cb[30] = {0,0,0,0,0,0, 0,0,0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0,0,
    //                0,0,0,0,0,0};
    //float Cb[42] = {0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0,
    //               0,0,0,0,0,0,0, 0,0,0,0,0, 0,0};   
 
    //float Cb[49] = {0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0,
    //                0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0};

    do 
    {
	B += 16;
	A += lda *16  ;
	trackA+=16*lda ; 
	trackB+=16;

              #pragma unroll 
              for( int j1=0;j1<8;j1++){

                  // TTT
                  // 2. Put 4 elements of B in registers
                  #pragma unroll
                  for( int y=0; y<by; y++)
                      Bxp[y]= Bb[j1][qot+y*16];

                  // TTT
                  // 3. Put 4 elements of A in registers
                  #pragma unroll
                  for( int y=0; y<bx; y++)
                      Axs[y] =  Abs[res+y*16][j1] ;

                  // TTT
                  // 4. Multiply them adding the result in C
                  #pragma unroll 
                  for( int x=0; x<bx; x++){
                     #pragma unroll 
                     for( int y=0; y<by; y++){
                         Cb[x*by + y] += Axs[x]*Bxp[y];
		     }
		  }
              } // j1 - loop

        #pragma unroll
        for( int y=0; y<by; y++)
 	   xxB[y] = B[y*ldb]; //fetch_x_B( trackB + y*ldb, B);

        #pragma unroll
        for( int y=0; y<bx; y++)
	   xxA[y] = A[y*16]; //fetch_x_A(trackA + y*16 , A);

              #pragma unroll 
              for( int j1=8;j1<16;j1++){

                  #pragma unroll
                  for( int y=0;y<by;y++)
                      Bxp[y]= Bb[j1][qot+y*16];

                  #pragma unroll
                  for( int y=0;y<bx;y++)
                      Axs[y] =  Abs[res+y*16][j1] ;

                  #pragma unroll 
                  for( int x=0;x<bx;x++){
                     #pragma unroll 
                     for( int y=0; y<by; y++){
                         Cb[x*by+y] += Axs[x]*Bxp[y];
		     }
		  }
               }// j1 - loop

	__syncthreads();
	#pragma unroll
	for(int y=0;y<bx;y++)
	    Abs[res+y*16][qot] =xxA[y];

	#pragma unroll
	for(int y=0; y<by; y++)
	    Bb[res][qot*by + y] =xxB[y];

	__syncthreads();

    } 
    while (B < Bend);

    // C += qot + ibx  + __mul24 (res +  iby ,ldc);
    C += res + ibx + __mul24(qot + iby, ldc);

        // TTT - this is unrolling the last iteration 
        //       (data has been already prepared)
	#pragma unroll 
        for( int j1=0;j1<16;j1++){

	   #pragma unroll
	   for( int y=0;y<by;y++)
		Bxp[y]= Bb[j1][qot+y*16];
 
	   #pragma unroll
	   for( int y=0;y<bx;y++)
		Axs[y] = Abs[res+y*16][j1] ;

	   #pragma unroll 
	   for( int x=0;x<bx;x++)
		#pragma unroll 
		for( int y=0; y<by; y++)
		   Cb[x*by+y]  += Axs[x]*Bxp[y];	
	}// j1 - loop

    #pragma unroll
    for( int y=0; y<by; y++){
 	#pragma unroll
        for(int x=0; x<bx; x++){
	   C[x*16] = alpha*Cb[y+x*by] + beta * C[x*16];
	}
	   
        C+=ldc*16;
    }
}

//========================================================================

extern "C" __global__ 
void fermiSgemm_v2_kernel_TN( const float *A, int lda,
                              const float *B, int ldb,
                              float* C, int ldc,
                              int k,
                              float alpha, float beta)
{
    const  int tx = threadIdx.x;
    const  int ty = threadIdx.y;

    const int iby = blockIdx.y * blk_N;
    const int ibx = blockIdx.x * blk_M;
    const int idt = ty * nth_x + tx;

    const int res = idt%nth_x_1;
    const int qot = idt/nth_x_1;

    __shared__ float Bb[blk_K][blk_N+1];
    __shared__ float Abs[blk_M][blk_K+1];

    float xxA[blks];
    float xxB[blks];

    B+= res+ __mul24(iby + qot*blks, ldb );
    int trackB = res+ __mul24(iby + qot*blks, ldb );

    A+= __mul24( ibx + qot, lda ) + res; 
    int trackA =  __mul24( ibx + qot,lda) + res;

    #pragma unroll
    for(int y=0; y<blks; y++)
	Bb[res][qot*blks+y] = B[y*ldb]; //fetch_x_B(trackB + y*ldb, B ) ;

    #pragma unroll
    for(int y=0; y<blks; y++)
	Abs[qot+16*y][res] = A[lda*16*y]; //fetch_x_A(trackA +  lda*16*y, A);

    __syncthreads();

    const float *Bend = B + k-16;
   
    float Axs[blks];
    float Bxp[blks];

    // float Cb[9] = {0,0,0, 0,0,0, 0,0,0};
    // float Cb[16] = {0,0,0,0,    0,0,0,0, 0,0,0,0, 0,0,0,0};
    //float Cb[25] = {0,0,0,0,0, 0,0,0,0,0,  0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0};
    float Cb[36] = {0,0,0,0,0,0, 0,0,0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0,0,
                    0,0,0,0,0,0, 0,0,0,0,0,0};

    do 
    {
	B += 16;
	A += 16;
	trackA += 16 ; 
	trackB += 16;

        #pragma unroll
        for( int y=0; y<blks; y++)
	   xxB[y] = B[y*ldb]; //fetch_x_B( trackB + y*ldb, B);

        #pragma unroll
        for( int y=0; y<blks; y++)
	   xxA[y] = A[lda*y*16]; //fetch_x_A(trackA + lda*y*16, A);

        #pragma unroll 
        for( int j1=0; j1<16; j1++){
           
            #pragma unroll
            for( int y=0; y<blks; y++)
                Bxp[y]= Bb[j1][qot + y*16];

            #pragma unroll
            for( int y=0; y<blks; y++)
                Axs[y] = Abs[res + y*16][j1];

            #pragma unroll 
            for( int x=0;x<tot_row;x++){
                #pragma unroll 
                for(int y=0; y<blks; y++){
                    Cb[x*blks + y] += Axs[x]*Bxp[y];
						}
					}
        }// j1 - loop
           
	__syncthreads();
	#pragma unroll
	for(int y=0; y<blks; y++)
	   Abs[qot + 16*y][res] = xxA[y]; 
	
	#pragma unroll
	for( int y=0; y<blks; y++)
	   Bb[res][qot*blks + y] = xxB[y];

	__syncthreads();
    } 
    while (B < Bend);

    C += res + ibx  + __mul24 (qot + iby, ldc);

    #pragma unroll 
    for( int j1=0; j1<16; j1++){

        #pragma unroll
        for( int y=0; y<blks; y++)
            Bxp[y] = Bb[j1][qot + y*16];

        #pragma unroll
        for( int y=0; y<blks; y++)
            Axs[y] =  Abs[res + y*16][j1] ;

        #pragma unroll 
        for( int x=0; x<blks; x++){
           #pragma unroll 
           for( int y=0; y<blks; y++){
               Cb[x*blks + y]  += Axs[x]*Bxp[y];
	   }
	}
    }// j1 - loop

    #pragma unroll
    for(int y=0;y<blks; y++){
       #pragma unroll
       for(int x=0; x<blks; x++){
	  C[x*16] = alpha*Cb[y+x*blks] + beta * C[x*16];
       }
	   
       C+=ldc*16;
    }
}

//========================================================================

extern "C" __global__ 
void fermiSgemm_v2_kernel_TT( const float *A, int lda,
                              const float *B, int ldb,
                              float* C, int ldc,
                              int k,
                              float alpha, float beta)
{
    const  int tx = threadIdx.x;
    const  int ty = threadIdx.y;

    const int iby = blockIdx.y * blk_N;
    const int ibx = blockIdx.x * blk_M;
    const int idt = ty * nth_x + tx;

    const int res = idt% nth_x_1;
    const int qot = idt/ nth_x_1;

    __shared__ float Bb[blk_K][blk_N+1];
    __shared__ float Abs[blk_M][blk_K+1];

    float xxA[blks];
    float xxB[blks];

    B += iby + res + __mul24(qot , ldb );
    A += __mul24(ibx + qot, lda) + res; 

    int trackA =  __mul24( ibx + qot, lda) + res;
    int trackB =  iby+ res + __mul24(qot, ldb);

    #pragma unroll
    for(int y=0; y<blks; y++)
      Bb[qot][res+16*y] = B[16*y]; //fetch_x_B(trackB+16*y, B);

    #pragma unroll
    for(int y=0; y<blks; y++)
       Abs[qot + 16*y][res] = A[lda*16*y]; //fetch_x_A(trackA +  lda*16*y, A);

    __syncthreads();

    const float *Bend = B + k*ldb - 16*ldb;
   
    float Axs[blks];
    float Bxp[blks];
    
    //float Cb[9] = {0,0,0, 0,0,0, 0,0,0};
    //float Cb[16] = {0,0,0,0,    0,0,0,0, 0,0,0,0, 0,0,0,0};
    //float Cb[25] = {0,0,0,0,0, 0,0,0,0,0,  0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0};
    float Cb[36] = {0,0,0,0,0,0, 0,0,0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0,0,
                    0,0,0,0,0,0, 0,0,0,0,0,0};

    do 
    {
	B += 16*ldb;
	A += 16;
	trackA+=16 ; 
	trackB+=16*ldb;

	#pragma unroll
        for( int y=0; y<blks; y++)
           xxB[y] = B[16*y]; //fetch_x_B(trackB + 16*y, B);
		
        #pragma unroll
        for( int y=0; y<blks; y++)
	   xxA[y] = A[lda*y*16]; //fetch_x_A(trackA + lda*y*16, A);

        #pragma unroll 
        for( int j1=0;j1<16;j1++){
           #pragma unroll
           for( int y=0; y<blks; y++)
              Bxp[y]= Bb[j1][qot + y*16];

           #pragma unroll
           for( int y=0; y<blks; y++)
              Axs[y] =  Abs[res + y*16][j1];

           #pragma unroll 
           for( int x=0; x<blks; x++){
               #pragma unroll 
               for( int y=0;y<blks;y++){
                  Cb[x*blks+y] += Axs[x]*Bxp[y];
	       }
	    }
        }// j1 - loop
            
	__syncthreads();
	#pragma unroll
	for( int y=0; y<blks; y++)
	   Abs[qot + 16*y][res] = xxA[y];
 
	#pragma unroll
	for( int y=0; y<blks; y++)
           Bb[qot][res+y*16] = xxB[y];

	__syncthreads();
    } 
    while (B < Bend);

    C += res + ibx  + __mul24 (qot +  iby ,ldc);

    #pragma unroll 
    for( int j1=0; j1<16; j1++){
        #pragma unroll
        for( int y=0; y<blks; y++)
            Bxp[y]= Bb[j1][qot + y*16];

        #pragma unroll
        for( int y=0; y<blks; y++)
            Axs[y] =  Abs[res + y*16][j1];

        #pragma unroll 
        for( int x=0; x<blks; x++){
           #pragma unroll 
           for( int y=0; y<blks; y++){
              Cb[x*blks+y]  += Axs[x]*Bxp[y];
	   }
	}
    }// j1 - loop

    #pragma unroll
    for( int y=0;y<blks;y++){
	#pragma unroll
        for(int x=0; x<blks; x++){
	    C[x*16] = alpha*Cb[y+x*blks] + beta * C[x*16];
	}
	   
	C+=ldc*16;
    }
}


//========================================================================

extern "C" __global__ 
void fermiSgemm_v2_kernel_NT( const float *A, int lda,
                              const float *B, int ldb,
                              float* C, int ldc,
                              int k,
                              float alpha, float beta)
{
    const int iby = blockIdx.y * blk_N;
    const int ibx = blockIdx.x * blk_M;
    const int idt = threadIdx.y * nth_x + threadIdx.x;

    const int res= idt%nth_x_1;
    const int qot= idt/nth_x_1;
/*
    const int res= threadIdx.x;
    const int qot= threadIdx.y;
*/
    __shared__ float Bb[blk_K][blk_N+1];
    __shared__ float Abs[blk_M][blk_K+1];

    float xxA[blks];
    float xxB[blks];

    B += iby + res + qot*ldb;
    int trackB = iby + res + qot*ldb;	
	
    A+= ibx + qot* lda + res ; 
    int trackA = ibx + qot*lda + res ;

    #pragma unroll
    for(int y=0; y<blks; y++)
        Bb[qot][res+16*y] = B[16*y]; //fetch_x_B(trackB+16*y, B);

    #pragma unroll
    for(int y=0; y<blks; y++)
	Abs[res+ y*16][qot] = A[y*16]; //fetch_x_A(trackA +  y*16, A);

    __syncthreads();

    const float *Bend = B + k*ldb - 16*ldb;
   
    float Axs[blks];
    float Bxp[blks];

    // float Cb[9] = {0,0,0, 0,0,0, 0,0,0};
    // float Cb[16] = {0,0,0,0,    0,0,0,0, 0,0,0,0, 0,0,0,0};
    //float Cb[25] = {0,0,0,0,0, 0,0,0,0,0,  0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0};
    float Cb[36] = {0,0,0,0,0,0, 0,0,0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0,0,
                    0,0,0,0,0,0, 0,0,0,0,0,0};

    do 
    {
	B += 16*ldb;
	A += lda *16  ;
	trackA+=16*lda ; 
	trackB+=16*ldb;

        #pragma unroll
        for( int y=0; y<blks; y++)
          xxB[y] = B[16*y]; //fetch_x_B( trackB + 16*y, B);

        #pragma unroll
        for( int y=0; y<blks; y++)
	   xxA[y] = A[y*16]; // fetch_x_A(trackA +  y*16, A);

        #pragma unroll 
        for( int j1=0;j1<16;j1++){
                  #pragma unroll
                  for( int y=0; y<blks; y++)
                      Bxp[y]= Bb[j1][qot + y*16];
                  #pragma unroll
                  for( int y=0; y<blks; y++)
                      Axs[y] =  Abs[res + y*16][j1] ;

                  #pragma unroll 
                  for( int x=0; x<blks; x++){
                     #pragma unroll 
                     for( int y=0; y<blks; y++){
                         Cb[x*blks+y] += Axs[x]*Bxp[y];
		     }
		  }
        }// j1 - loop
            
	__syncthreads();
	#pragma unroll
	for( int y=0; y<blks; y++)
	    Abs[res + y*16][qot] = xxA[y]; 

	#pragma unroll
	for( int y=0; y<blks; y++)
            Bb[qot][res+y*16] = xxB[y];

	__syncthreads();
     } 
     while (B < Bend);

     C += res + ibx +(qot + iby)*ldc;

	#pragma unroll 
        for(int j1=0; j1<16; j1++){
                  #pragma unroll
                  for( int y=0; y<blks; y++)
                      Bxp[y] = Bb[j1][qot + y*16];

                  #pragma unroll
                  for( int y=0; y<blks; y++)
                      Axs[y] =  Abs[res + y*16][j1] ;

                  #pragma unroll 
                  for( int x=0; x<blks; x++){
                     #pragma unroll 
                     for( int y=0;y<blks;y++){
                         Cb[x*blks+y]  += Axs[x]*Bxp[y];
		     }
		  }
        }// j1 - loop

	#pragma unroll
   	for( int y=0;y<blks;y++){
	   #pragma unroll
           for(int x=0; x<blks; x++){
	       C[x*16] = alpha*Cb[y + x*blks] + beta * C[x*16];
	   }
	   
	   C+=ldc*16;
	}
}

//=================================================================================
#if 0
#include<stdio.h>
#include "cublas.h"

extern "C" void
magmablas_fermi_sgemm(char TRANSA, char TRANSB, int m , int n , int k , float alpha, 
                      const float *A, int lda, const float *B, int ldb, float beta, 
                      float *C, int ldc ) 
{

        if (m<=0 || n<=0 || k<=0)
           return;

        if( m % (16*bx) !=0 || n% (16*by) !=0 || k% (16) !=0 )
	{
		printf("Dimension Should Be multiple of %d\n", 16*blks);
		printf("Calling cublasSgemm\n");
		cublasSgemm(TRANSA, TRANSB, m, n, k, alpha, A, lda, B,ldb, 
                            beta, C, ldc);
		return;
	}

        dim3 threads( 64, 4 );
	dim3 grid(m/(16*bx)+(m%(16*bx)!=0),n/(16*by)+(n%(16*by)!=0));

        if( TRANSB == 'T' || TRANSB == 't') 
	  if( TRANSA == 'N' ||  TRANSA == 'n') 
           fermiSgemm_v2_kernel_NT<<< grid, threads>>>(C, A, B, m, n, k, lda, ldb, 
                                                       ldc, alpha, beta);
	  else
           fermiSgemm_v2_kernel_TT<<< grid, threads>>>(C, A, B, m, n, k, lda, ldb, 
                                                       ldc, alpha, beta);
        else
	  if( TRANSA == 'N' || TRANSA == 'n') 
           fermiSgemm_v2_kernel_NN<<< grid, threads>>>(C, A, B, m, n, k, lda, ldb, 
                                                       ldc, alpha, beta);
          else
           fermiSgemm_v2_kernel_TN<<< grid, threads>>>(C, A, B, m, n, k, lda, ldb, 
                                                       ldc, alpha, beta);

}
#endif
//====================================================================================

