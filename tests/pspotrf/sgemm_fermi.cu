/*
	blk_M=64 blk_N=64 blk_K=16 nthd_x=64 nthd_y=4
*/
#define  blk_M 64
#define  blk_N 64 
#define  blk_K 16

#define fermi_sgemm_kernel_N_N_64_64_16_64_4 sgemmNT

extern "C" __global__ void fermi_sgemm_kernel_N_N_64_64_16_64_4( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta ) {
	const  int tx = threadIdx.x;
	const  int ty = threadIdx.y;

    int iby = blockIdx.y ;
    int ibx = blockIdx.x ;
    //ibx = (blockIdx.x+ blockIdx.y ) % (m / 64 ) ;  
    //iby = (ibx+blockIdx.y ) % (m / 64 ) ;  
    ibx*=64 ; 
    iby*=64 ; 
    const  int idt = ty * 64 + tx;


    int txr = idt /16 ; 
	const  int tyr = idt %16 ; 
	const  int res= idt%16;
	int qot= idt/16;
    //	const int qot_map[16]={0,4,1,5,2,6,3,7,8,12,9,13,10,14,11,15};
	int qot_map[16]={0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15};
	//int qot_map[16]={0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15};
	//int qot_map[16]={0,13,2,11,4,9,6,7,8,5,10,1,12,3, 14,15};
	qot=qot_map[qot];
	txr = qot ; 
	
	//C += ibx  + idt%64 + __mul24 ( idtdivN_nc + iby ,ldc);


	__shared__ float Bb[16][65];
	__shared__ float Abs[64][17];

    float Cb[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
#undef preloadB 
#undef preloadA
#define preloadB 
#define preloadA
#ifdef  preloadA
    float xxA[4];
#endif
#ifdef preloadB
    float xxB[4];
#endif
    
    B+= res+ __mul24(iby+qot*4, ldb );
#ifdef preloadB 
    xxB[0] =B[0*ldb] ; 
    xxB[1] =B[1*ldb] ; 
    xxB[2] =B[2*ldb] ; 
    xxB[3] =B[3*ldb] ; 
#endif
#ifndef preloadA 
    xxA[0] =A[ty*4*lda + 0*lda];
    xxA[1] =A[ty*4*lda + 1*lda];
    xxA[2] =A[ty*4*lda + 2*lda];
    xxA[3] =A[ty*4*lda + 3*lda];
#endif
#ifdef preloadB 
    Bb[res][qot*4+0] =xxB[0] ;
    Bb[res][qot*4+1] =xxB[1] ; 
    Bb[res][qot*4+2] =xxB[2] ; 
    Bb[res][qot*4+3] =xxB[3] ; 
	//B+= __mul24(qot, ldb );
	const float *Bend = B + k-16;
#endif
#ifdef preloadA 
    A+= ibx +qot*lda + res ; 
    Abs[res   ][qot] = A[ 0]; 
    Abs[res+16][qot] = A[16]; 
    Abs[res+32][qot] = A[32]; 
    Abs[res+48][qot] = A[48]; 
#endif

#ifndef preloadB 
	Bb[res][qot+0] = B[0*ldb];
	Bb[res][qot+16] = B[16*ldb];
	Bb[res][qot+32] = B[32*ldb];
	Bb[res][qot+48] = B[48*ldb];
#endif
#ifndef preloadA 
    Abs[tx][ty*4+0] = A[ty*4*lda + 0*lda]; 
    Abs[tx][ty*4+1] = A[ty*4*lda + 1*lda]; 
    Abs[tx][ty*4+2] = A[ty*4*lda + 2*lda]; 
    Abs[tx][ty*4+3] = A[ty*4*lda + 3*lda]; 
#endif
	__syncthreads();

   
    int j=0 ; 
    float Axs[4];
    float Bxp[4];

#pragma unroll
    for(  int y=0;y<4;y++)
        Axs[y] = Abs[txr+y*16][j*4];
#pragma unroll
    for( int y=0;y<4;y++)
        Bxp[y]= Bb[j*4][tyr+y*16]; 

#define block_global
#define block_shared
#undef block_shared
#undef block_global
	do {
		B += 16;
		A += 16 * lda;
#ifndef block_global
#ifdef preloadA 
#pragma unroll 
        for( int y=0;y<4;y++)
            xxA[y]=  A[y*16]; 
#endif
#endif
        {
            j=0;
#pragma unroll 
            for( int j1=0;j1<3;j1++){
#pragma unroll 
                for( int x=0;x<4;x++)
#pragma unroll 
                    for( int y=0;y<4;y++)
                        Cb[x*4+y]  += Axs[x]*Bxp[y];	
#pragma unroll
                for( int y=0;y<4;y++)
                    Bxp[y]= Bb[j*4+j1+1][tyr+y*16]; 
#pragma unroll
                for( int y=0;y<4;y++)
                    Axs[y] = Abs[txr+y*16][j*4+j1+1] ;
			}// j1 - loop
#pragma unroll 
	  	    for( int x=0;x<3;x++)
#pragma unroll 
                for( int y=0;y<4;y++)
                    Cb[x*4+y]  += Axs[x]*Bxp[y];	
#pragma unroll 
			for( int x=3;x<4;x++)
#pragma unroll 
                for( int y=0;y<4;y++){
                    Cb[x*4+y]  += Axs[x]*Bxp[y];	
#ifdef preloadB 
                    //xxB[y] = B[y*ldb] ;
#endif
                }
        }// j  - loop 
#ifndef block_global
#ifdef preloadB 
#pragma unroll 
        for( int y=0;y<4;y++)
            xxB[y] = B[y*ldb] ;
#endif
#endif
        {
		    j=1;
#ifndef block_shared
#pragma unroll
            for(  int y=0;y<4;y++)
                Axs[y] = Abs[txr+y*16][j*4] ;
#pragma unroll
            for( int y=0;y<4;y++)
                Bxp[y]= Bb[j*4][tyr+y*16]; 
#endif
#pragma unroll 
            for( int j1=0;j1<3;j1++){
#pragma unroll 
                for( int x=0;x<4;x++)
#pragma unroll 
                    for( int y=0;y<4;y++)
                        Cb[x*4+y]  += Axs[x]*Bxp[y];	
#pragma unroll
                for( int y=0;y<4;y++)
                    Bxp[y]= Bb[j*4+j1+1][tyr+y*16]; 
#pragma unroll
                for( int y=0;y<4;y++)
                    Axs[y] = Abs[txr+y*16][j*4+j1+1] ;
			}// j1 - loop
#pragma unroll 
            for( int x=0;x<4;x++)
#pragma unroll 
                for( int y=0;y<4;y++){
                    Cb[x*4+y]  += Axs[x]*Bxp[y];	
#ifdef preloadA 
                    //xxA[y]=  A[y*16]; 
#endif
                }
        }// j  - loop 
			 
#pragma unroll
        for(j=2;j<4;j++){
#ifndef block_shared
#pragma unroll
            for(  int y=0;y<4;y++)
                Axs[y] = Abs[txr+y*16][j*4] ;
#pragma unroll
            for( int y=0;y<4;y++)
                Bxp[y]= Bb[j*4][tyr+y*16]; 
#endif
#pragma unroll 
            for( int j1=0;j1<3;j1++){
#pragma unroll 
                for( int x=0;x<4;x++)
#pragma unroll 
                    for( int y=0;y<4;y++)
                        Cb[x*4+y]  += Axs[x]*Bxp[y];	
#pragma unroll
                for( int y=0;y<4;y++)
                    Bxp[y]= Bb[j*4+j1+1][tyr+y*16]; 
#pragma unroll
                for( int y=0;y<4;y++)
                    Axs[y] = Abs[txr+y*16][j*4+j1+1] ;
			}// j1 - loop
#pragma unroll 
            for( int x=0;x<4;x++)
#pragma unroll 
                for( int y=0;y<4;y++)
                    Cb[x*4+y]  += Axs[x]*Bxp[y];	
        }// j - loop 
#ifndef preloadB 
		//xxB[0] =B[0*ldb] ; 
		//xxB[1] =B[16*ldb] ; 
		//xxB[2] =B[32*ldb] ; 
		//xxB[3] =B[48*ldb] ; 
#endif
#ifndef preloadA 
		//xxA[0] =A[ty*4*lda + 0*lda];
		//xxA[1] =A[ty*4*lda + 1*lda];
		//xxA[2] =A[ty*4*lda + 2*lda];
		//xxA[3] =A[ty*4*lda + 3*lda];
#endif
		__syncthreads();
#ifndef preloadB 
		Bb[res][qot+0]  = B[0*ldb];
		Bb[res][qot+16] = B[16*ldb];
		Bb[res][qot+32] = B[32*ldb];
		Bb[res][qot+48] = B[48*ldb];
#endif
#ifdef preloadB 
#pragma unroll
		for(  int y=0;y<4;y++)
            Bb[res][qot*4+y] =xxB[y];
#endif
#ifndef preloadA 
	    Abs[tx][ty*4+0] = A[ty*4*lda + 0*lda]; 
	    Abs[tx][ty*4+1] = A[ty*4*lda + 1*lda]; 
	    Abs[tx][ty*4+2] = A[ty*4*lda + 2*lda]; 
	    Abs[tx][ty*4+3] = A[ty*4*lda + 3*lda]; 
#endif
#ifdef preloadA 
#pragma unroll
		for(  int y=0;y<4;y++)
            Abs[res+y*16   ][qot] =xxA[y]; 
#endif
		__syncthreads();
		j=0;
#ifndef block_shared
#pragma unroll
        for(  int y=0;y<4;y++)
            Axs[y] = Abs[txr+y*16][j*4] ;
#pragma unroll
        for( int y=0;y<4;y++)
            Bxp[y]= Bb[j*4][tyr+y*16]; 
#endif

	} while (B < Bend);

    {
#pragma unroll 
        for( int j1=0;j1<3;j1++){
#pragma unroll 
            for( int x=0;x<4;x++)
#pragma unroll 
                for( int y=0;y<4;y++)
                    Cb[x*4+y]  += Axs[x]*Bxp[y];	
#pragma unroll
            for( int y=0;y<4;y++)
                Bxp[y]= Bb[j*4+j1+1][tyr+y*16]; 
#pragma unroll
            for( int y=0;y<4;y++)
                Axs[y] = Abs[txr+y*16][j*4+j1+1] ;
        }// j1 - loop
#pragma unroll 
        for( int x=0;x<4;x++)
#pragma unroll 
            for( int y=0;y<4;y++)
                Cb[x*4+y]  += Axs[x]*Bxp[y];	
    }// j - loop 
#pragma unroll
    for(j=1;j<4;j++){
#pragma unroll
        for(  int y=0;y<4;y++)
            Axs[y] = Abs[txr+y*16][j*4] ;
#pragma unroll
        for( int y=0;y<4;y++)
            Bxp[y]= Bb[j*4][tyr+y*16]; 
#pragma unroll 
        for( int j1=0;j1<3;j1++){
#pragma unroll 
            for( int x=0;x<4;x++)
#pragma unroll 
                for( int y=0;y<4;y++)
                    Cb[x*4+y]  += Axs[x]*Bxp[y];	
#pragma unroll
            for( int y=0;y<4;y++)
                Bxp[y]= Bb[j*4+j1+1][tyr+y*16]; 
#pragma unroll
            for( int y=0;y<4;y++)
                Axs[y] = Abs[txr+y*16][j*4+j1+1] ;
        }// j1 - loop
#pragma unroll 
        for( int x=0;x<4;x++)
#pragma unroll 
            for( int y=0;y<4;y++)
                Cb[x*4+y]  += Axs[x]*Bxp[y];	
    }// j - loop 
	//C += ibx  + txr*4 + __mul24 ( tyr*4 + iby ,ldc);
	C += txr + ibx  + __mul24 (tyr +  iby ,ldc);
#pragma unroll
    for( int y=0;y<4;y++){
#pragma unroll
        for( int x=0;x<4;x++){
            //C[txr+tyr*ldc] =alpha*Cb[y+x*4] + beta * C[txr+tyr*ldc];
            C[x*16] =alpha*Cb[y+x*4] + beta * C[x*16];
            //C[txr+x*16+y*ldc*16+tyr*ldc] =alpha*Cb[y+x*4] + beta * C[txr+x*16+y*ldc*16+tyr*ldc];
            //C+=16;
        }
        //C-=16*4;
        C+=ldc*16;
	}

    /*
      #pragma unroll 16
      for ( int i = 0; i < 16; i++, C += ldc) {
      C[0] =alpha*Cb[i] + beta * C[0];
      }
    */

}


#if 0
extern "C" void
magmablas_fermi_sgemm_kernel_N_N_64_64_16_64_4(char TRANSA, char TRANSB, 
 int m , int n , int k , float alpha, float *A , int lda , 
float *B, int ldb , float beta , float *C, int ldc ) 
{
        dim3 threads( 64, 4 );
        dim3 grid(m/64+(m%64!=0),n/64+(n%64!=0));
        fermi_sgemm_kernel_N_N_64_64_16_64_4<<< grid, threads >>> ( C, A, B, m, n, k, lda , ldb , ldc , alpha , beta ) ;
}
#endif

