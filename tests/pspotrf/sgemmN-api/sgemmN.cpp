#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"

inline float mmax( float a, float b ) { return a > b ? a : b; }

//
//	generic functions first
//
#define BEGIN_TIMING(time)	\
{	\
	unsigned int n_iterations;	\
	for( n_iterations = 1; n_iterations < 0x80000000; n_iterations *= 2 )	\
	{	\
		cuCtxSynchronize();		\
		Q( cuEventRecord( start, 0 ) );	\
		for( unsigned int iteration = 0; iteration < n_iterations; iteration++ ){

#define END_TIMING(time,timer_tol) }\
		Q( cuEventRecord( end, 0 ) );	\
		Q( cuEventSynchronize( end ) );	\
		float sec;	\
		Q( cuEventElapsedTime( &sec, start, end ) ); \
		time = sec/1e3f;	\
		if (time >= timer_tol)	\
			break;	\
	}	\
	time /= n_iterations;\
}

void error( const char *message )
{
	fprintf( stderr, "ERROR: %s\n", message );
	exit (1);
}

void cuerror( const char* cuerror, const char *message )
{
	fprintf( stderr, "ERROR: %s (%s)\n", message, cuerror );
	exit (1);
}

#define assert( condition, ... ) { if( !( condition ) ) error( __VA_ARGS__ ); }
#define cuassert( condition, ... ) { if( !( condition ) ) cuerror( __VA_ARGS__ ); }
inline void Q( cudaError_t status ) { cuassert( status == cudaSuccess, cudaGetErrorString((cudaError_t)status), "CUDA Runtime fails" ); }
inline void Q( cublasStatus status ){ cuassert( status == CUBLAS_STATUS_SUCCESS, cudaGetErrorString((cudaError_t)status), "CUBLAS fails" ); }
inline void Q( CUresult status ) { cuassert( status == CUDA_SUCCESS, cudaGetErrorString((cudaError_t)status), "CUDA Driver fails" ); }

CUmodule module;
CUfunction sgemmNN, sgemmNT;

struct sgemm_params_t
{
	CUdeviceptr A;
	int lda;
	CUdeviceptr B;
	int ldb;
	CUdeviceptr C;
	int ldc;
	int k;
	float alpha;
	float beta;
};

void setup_sgemm( int device )
{
        CUdevice dev;
        int major, minor;
        char sgemmfile[128];

        Q( cuDeviceGet( &dev, device ) );
        Q( cuDeviceComputeCapability( &major, &minor, dev ) );

        snprintf( sgemmfile, 128, "sgemmN-%d%d.cubin", major, minor );
        printf( "Load cubin from %s\n", sgemmfile );

	Q( cuModuleLoad( &module, sgemmfile ) );
	Q( cuModuleGetFunction( &sgemmNN, module, "sgemmNN" ) );
	Q( cuModuleGetFunction( &sgemmNT, module, "sgemmNT" ) );
	Q( cuFuncSetBlockShape( sgemmNN, 16, 4, 1 ) );
	Q( cuFuncSetBlockShape( sgemmNT, 16, 4, 1 ) );
	Q( cuParamSetSize( sgemmNN, sizeof(sgemm_params_t) ) );
	Q( cuParamSetSize( sgemmNT, sizeof(sgemm_params_t) ) );
}

void release_sgemm( )
{
	Q( cuModuleUnload( module ) );
}

extern "C" void ourSgemm( char transa, char transb, int m, int n, int k, float alpha, CUdeviceptr A, int lda, CUdeviceptr B, int ldb, float beta, CUdeviceptr C, int ldc )
{	
	sgemm_params_t params = { A, lda, B, ldb, C, ldc, k, alpha, beta };

	assert( transa == 'N' || transa == 'n', "unsupported value of 'transa' in ourSgemm()" );
	assert( transb == 'N' || transb == 'n' || transb == 'T' || transb == 't' || transb == 'C' || transb == 'c', "invalid value of 'transb' in ourSgemm()" );
	assert( (m%64) == 0 && (n%16) == 0, "unsupported dimensions of matrix C in ourSgemm()" );
	
	if( transb == 'N' || transb == 'n' )
	{
		assert( (k%16) == 0 && k > 0, "unsupported shared dimension in ourSgemm( 'N', 'N', ... )" );
		Q( cuParamSetv( sgemmNN, 0, &params, sizeof(params) ) );
		Q( cuLaunchGrid( sgemmNN, m/64, n/16 ) );
	}
	else
	{
		assert( (k%4) == 0 && k > 4, "unsupported shared dimension in ourSgemm( 'N', 'T', ... )" );
		Q( cuParamSetv( sgemmNT, 0, &params, sizeof(params) ) );
		Q( cuLaunchGrid( sgemmNT, m/64, n/16 ) );
	}
}	
	
//	
//	auxiliary routines
//	
void fill( float *A, int n, int maxi )
{	
	for( int j = 0; j < n; j++ )
		A[j] = float( (rand()%(maxi*2+1)) - maxi ) / ( maxi + 1.f );
}	

float diff( int m, int n, float *A, int lda, float *B, int ldb )
{
	float err = 0;
	for( int j = 0; j < n; j++ )
		for( int i = 0; i < m; i++ )
			err = mmax( err, fabs( A[i+j*lda] - B[i+j*ldb] ) );
	return err;
}


void printout_devices( )
{
	int ndevices;
	Q( cuDeviceGetCount( &ndevices ) );
	for( int idevice = 0; idevice < ndevices; idevice++ )
	{
		char name[200];
		unsigned int totalMem, clock;
		CUdevice dev;

		Q( cuDeviceGet( &dev, idevice ) );
		Q( cuDeviceGetName( name, sizeof(name), dev ) );
		Q( cuDeviceTotalMem( &totalMem, dev ) );
		Q( cuDeviceGetAttribute( (int*)&clock, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev ) );
		printf( "device %d: %s, %.1f MHz clock, %.1f MB memory\n", idevice, name, clock/1000.f, totalMem/1024.f/1024.f );
	}
}

//
//	main()
//
int main(int argc, char* argv[])
{	
	//init
	CUdevice dev;
	CUcontext context;
        int device = 0;
	
	Q( cuInit( 0 ) );
        if( argc > 1 ) {
            device = atoi(argv[1]);
        }
	//Q( cuDeviceGet( &dev, 1 ) ); 
        Q( cuDeviceGet( &dev, device ) );
	Q( cuCtxCreate( &context, 0, dev ) );
	Q( cudaSetDevice(device) );
	Q( cublasInit( ) );

	printout_devices( );
	
	//int device = 1;

        fprintf(stderr,"1\n");

	setup_sgemm(device );

	fprintf(stderr,"2\n");
	
	//set up
	//const int N = 4096;
	const int N = 5120+64;

	float *A = (float*)malloc( N*N*sizeof( float ) );
	float *B = (float*)malloc( N*N*sizeof( float ) );
	float *C = (float*)malloc( N*N*sizeof( float ) );
	
	assert(A!= NULL && B != NULL && C != NULL, "memory allocation error" );

	fill( A, N*N, 31 );
	fill( B, N*N, 31 );
	fill( C, N*N, 31 );

	fprintf(stderr,"3\n");

	CUevent start, end;
	Q( cuEventCreate( &start, 0 ) );
	Q( cuEventCreate( &end, 0 ) );

	float *cublas_result = (float*)malloc( N*N*sizeof( float ) );
	float *our_result = (float*)malloc( N*N*sizeof( float ) );
	
	assert( cublas_result != NULL && our_result != NULL, "memory allocation error" );
	
	//
	//	bench square matrices
	//
	//for( int i = 0; i < 2; i++ )
	for( int i = 1; i < 2; i++ )
	{
		const char transa = 'N';
		const char transb = i ? 'T' : 'N';

		printf( "\ntesting sgemm( '%c', '%c', n, n, n, ... )\n\n", transa, transb );
		
		const int nb = 64;
		printf( "   n   CUBLAS,Gflop/s   we,Gflop/s   \"error\"\n" );
       //         for( int idim = 2560; idim <=1024*5; idim +=128 )
	
		//for( int idim = 1; idim <= N/nb; idim = int((idim+1)*1.1) )
		for( int idim = 1; idim <= N/nb; idim++ )
		{
			int dim = idim*nb;
			//int dim = idim ; 
			//
			//	set up the parameters
			//
			const int m = dim;
			const int n = dim;
			const int k = dim;
			const int lda = m;
			const int ldb = m;
			const int ldc = m;
			const float alpha = 1;
			const float beta = -1;
			
			printf( "%5d ", n );
			
			//
			// compute with CUBLAS
			//
			{
				float *dA, *dB, *dC;
				//printf("A = %d ", dA);
				Q( cublasAlloc( N*N, sizeof(float), (void**)&dA ) );
                                //printf("= %d \n", dA+1);
				Q( cublasAlloc( N*N, sizeof(float), (void**)&dB ) );
				Q( cublasAlloc( N*N, sizeof(float), (void**)&dC ) );
				
				Q( cublasSetMatrix( m, k, sizeof( float ), A, lda, dA, lda ) );
				Q( cublasSetMatrix( k, n, sizeof( float ), B, ldb, dB, ldb ) );
				Q( cublasSetMatrix( m, n, sizeof( float ), C, ldc, dC, ldc ) );
				cublasSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
				Q( cublasGetError( ) );
				Q( cublasGetMatrix( m, n, sizeof( float ), dC, ldc, cublas_result, ldc ) );
				
				double cublas_time;
				//cublasSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
				BEGIN_TIMING( cublas_time );
					cublasSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
				END_TIMING( cublas_time, 0.1 );
				
				cublasFree( dA );
				cublasFree( dB );
				cublasFree( dC );
				
				double cublas_gflops = 2.*m*n*k/cublas_time/1e9;
				printf( "%11.2f ", cublas_gflops );
			}

			//
			// compute with our routine
			//
			{
				CUdeviceptr dA, dB, dC;
				// printf("A = %d ", dA);
				Q( cuMemAlloc( &dA, N*N*sizeof(float) ) );
				// printf("= %d \n", dA+1);
				Q( cuMemAlloc( &dB, N*N*sizeof(float) ) );
				Q( cuMemAlloc( &dC, N*N*sizeof(float) ) );
				
				Q(cuMemcpyHtoD( dA, A, lda*k*sizeof(float) ) );
				Q(cuMemcpyHtoD( dB, B, ldb*n*sizeof(float) ) );
				Q(cuMemcpyHtoD( dC, C, ldc*n*sizeof(float) ) );
				ourSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
				Q( cuMemcpyDtoH( our_result, dC, ldc*n*sizeof(float) ) );
				
				double our_time;
				//ourSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
				BEGIN_TIMING( our_time );
					ourSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
				END_TIMING( our_time, 0.1 );
				
				Q( cuMemFree( dA ) );
				Q( cuMemFree( dB ) );
				Q( cuMemFree( dC ) );
				
				double our_gflops = 2.*m*n*k/our_time/1e9;
				printf( "%14.2f ", our_gflops );
			}
			
			//
			//	check the difference in results
			//
			float difference = diff( m, n, cublas_result, ldc, our_result, ldc );
			printf( "%8g\n", difference );
		}
	}
	
	//
	//	shutdown
	//
		
	free( A );
	free( B );
	free( C );
	
	free( cublas_result );
	free( our_result );
		
	cuEventDestroy( start );
	cuEventDestroy( end );

	//setup_sgemm(device );
	Q( cuCtxDetach( context ) );
	Q( cublasShutdown( ) );
	
	return 0;
}	
