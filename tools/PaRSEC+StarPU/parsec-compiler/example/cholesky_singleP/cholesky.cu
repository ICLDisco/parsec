#include <starpu.h>
#include <starpu_cuda.h>
#include "cholesky.h"
#include "parsec.h"

extern "C" void hook_of_cholesky_GEMM_cuda_wrapper(void *buffers[], void *cl_arg)
{
    struct func_args *args = (struct func_args*) cl_arg;
/* Retrieving global variables */
    int NB = *((int*) args->glob[0]);
    int SIZE = *((int*) args->glob[1]);
    PLASMA_enum uplo = *((PLASMA_enum*) args->glob[2]);
    int* INFO = *((int**) args->glob[3]);



 /* Parameters */ 
    void *A =  get_data_handle_of(buffers[0]); (void) A;
    void *B =  get_data_handle_of(buffers[1]); (void) B;
    void *C =  get_data_handle_of(buffers[2]); (void) C;
 /* Local variables */ 
    int k = args->var[0]; (void) k;
    int m = args->var[1]; (void) m;
    int n = args->var[2]; (void) n;


#line 189 "cholesky.jdf"
    if( uplo == PlasmaLower ){
        cublasSgemm(PlasmaNoTrans, PlasmaTrans, NB, NB, NB,
    	   	    (float)  -1.0,
		    (float*) (B), NB,
       		    (float*) (A), NB,
    	 	    (float)  1.0,
		    (float*) (C), NB);
    } else {
        cublasSgemm(PlasmaTrans, PlasmaNoTrans, NB, NB, NB,
      	 	    (float)  -1.0,
	  	    (float *) (A), NB,
       		    (float *) (B), NB,
    	 	    (float)   1.0,
		    (float *) (C), NB);
    }
    cudaDeviceSynchronize();

#line 957 "cholesky.c"
    free(cl_arg);
}

extern "C" void hook_of_cholesky_HERK_cuda_wrapper(void *buffers[], void *cl_arg)
{
    struct func_args *args = (struct func_args*) cl_arg;
/* Retrieving global variables */
    int NB = *((int*) args->glob[0]);
    int SIZE = *((int*) args->glob[1]);
    PLASMA_enum uplo = *((PLASMA_enum*) args->glob[2]);
    int* INFO = *((int**) args->glob[3]);



 /* Parameters */ 
    void *A =  get_data_handle_of(buffers[0]); (void) A;
    void *T =  get_data_handle_of(buffers[1]); (void) T;
 /* Local variables */ 
    int k = args->var[0]; (void) k;
    int n = args->var[1]; (void) n;


#line 138 "cholesky.jdf"
    if( uplo == PlasmaLower ) {
        cublasSsyrk(PlasmaLower, PlasmaNoTrans, NB, NB,		
	            (float)-1.0, 
		    (float *) (A), NB, 
	  	    (float) 1.0, 
       		    (float *) (T), NB);
    } else {
       	cublasSsyrk(PlasmaUpper, PlasmaTrans, NB, NB,		
	            (float)-1.0, 
		    (float *) (A), NB, 
	  	    (float) 1.0, 
       		    (float *) (T), NB);
    }
    cudaDeviceSynchronize();

#line 1669 "cholesky.c"
    free(cl_arg);
}

extern "C" void hook_of_cholesky_TRSM_cuda_wrapper(void *buffers[], void *cl_arg)
{
    struct func_args *args = (struct func_args*) cl_arg;
/* Retrieving global variables */
    int NB = *((int*) args->glob[0]);
    int SIZE = *((int*) args->glob[1]);
    PLASMA_enum uplo = *((PLASMA_enum*) args->glob[2]);
    int* INFO = *((int**) args->glob[3]);



 /* Parameters */ 
    void *T =  get_data_handle_of(buffers[0]); (void) T;
    void *C =  get_data_handle_of(buffers[1]); (void) C;
 /* Local variables */ 
    int m = args->var[0]; (void) m;
    int k = args->var[1]; (void) k;


#line 89 "cholesky.jdf"
    if( uplo == PlasmaLower ) {
    	cublasStrsm((char)PlasmaRight, (char)PlasmaLower,
	            (char)PlasmaTrans, (char)PlasmaNonUnit, NB, NB,
    	 	    (float) 1.0,
		    (float *) (T), NB, 
		    (float *) (C), NB); 

    } else {
      	cublasStrsm((char)PlasmaLeft, (char)PlasmaUpper,
		    (char)PlasmaTrans, (char)PlasmaNonUnit, NB, NB,
		    (float) 1.0,
		    (float *) (T), NB, 
		    (float *) (C), NB); 
   
    }
    cudaDeviceSynchronize();

#line 2524 "cholesky.c"
    free(cl_arg);
}

extern "C" void hook_of_cholesky_POTRF_cuda_wrapper(void *buffers[], void *cl_arg)
{
    struct func_args *args = (struct func_args*) cl_arg;
/* Retrieving global variables */
    int NB = *((int*) args->glob[0]);
    int SIZE = *((int*) args->glob[1]);
    PLASMA_enum uplo = *((PLASMA_enum*) args->glob[2]);
    int* INFO = *((int**) args->glob[3]);



 /* Parameters */ 
    void *T =  get_data_handle_of(buffers[0]); (void) T;
 /* Local variables */ 
    int k = args->var[0]; (void) k;


#line 47 "cholesky.jdf"
    if(uplo == PlasmaLower)
    	    magma_spotrf_gpu('L', NB, (float*) T, NB, INFO);
    else
            magma_spotrf_gpu('U', NB, (float*) T, NB, INFO);

     cudaDeviceSynchronize();

#line 3100 "cholesky.c"
    free(cl_arg);
}

