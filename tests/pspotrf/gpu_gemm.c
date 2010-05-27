/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma_config.h"
#include "profiling.h"
#include "gpu_data.h"


#include <stdio.h>
#include <cublas.h>
#include <plasma.h>

#include "data_management.h"
extern DPLASMA_desc ddescA;

#define DPLASMA_CUDA_CHECK_ERROR( STR, ERROR, CODE )                    \
    do {                                                                \
        cudaError_t cuda_error = (ERROR);                               \
        if( cudaSuccess != cuda_error ) {                               \
            printf( "%s:%d %s%s\n", __FILE__, __LINE__,                 \
                    (STR), cudaGetErrorString(cuda_error) );            \
            CODE;                                                       \
        }                                                               \
    } while(0)

#define DPLASMA_CONTEXT_PER_GPU 1
#define DPLASMA_CONTEXT_OVERLAP 0
static dplasma_atomic_lifo_t gpu_devices;
int dplasma_show_detailed_capabilities = 0;
volatile int32_t cpu_counter = 0;
volatile int32_t entering = 0;
volatile int32_t computing = 0;
static int ndevices = 0;
dplasma_thread_profiling_t *thread_movein;
dplasma_thread_profiling_t *thread_compute;
dplasma_thread_profiling_t *thread_moveout; 
int movein_key_start;
int movein_key_end;
int compute_key_start;
int compute_key_end;
int moveout_key_start;
int moveout_key_end;


int spotrf_cuda_init( int use_gpu )
{
    cublasStatus cublas_status;
    int i, j, k, hcuDevice;

    if(use_gpu != -1){
        cuInit(0);

        dplasma_atomic_lifo_construct(&gpu_devices);
        cuDeviceGetCount( &ndevices );

        if( ndevices > use_gpu )
            ndevices = use_gpu;

        if( 0 == ndevices ) {
            return -1;
        }
	ndevices = 1;
        for( i = 0; i < ndevices; i++ ) {
            dplasma_atomic_lifo_t* gpu_mem_lifo;
            gpu_device_t* gpu_device;
            CUdevprop devProps;
            char szName[256];
            CUresult status;
            int major, minor;

            status = cuDeviceGet( &hcuDevice, i );
            DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {use_gpu = 0; return -1;} );

            status = cuDeviceGetName( szName, 256, hcuDevice );
            DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGetName ", status, {use_gpu = 0; return -1;} );

            status = cuDeviceComputeCapability( &major, &minor, hcuDevice);
            DPLASMA_CUDA_CHECK_ERROR( "cuDeviceComputeCapability ", status, {use_gpu = 0; return -1;} );

            status = cuDeviceGetProperties( &devProps, hcuDevice );
            DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGetProperties ", status, {use_gpu = 0; return -1;} );

            printf("Device %d (capability %d.%d): %s\n", i, major, minor, szName );
            if( dplasma_show_detailed_capabilities ) {
                printf("\tsharedMemPerBlock  : %d\n", devProps.sharedMemPerBlock );
                printf("\tmaxThreadsPerBlock : %d\n", devProps.maxThreadsPerBlock );
                printf("\tmaxThreadsDim      : %d\n", devProps.maxThreadsDim );
                printf("\tconstantMemory     : %d\n", devProps.totalConstantMemory );
                printf("\tmemPitch           : %d\n", devProps.memPitch );
                printf("\tregsPerBlock       : %d\n", devProps.regsPerBlock );
                printf("\tclockRate          : %d\n", devProps.clockRate );
#if 0
                > 1.2 printf("\tdeviceOverlap    : %ssupported\n", (devProps.deviceOverlap ? "" : "not ") );
                > 2.0 printf("\tconcurrentKernels: %ssupported\n", (devProps.concurrentKernels ? "" : "not ") );
#endif
            }
            for( j = 0; j < DPLASMA_CONTEXT_PER_GPU; j++ ) {
                cudaError_t cuda_status;

		dplasma_profiling_add_dictionary_keyword("movein","#00FF00",&movein_key_start,&movein_key_end);
		dplasma_profiling_add_dictionary_keyword("compute","#00FF00",&compute_key_start,&compute_key_end);
		dplasma_profiling_add_dictionary_keyword("moveout","#00FF00",&moveout_key_start,&moveout_key_end);

		thread_movein = dplasma_profiling_thread_init(2*4096,"GPU HtoD: %d",j);
		thread_compute = dplasma_profiling_thread_init(2*4096,"GPU Computing %d",j);
		thread_moveout = dplasma_profiling_thread_init(2*4096,"GPU DtoH %d",j);

                gpu_device = (gpu_device_t*)malloc(sizeof(gpu_device_t));
                DPLASMA_LIST_ITEM_SINGLETON( &(gpu_device->item) );

    	        /* cuCtxCreate: Function works on floating contexts and current context */
                status = cuCtxCreate( &(gpu_device->ctx), 0 /*CU_CTX_BLOCKING_SYNC*/, hcuDevice );
                DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuCtxCreate ", status,
                                          {free(gpu_device); return -1;} );

                {
                    char *module_path = "./sgemm.cubin";

                    status = cuModuleLoad(&(gpu_device->hcuModule), module_path);
                    if ( CUDA_SUCCESS != status ) {
                        printf( "cuModuleLoad failed %d\n", status );
                    }
                    
                    status = cuModuleGetFunction( &(gpu_device->hcuFunction), gpu_device->hcuModule, "sgemmNT" );
                    if ( CUDA_SUCCESS != status ) {
                        printf( "cuModuleGetFunction(%s) failed %d\n", "sgemmNT", status );
                    }
                    cuFuncSetBlockShape( gpu_device->hcuFunction, 16, 4, 1 );
                }

                /**
                 * Prepare the reusable memory on the GPU.
                 */
                dplasma_data_map_init( gpu_device, &ddescA );

                for( k = 0; k < 144; k++ ) {
                    gpu_elem_t* gpu_elem = (gpu_elem_t*)malloc(sizeof(gpu_elem_t));
                    dplamsa_linked_list_item_construct( (dplasma_list_item_t*)gpu_elem );

                    cuda_status = cuMemAlloc( &(gpu_elem->gpu_mem), ddescA.mb*ddescA.nb*sizeof(float));
                    DPLASMA_CUDA_CHECK_ERROR( "cuMemAlloc ", cuda_status,
                                              {use_gpu = 0; return -1;} );
                    gpu_elem->memory = NULL;
                    dplasma_linked_list_add_tail( gpu_device->gpu_mem_lru, (dplasma_list_item_t*)gpu_elem );
                }
                gpu_device->id  = i;
                gpu_device->executed_tasks = 0;
                gpu_device->transferred_data_in = 0;
                gpu_device->transferred_data_out = 0;

                status = cuCtxPopCurrent(NULL);
                DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                          {free(gpu_device); return -1;} );

                dplasma_atomic_lifo_push( &(gpu_devices), (dplasma_list_item_t*)gpu_device );
            }
        }

        return 0;
    }
    return -1;
}

static void compute_best_unit( uint64_t length, float* updated_value, char** best_unit )
{
    float measure = (float)length;

    *best_unit = "B";
    if( measure > 1024.0 ) { /* 1KB */
        *best_unit = "KB";
        measure = measure / 1024.0;
        if( measure > 1024.0 ) { /* 1MB */
            *best_unit = "MB";
            measure = measure / 1024.0;
            if( measure > 1024.0 ) {
                *best_unit = "GB";
                measure = measure / 1024.0;
            }
        }
    }
    *updated_value = measure;
    return;
}

int spotrf_cuda_fini( int use_gpu )
{
    cudaError_t status;

    if (use_gpu == 1) {
        gpu_elem_t* gpu_elem;
        gpu_device_t* gpu_device;
        int total = 0, *gpu_counter, i, overlap_counter;
        uint64_t *transferred_in, *transferred_out, total_data_in = 0, total_data_out = 0;
        uint64_t *required_in, *required_out;
        float gtotal = 0.0, best_data_in, best_data_out;
        char *data_in_unit, *data_out_unit;

        /* GPU counter for GEMM / each */
        gpu_counter = (int*)calloc(ndevices, sizeof(int));
        transferred_in  = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
        transferred_out = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
        required_in     = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
        required_out    = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
        while( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&gpu_devices)) ) {
            status = cuCtxPushCurrent( gpu_device->ctx );
            DPLASMA_CUDA_CHECK_ERROR( "(FINI) cuCtxPushCurrent ", status,
                                      {continue;} );
            /* Save the statistics */
            gpu_counter[gpu_device->id]     += gpu_device->executed_tasks;
            transferred_in[gpu_device->id]  += gpu_device->transferred_data_in;
            transferred_out[gpu_device->id] += gpu_device->transferred_data_out;
            required_in[gpu_device->id]     += gpu_device->required_data_in;
            required_out[gpu_device->id]    += gpu_device->required_data_out;

            /**
             * Release the GPU memory.
             */
            while( NULL != (gpu_elem = (gpu_elem_t*)dplasma_linked_list_remove_head( gpu_device->gpu_mem_lru )) ) {
                cuMemFree( gpu_elem->gpu_mem );
                free( gpu_elem );
            }
            status = cuCtxDestroy( gpu_device->ctx );
            DPLASMA_CUDA_CHECK_ERROR( "(FINI) cuCtxDestroy ", status,
                                      {continue;} );
            free(gpu_device->gpu_mem_lru);
            free(gpu_device);
        }
        /* Print statisitics */
        for( i = 0; i < ndevices; i++ ) {
            total += gpu_counter[i];
            total_data_in  += transferred_in[i];
            total_data_out += transferred_out[i];
        }
        if( 0 == total_data_in ) total_data_in = 1;
        if( 0 == total_data_out ) total_data_out = 1;

        gtotal = total + cpu_counter;
        printf("------------------------------------------------------------------------------\n");
        printf("|PU       |  # GEMM   |    %%   |   Data In   |    %%   |   Data Out  |    %%   |\n");
        printf("|---------|-----------|--------|-------------|--------|-------------|--------|\n");
        for( i = 0; i < ndevices; i++ ) {
            compute_best_unit( transferred_in[i],  &best_data_in, &data_in_unit );
            compute_best_unit( transferred_out[i], &best_data_out, &data_out_unit );
            printf("|GPU:  %2d |%10d | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
                   i, gpu_counter[i], (gpu_counter[i]/gtotal)*100.00,
                   best_data_in, data_in_unit, (((float)transferred_in[i]) / required_in[i]) * 100.0,
                   best_data_out, data_out_unit, (((float)transferred_out[i]) / required_out[i]) * 100.0 );
        }
        printf("|---------|-----------|--------|-------------|--------|-------------|--------|\n");
        compute_best_unit( total_data_in,  &best_data_in, &data_in_unit );
        compute_best_unit( total_data_out, &best_data_out, &data_out_unit );
        printf("|All GPUs |%10d | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
               total, (total/gtotal)*100.00,
               best_data_in, data_in_unit, 100.0,
               best_data_out, data_out_unit, 100.0);
        printf("|All CPUs |%10d | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
               cpu_counter, (cpu_counter / gtotal)*100.00,
               0.0, " ", 0.0, 0.0, " ", 0.0);
        /*printf("|---------|-----------|--------|-------------|--------|-------------|--------|\n");
          printf("|Overlap  |%10d  %10.5s |\n", overlap_counter, "times");*/
        printf("------------------------------------------------------------------------------\n");
        free(gpu_counter);
        free(transferred_in);
        free(transferred_out);
        free(required_in);
        free(required_out);
    }
}

#define ALIGN_UP(OFFSET, ALIGN) \
    (OFFSET) = ((OFFSET) + (ALIGN) - 1) & ~((ALIGN) - 1)
#define CU_PUSH_POINTER( FUNCTION, OFFSET, PTR )                        \
        do {                                                            \
            void* __ptr = (void*)(size_t)(PTR);                         \
            ALIGN_UP((OFFSET), __alignof(void*));                       \
            cuParamSetv( (FUNCTION), (OFFSET), &__ptr, sizeof(void*));  \
            (OFFSET) += sizeof(void*);                                  \
        } while (0)
#define CU_PUSH_INT( FUNCTION, OFFSET, VALUE )                          \
        do {                                                            \
            ALIGN_UP((OFFSET), __alignof(int));                         \
            cuParamSeti( (FUNCTION), (OFFSET), (VALUE) );               \
            (OFFSET) += sizeof(int);                                    \
        } while (0)
#define CU_PUSH_FLOAT( FUNCTION, OFFSET, VALUE )                        \
        do {                                                            \
            ALIGN_UP((OFFSET), __alignof(float));                       \
            cuParamSetf( (FUNCTION), (OFFSET), (VALUE) );               \
            (OFFSET) += sizeof(float);                                  \
        } while (0)

int gpu_sgemm( int uplo, void* A, void* B, void* C, int k, int n, int m )
{
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_B = NULL, *gpu_elem_C = NULL;
    CUdeviceptr d_A, d_B, d_C;
    gpu_device_t* gpu_device;
    int offset, on_gpu, return_code = -1, tile_size;  /* by default suppose an error */
    void* ptr;

#if DPLASMA_CONTEXT_OVERLAP
    if(!(dplasma_atomic_cas(&entering,0,1) == 1))
	    return -1;
#endif

    if( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&gpu_devices)) ) {
	dplasma_profiling_trace(thread_movein, movein_key_start,0);
        CUstream stream;
        cudaError_t status;
        float alpha = -1.0, beta = 1.0;

        status = cuCtxPushCurrent(gpu_device->ctx);
        DPLASMA_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                  {goto return_error;} );
        tile_size = ddescA.mb*ddescA.nb*sizeof(float);

        /*cuStreamCreate(&stream, 0);*/
        on_gpu = dplasma_data_is_on_gpu(gpu_device, &ddescA, DPLASMA_READ, n, k, &gpu_elem_A);
        d_A = gpu_elem_A->gpu_mem;
        gpu_device->required_data_in += tile_size;
        if( !on_gpu ) {
            /* Push A into the GPU */
            status = cuMemcpyHtoD( d_A, A, tile_size );
            DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyHtoD to device (d_A) ", status, 
                                      {printf("<<%p>>\n", (void*)(long)d_A); goto release_and_return_error;} );
            gpu_device->transferred_data_in += tile_size;
        }

        on_gpu = dplasma_data_is_on_gpu(gpu_device, &ddescA, DPLASMA_READ, m, k, &gpu_elem_B);
        d_B = gpu_elem_B->gpu_mem;
        gpu_device->required_data_in += tile_size;
        if( !on_gpu ) {
            /* Push B into the GPU */
            status = cuMemcpyHtoD( d_B, B, tile_size );
            DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyHtoD to device (d_B) ", status,
                                      {printf("<<%p>>\n", (void*)(long)d_B); goto release_and_return_error;} );
            gpu_device->transferred_data_in += tile_size;
        }

        on_gpu = dplasma_data_is_on_gpu(gpu_device, &ddescA, DPLASMA_READ, m, n, &gpu_elem_C);
        d_C = gpu_elem_C->gpu_mem;
        gpu_device->required_data_in += tile_size;
        if( !on_gpu ) {
            /* Push C into the GPU */
            status = cuMemcpyHtoD( d_C, C, tile_size );
            DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyHtoD to device (d_C) ", status,
                                      {printf("<<%p>>\n", (void*)(long)d_C); goto release_and_return_error;} );
            gpu_device->transferred_data_in += tile_size;
        }
	dplasma_profiling_trace(thread_movein,movein_key_end,0);
        /* Wait until all data are on the GPU */
        /*status = cuStreamSynchronize(stream);
        DPLASMA_CUDA_CHECK_ERROR( "cuStreamSynchronize", status,
        {goto release_and_return_error;} );*/

#if 0
        if(uplo == PlasmaLower) {
            /*cublasSgemm('N','T', ddescA.nb, ddescA.nb, ddescA.nb, -1.0, (float*)(long)d_B, ddescA.nb, (float*)(long)d_A, ddescA.nb, 1.0, (float*)(long)d_C, ddescA.nb );*/
            cublasSgemm('N','T', ddescA.nb, ddescA.nb, ddescA.nb, -1.0, (float*)&d_B, ddescA.nb, (float*)&d_A, ddescA.nb, 1.0, (float*)&d_C, ddescA.nb );
        } else {
            /*cublasSgemm('T','N', ddescA.nb, ddescA.nb, ddescA.nb, -1.0, (float*)(long)d_A, ddescA.nb, (float*)(long)d_B, ddescA.nb, 1.0, (float*)(long)d_C, ddescA.nb );*/
            cublasSgemm('T','N', ddescA.nb, ddescA.nb, ddescA.nb, -1.0, (float*)&d_A, ddescA.nb, (float*)&d_B, ddescA.nb, 1.0, (float*)&d_C, ddescA.nb );
        }
        status = cublasGetError();
        if( CUBLAS_STATUS_SUCCESS != status ) {
            if( CUBLAS_STATUS_NOT_INITIALIZED == status ) {
                printf("if CUBLAS library was not initialized\n");
            } else if( CUBLAS_STATUS_INVALID_VALUE == status ) {
                printf("if m < 0, n < 0, or k < 0\n");
            } else if( CUBLAS_STATUS_EXECUTION_FAILED == status ) {
                printf("if function failed to launch on GPU\n");
            } else
                printf( "Unknown Error during the Sgemm\n" );
        }
#endif

        offset = 0;
        CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_A );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
        CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_B );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
        CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_C );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
        CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, alpha );
        CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, beta );
        cuParamSetSize( gpu_device->hcuFunction, offset );

	
        // cuLaunch: we kick off the CUDA
	dplasma_profiling_trace(thread_compute,compute_key_start,1);
        status = cuLaunch( gpu_device->hcuFunction );
        if ( CUDA_SUCCESS != status ) {
            printf( "cuLaunch failed %d\n", status );
            return -1;
        }
	dplasma_profiling_trace(thread_compute,compute_key_end,1);

	#if DPLASMA_CONTEXT_OVERLAP
		entering = 0;		
	#endif
        /* Pop C from the GPU */
	dplasma_profiling_trace(thread_moveout,moveout_key_start,2);
        gpu_device->required_data_out += tile_size;
        status = cuMemcpyDtoH( C , d_C, tile_size );
        DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyDtoH from device (d_C) ", status,
                                  {printf("<<%p>>\n", d_C); goto release_and_return_error;} );
        gpu_device->transferred_data_out += tile_size;
	dplasma_profiling_trace(thread_moveout,moveout_key_end,2);

        /* Wait until the data is back on the memory */
        /*status = cuStreamSynchronize(stream);
        DPLASMA_CUDA_CHECK_ERROR( "cuStreamSynchronize", status,
                                  {goto release_and_return_error;} );
                                  cuStreamDestroy(stream);*/

        /* Everything went fine so far, the result is correct and back in the main memory */
        return_code = 0;
        gpu_device->executed_tasks++;

    release_and_return_error:
        status = cuCtxPopCurrent(NULL);
        DPLASMA_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                  {goto return_error;} );
    return_error:
        dplasma_atomic_lifo_push(&gpu_devices, (dplasma_list_item_t*)gpu_device);

        return return_code;
    }
    dplasma_atomic_inc_32b(&cpu_counter);
    return -1;  /* fails to atomically get the ownership of the device */
}
