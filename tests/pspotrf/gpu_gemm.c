#include "dplasma_config.h"
#include "lifo.h"

#include "cuda.h"
#include "cublas.h"
#include "cuda_runtime_api.h"

#include <plasma.h>

#include <stdio.h>

typedef struct _gpu_device {
    dplasma_list_item_t item;
    CUcontext ctx;
    CUmodule hcuModule;
    CUfunction hcuFunction;
    int id;
    int executed_tasks;
    uint64_t transferred_data_in;
    uint64_t transferred_data_out;
    dplasma_atomic_lifo_t* gpu_mem_lifo;
} gpu_device_t;

typedef struct _gpu_elem {
  dplasma_list_item_t item;
  CUdeviceptr gpu_mem;
} gpu_elem_t;

#define DPLASMA_CUDA_CHECK_ERROR( STR, ERROR, CODE )                    \
    do {                                                                \
        cudaError_t cuda_error = (ERROR);                               \
        if( cudaSuccess != cuda_error ) {                               \
            printf( "%s:%d %s%s\n", __FILE__, __LINE__,                 \
                    (STR), cudaGetErrorString(cuda_error) );            \
            CODE;                                                       \
        }                                                               \
    } while(0)

#define DPLASMA_USE_GPUS        2
#define DPLASMA_CONTEXT_PER_GPU 2

static dplasma_atomic_lifo_t gpu_devices;
volatile int32_t cpu_counter = 0;
static int ndevices = 0;

int spotrf_cuda_init( int use_gpu, int NB )
{
    cublasStatus cublas_status;
    int i, j, k, hcuDevice;

    if(use_gpu != -1){
        cuInit(0);

        dplasma_atomic_lifo_construct(&gpu_devices);
        cuDeviceGetCount( &ndevices );

        if( (-1 != DPLASMA_USE_GPUS) && (ndevices > DPLASMA_USE_GPUS) )
            ndevices = DPLASMA_USE_GPUS;

        if( 0 == ndevices ) {
            return -1;
        }
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
            for( j = 0; j < DPLASMA_CONTEXT_PER_GPU; j++ ) {
                cudaError_t cuda_status;

                gpu_device = (gpu_device_t*)malloc(sizeof(gpu_device_t));
                DPLASMA_LIST_ITEM_SINGLETON( &(gpu_device->item) );

    	        /* cuCtxCreate: Function works on floating contexts and current context */
                status = cuCtxCreate( &(gpu_device->ctx), 0 /*CU_CTX_BLOCKING_SYNC*/, hcuDevice );
                DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuCtxCreate ", status,
                                          {free(gpu_device); return -1;} );

                {
                    char *module_path = "./mysgemm.cubin";

                    status = cuModuleLoad(&(gpu_device->hcuModule), module_path);
                    if ( CUDA_SUCCESS != status ) {
                        printf( "cuModuleLoad failed %d\n", status );
                    }
                    
                    status = cuModuleGetFunction( &(gpu_device->hcuFunction), gpu_device->hcuModule, "_Z7sgemmNTPKfiS0_iPfiiff" );
                    if ( CUDA_SUCCESS != status ) {
                        printf( "cuModuleGetFunction(%s) failed %d\n", "_Z7sgemmNTPKfiS0_iPfiiff", status );
                    }
                    cuFuncSetBlockShape( gpu_device->hcuFunction, 16, 4, 1 );
                }

                /**
                 * Prepare the reusable memory on the GPU.
                 */
                gpu_mem_lifo = (dplasma_atomic_lifo_t*)malloc(sizeof(dplasma_atomic_lifo_t));
                dplasma_atomic_lifo_construct(gpu_mem_lifo);

                for( k = 0; k < 10; k++ ) {
                    gpu_elem_t* gpu_elem = (gpu_elem_t*)malloc(sizeof(gpu_elem_t));
                    DPLASMA_LIST_ITEM_SINGLETON( &(gpu_elem->item) );

                    cuda_status = cuMemAlloc( &(gpu_elem->gpu_mem), NB*NB*sizeof(float));
                    DPLASMA_CUDA_CHECK_ERROR( "cuMemAlloc ", cuda_status,
                                              {use_gpu = 0; return -1;} );
                    dplasma_atomic_lifo_push( gpu_mem_lifo, (dplasma_list_item_t*)gpu_elem );
                }
                gpu_device->id  = i;
                gpu_device->gpu_mem_lifo = gpu_mem_lifo;
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
        uint64_t *data_in, *data_out, total_data_in = 0, total_data_out = 0;
        float gtotal = 0.0, best_data_in, best_data_out;
        char *data_in_unit, *data_out_unit;

        /* GPU counter for GEMM / each */
        gpu_counter = (int*)calloc(ndevices, sizeof(int));
        data_in     = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
        data_out    = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
        while( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&gpu_devices)) ) {
            status = cuCtxPushCurrent( gpu_device->ctx );
            DPLASMA_CUDA_CHECK_ERROR( "(FINI) cuCtxPushCurrent ", status,
                                      {continue;} );
            /* Save the statistics */
            gpu_counter[gpu_device->id] += gpu_device->executed_tasks;
            data_in[gpu_device->id]     += gpu_device->transferred_data_in;
            data_out[gpu_device->id]    += gpu_device->transferred_data_out;

            /**
             * Release the GPU memory.
             */
            while( NULL != (gpu_elem = (gpu_elem_t*)dplasma_atomic_lifo_pop( gpu_device->gpu_mem_lifo )) ) {
                cuMemFree( gpu_elem->gpu_mem );
                free( gpu_elem );
            }
            status = cuCtxDestroy( gpu_device->ctx );
            DPLASMA_CUDA_CHECK_ERROR( "(FINI) cuCtxDestroy ", status,
                                      {continue;} );
            free(gpu_device);
        }
        /* Print statisitics */
        for( i = 0; i < ndevices; i++ ) {
            total += gpu_counter[i];
            total_data_in += data_in[i];
            total_data_out += data_out[i];
        }
        gtotal = total + cpu_counter;
        printf("------------------------------------------------------------------------------\n");
        printf("|PU       |  # GEMM   |    %%   |   Data In   |    %%   |   Data Out  |    %%   |\n");
        printf("|---------|-----------|--------|-------------|--------|-------------|--------|\n");
        for( i = 0; i < ndevices; i++ ) {
            compute_best_unit( data_in[i],  &best_data_in, &data_in_unit );
            compute_best_unit( data_out[i], &best_data_out, &data_out_unit );
            printf("|GPU:  %2d |%10d | %6.2f |%10.2f%2s | %6.2f |%10.2f%2s | %6.2f |\n",
                   i, gpu_counter[i], (gpu_counter[i]/gtotal)*100.00,
                   best_data_in, data_in_unit, (((float)data_in[i]) / total_data_in) * 100.0,
                   best_data_out, data_out_unit, (((float)data_out[i]) / total_data_out) * 100.0 );
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

int gpu_sgemm( int uplo, void* A, void* B, void* C, int NB )
{
    gpu_elem_t *gpu_elem_A = NULL, *gpu_elem_B = NULL, *gpu_elem_C = NULL;
    CUdeviceptr d_A, d_B, d_C;
    gpu_device_t* gpu_device;
    int offset, return_code = -1;  /* by default suppose an error */
    void* ptr;

    if( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&gpu_devices)) ) {
        CUstream stream;
        cudaError_t status;
        float alpha = -1.0, beta = 1.0;

        status = cuCtxPushCurrent(gpu_device->ctx);
        DPLASMA_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                  {goto return_error;} );

        gpu_elem_A = (gpu_elem_t*)dplasma_atomic_lifo_pop( gpu_device->gpu_mem_lifo );
        d_A = gpu_elem_A->gpu_mem;
        gpu_elem_B = (gpu_elem_t*)dplasma_atomic_lifo_pop( gpu_device->gpu_mem_lifo );
        d_B = gpu_elem_B->gpu_mem;
        gpu_elem_C = (gpu_elem_t*)dplasma_atomic_lifo_pop( gpu_device->gpu_mem_lifo );
        d_C = gpu_elem_C->gpu_mem;
        

        /*cuStreamCreate(&stream, 0);*/
        /* Push A into the GPU */
        status = cuMemcpyHtoD( d_A, A, sizeof(float)*NB*NB );
        DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyHtoD to device (d_A) ", status, 
                                  {printf("<<%p>>\n", (void*)(long)d_A); goto release_and_return_error;} );
        gpu_device->transferred_data_in += sizeof(float)*NB*NB;

        /* Push B into the GPU */
        status = cuMemcpyHtoD( d_B, B, sizeof(float)*NB*NB );
        DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyHtoD to device (d_B) ", status,
                                  {printf("<<%p>>\n", (void*)(long)d_B); goto release_and_return_error;} );
        gpu_device->transferred_data_in += sizeof(float)*NB*NB;

        /* Push C into the GPU */
        status = cuMemcpyHtoD( d_C, C, sizeof(float)*NB*NB );
        DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyHtoD to device (d_C) ", status,
                                  {printf("<<%p>>\n", (void*)(long)d_C); goto release_and_return_error;} );
        gpu_device->transferred_data_in += sizeof(float)*NB*NB;

        /* Wait until all data are on the GPU */
        /*status = cuStreamSynchronize(stream);
        DPLASMA_CUDA_CHECK_ERROR( "cuStreamSynchronize", status,
        {goto release_and_return_error;} );*/

#if 0
        if(uplo == PlasmaLower) {
            /*cublasSgemm('N','T', NB, NB, NB, -1.0, (float*)(long)d_B, NB, (float*)(long)d_A, NB, 1.0, (float*)(long)d_C, NB );*/
            cublasSgemm('N','T', NB, NB, NB, -1.0, (float*)&d_B, NB, (float*)&d_A, NB, 1.0, (float*)&d_C, NB );
        } else {
            /*cublasSgemm('T','N', NB, NB, NB, -1.0, (float*)(long)d_A, NB, (float*)(long)d_B, NB, 1.0, (float*)(long)d_C, NB );*/
            cublasSgemm('T','N', NB, NB, NB, -1.0, (float*)&d_A, NB, (float*)&d_B, NB, 1.0, (float*)&d_C, NB );
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
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, NB );
        CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_B );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, NB );
        CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_C );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, NB );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, NB );
        CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, alpha );
        CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, beta );
        cuParamSetSize( gpu_device->hcuFunction, offset );

        // cuLaunch: we kick off the CUDA
        status = cuLaunch( gpu_device->hcuFunction );
        if ( CUDA_SUCCESS != status ) {
            printf( "cuLaunch failed %d\n", status );
            return -1;
        }

        /* Pop C from the GPU */
        status = cuMemcpyDtoH( C , d_C, sizeof(float)*NB*NB );
        DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyDtoH from device (d_C) ", status,
                                  {printf("<<%p>>\n", d_C); goto release_and_return_error;} );
        gpu_device->transferred_data_out += sizeof(float)*NB*NB;
        /* Wait until the data is back on the memory */
        /*status = cuStreamSynchronize(stream);
        DPLASMA_CUDA_CHECK_ERROR( "cuStreamSynchronize", status,
                                  {goto release_and_return_error;} );
                                  cuStreamDestroy(stream);*/

        /* Everything went fine so far, the result is correct and back in the main memory */
        return_code = 0;
        gpu_device->executed_tasks++;

    release_and_return_error:
        if( NULL != gpu_elem_C )
            dplasma_atomic_lifo_push( gpu_device->gpu_mem_lifo, (dplasma_list_item_t*)gpu_elem_C );
        if( NULL != gpu_elem_B )
            dplasma_atomic_lifo_push( gpu_device->gpu_mem_lifo, (dplasma_list_item_t*)gpu_elem_B );
        if( NULL != gpu_elem_A )
            dplasma_atomic_lifo_push( gpu_device->gpu_mem_lifo, (dplasma_list_item_t*)gpu_elem_A );

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
