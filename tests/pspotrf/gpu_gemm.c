/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma_config.h"
#include "gpu_data.h"

#include <stdio.h>
#include <cublas.h>
#include <plasma.h>

#include "data_management.h"
extern DPLASMA_desc ddescA;

#define DPLASMA_CONTEXT_PER_GPU 1

#if DPLASMA_SMART_SCHEDULING
	float	cpu_usage = 4.0;		/* CPU core is slower than the fastest GPU 7.0 times */
	gpu_device_t* get_best_gpu(int);	/* Function to get best choice of avaiblable Contexts !! */
#else						/* I don't use gpu_devices anymore, I will be subset of gpu-array */
						/* gpu_array - list of GPU by order of their performance */
	dplasma_atomic_lifo_t gpu_devices;	
#endif
int dplasma_show_detailed_capabilities = 0;
volatile int32_t cpu_counter = 0;
int ndevices = 0;
#if defined(DPLASMA_PROFILING)
static int movein_key_start;
static int movein_key_end;
static int compute_key_start;
static int compute_key_end;
static int moveout_key_start;
static int moveout_key_end;
#endif  /* defined(PROFILING) */

int spotrf_cuda_init( int* puse_gpu )
{
    cublasStatus cublas_status;
    CUdevice hcuDevice;
    int i, j, k;

    if( (*puse_gpu) != -1){
        cuInit(0);
#if DPLASMA_SMART_SCHEDULING
	/* do not need to construct gpu_devices ! */
#else
        dplasma_atomic_lifo_construct(&gpu_devices);
#endif
        cuDeviceGetCount( &ndevices );

        if( ndevices > (*puse_gpu) )
            ndevices = (*puse_gpu);
        /* Update the number of GPU for the upper layer */
        *puse_gpu = ndevices;
        if( 0 == ndevices ) {
            return -1;
        }	
#if DPLASMA_SMART_SCHEDULING
	CUresult status;
	CUdevice hcuDevice_; /* use to compare */
	/* Choose GPU by requirement [Capability - sometimes > 1.3 because of needs of double precision ]*/
        int pi,pi_,tmp,major,minor;
	int rmajor=1, rminor=1;
	/* gpu_array - list of GPU which we're gonna use ! */
	gpu_array = (gpu_item_t*)calloc(ndevices, sizeof(gpu_item_t));
	j = 0;
	for(i = 0; i < ndevices; i++){
		status = cuDeviceGet( &hcuDevice, i );
		/* PASS requirement ?*/
		cuDeviceComputeCapability(&major, &minor, hcuDevice);
		if(major > rmajor || (major == rmajor && minor >= rminor)){
			gpu_array[j].gpu_id = i;
			
			/* Assign usage 
			 * the FASTEST will be 1.0 
			 * then the next would be 5.0 due to slower than the FASTEST 5.0 times */
			if(major == 1 & minor == 3){
				gpu_array[j].func1_usage = 1.0;
			}else{
				gpu_array[j].func1_usage = 2.0;
			}
			/* working status - would it divided by function ? , probably */
			gpu_array[j].working = 0;
			gpu_array[j].func1_current = 0;	/* function 1 designed for top priority*/
			gpu_array[j].func2_current = 0;	/* function 2 designed for low priority trying to catch left GPU from func1*/

			dplasma_atomic_lifo_construct(&(gpu_array[j].gpu_devices)); /* gpu_devies Context*/
			/* EACH GPU have their own stack , we don't mix up due to need of selecting smart choice while we have different cards */
			/* Also, if there's a Card that can have many Context, we will be able to control it properly */
			j++;
		}
	}
	/* number of device changed due to requirement */
	ndevices = j;
	
	/* Sort GPU by Multiprocessor count */
	/* gpu_array[ best -> worst ] */
	/* We will be able to choose best available , near 0 */
	if(ndevices > 1){
		for(i = 0; i < ndevices - 1 ; i++){
			for(j = 0; j < ndevices - 1; j++){
				status = cuDeviceGet( &hcuDevice, gpu_array[j].gpu_id );
				DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {*puse_gpu = 0; return -1;} );
				cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, hcuDevice);
				
				status = cuDeviceGet( &hcuDevice_, gpu_array[j+1].gpu_id );
				DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {*puse_gpu = 0; return -1;} );
				cuDeviceGetAttribute(&pi_, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, hcuDevice_);
				if(pi < pi_){
					tmp = gpu_array[j].gpu_id;
					gpu_array[j].gpu_id = gpu_array[j+1].gpu_id;
					gpu_array[j+1].gpu_id = tmp;
				}else if(pi == pi_){
					cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, hcuDevice);
					cuDeviceGetAttribute(&pi_, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, hcuDevice_);
					if(pi < pi_){
						tmp = gpu_array[j].gpu_id;
						gpu_array[j].gpu_id = gpu_array[j+1].gpu_id;
						gpu_array[j+1].gpu_id = tmp;
					}
				}
			}
		}
	}
	/* Setup default vaule */
	for( i = 0; i < ndevices ; i++){
		/* every GPU has their own queue fix at 10 */
		gpu_array[i].waiting = (int*)calloc(10, sizeof(int));
		for( j = 0; j < 10 - 1; j++){
			gpu_array[i].waiting[j] = 0;
		}
	}
#endif

	for( i = 0; i < ndevices; i++ ) {
            unsigned int total_mem, tile_size, memory_left_for_context, chunk_size;
            dplasma_atomic_lifo_t* gpu_mem_lifo;
            gpu_device_t* gpu_device;
            CUdevprop devProps;
            char szName[256];
            CUresult status;
            int major, minor;

#if DPLASMA_SMART_SCHEDULING
	    /* instead of go from ID 0 -> n - 1 
	     * we have to use ID which's in gpu_array 
	     * because they might not be in order due to sorting/reordering */
	    status = cuDeviceGet( &hcuDevice, gpu_array[i].gpu_id);
	    DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {*puse_gpu = 0; return -1;} );
#else
	    status = cuDeviceGet( &hcuDevice, i );
	    DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGet ", status, {*puse_gpu = 0; return -1;} );
#endif
            status = cuDeviceGetName( szName, 256, hcuDevice );
            DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGetName ", status, {*puse_gpu = 0; return -1;} );

            status = cuDeviceComputeCapability( &major, &minor, hcuDevice);
            DPLASMA_CUDA_CHECK_ERROR( "cuDeviceComputeCapability ", status, {*puse_gpu = 0; return -1;} );

            status = cuDeviceGetProperties( &devProps, hcuDevice );
            DPLASMA_CUDA_CHECK_ERROR( "cuDeviceGetProperties ", status, {*puse_gpu = 0; return -1;} );

#if DPLASMA_SMART_SCHEDULING
	    printf("Device %d (capability %d.%d): %s\n", gpu_array[i].gpu_id, major, minor, szName );
#else
            printf("Device %d (capability %d.%d): %s\n", i, major, minor, szName );
#endif
            if( dplasma_show_detailed_capabilities ) {
                printf("\tmaxThreadsPerBlock : %d\n", devProps.maxThreadsPerBlock );
                printf("\tmaxThreadsDim      : [%d %d %d]\n", devProps.maxThreadsDim[0],
                       devProps.maxThreadsDim[1], devProps.maxThreadsDim[2] );
                printf("\tmaxGridSize        : [%d %d %d]\n", devProps.maxGridSize[0],
                       devProps.maxGridSize[1], devProps.maxGridSize[2] );
                printf("\tsharedMemPerBlock  : %d\n", devProps.sharedMemPerBlock );
                printf("\tconstantMemory     : %d\n", devProps.totalConstantMemory );
                printf("\tSIMDWidth          : %d\n", devProps.SIMDWidth );
                printf("\tmemPitch           : %d\n", devProps.memPitch );
                printf("\tregsPerBlock       : %d\n", devProps.regsPerBlock );
                printf("\tclockRate          : %d\n", devProps.clockRate );
#if 0
                > 1.2 printf("\tdeviceOverlap    : %ssupported\n", (devProps.deviceOverlap ? "" : "not ") );
                > 2.0 printf("\tconcurrentKernels: %ssupported\n", (devProps.concurrentKernels ? "" : "not ") );
#endif
            }
            status = cuDeviceTotalMem( &total_mem, hcuDevice );
            DPLASMA_CUDA_CHECK_ERROR( "cuDeviceTotalMem ", status, {*puse_gpu = 0; return -1;} );

            for( j = 0; j < DPLASMA_CONTEXT_PER_GPU; j++ ) {
                cudaError_t cuda_status;

                gpu_device = (gpu_device_t*)malloc(sizeof(gpu_device_t));
                DPLASMA_LIST_ITEM_SINGLETON( &(gpu_device->item) );

    	        /* cuCtxCreate: Function works on floating contexts and current context */
                status = cuCtxCreate( &(gpu_device->ctx), 0 /*CU_CTX_BLOCKING_SYNC*/, hcuDevice );
                DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuCtxCreate ", status,
                                          {free(gpu_device); return -1;} );

                {
                    char *module_path = "./mysgemm_generated_sgemm.cu.o.cubin.txt";

                    status = cuModuleLoad(&(gpu_device->hcuModule), module_path);
                    DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuModuleLoad ", status,
                                              {
                                                  cuCtxDestroy( gpu_device->ctx );
                                                  free(gpu_device);
                                                  break;
                                              } );
                    
                    status = cuModuleGetFunction( &(gpu_device->hcuFunction), gpu_device->hcuModule, "sgemmNT" );
                    DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuModuleGetFunction ", status,
                                              {
                                                  cuCtxDestroy( gpu_device->ctx );
                                                  free(gpu_device);
                                                  break;
                                              } );
                    cuFuncSetBlockShape( gpu_device->hcuFunction, 16, 4, 1 );
                }

                /**
                 * Prepare the reusable memory on the GPU.
                 */
                dplasma_data_map_init( gpu_device, &ddescA );
                /**
                 * It appears that CUDA allocate the memory in chunks of 1MB,
                 * so we need to adapt to this.
                 */
                tile_size = ddescA.mb*ddescA.nb*sizeof(float);
                chunk_size = (tile_size + 1024*1024-1) & ~(1024*1024);
                memory_left_for_context = (total_mem - total_mem / 100) / DPLASMA_CONTEXT_PER_GPU;
                for( k = 0; tile_size < memory_left_for_context; k++ ) {
                    gpu_elem_t* gpu_elem = (gpu_elem_t*)malloc(sizeof(gpu_elem_t));
                    dplamsa_linked_list_item_construct( (dplasma_list_item_t*)gpu_elem );

                    cuda_status = cuMemAlloc( &(gpu_elem->gpu_mem), tile_size);
                    DPLASMA_CUDA_CHECK_ERROR( "cuMemAlloc ", cuda_status,
                                              ({
                                                  unsigned int free_mem, total_mem;
                                                  cuMemGetInfo( &free_mem, &total_mem );
                                                  printf("Per context: free mem %u total mem %u\n", free_mem, total_mem);
                                                  break;
                                              }) );
                    gpu_elem->memory_elem = NULL;
                    dplasma_linked_list_add_tail( gpu_device->gpu_mem_lru, (dplasma_list_item_t*)gpu_elem );
                    memory_left_for_context -= chunk_size;
                }
                printf( "Allocate %d tiles on the GPU memory\n", k );
#if DPLASMA_SMART_SCHEDULING
                /* because GPU might not be in sequence
		 * but we can get the GPU ID from gpu_array[i].gpu_id
		 * Don't forget - we're running around gpu_array[i] -> n-1
		 *  */
                gpu_device->id = gpu_array[i].gpu_id;
#else
                gpu_device->id  = i;
#endif
                gpu_device->executed_tasks = 0;
                gpu_device->transferred_data_in = 0;
                gpu_device->transferred_data_out = 0;

                status = cuCtxPopCurrent(NULL);
                DPLASMA_CUDA_CHECK_ERROR( "(INIT) cuCtxPopCurrent ", status,
                                          {free(gpu_device); return -1;} );

#if defined(DPLASMA_PROFILING)
	#if DPLASMA_SMART_SCHEDULING
		gpu_device->profiling = dplasma_profiling_thread_init( 6*4096, "GPU %d.%d", gpu_array[i].gpu_id, j);
	#else
                gpu_device->profiling = dplasma_profiling_thread_init( 6*4096, "GPU %d.%d", i, j );
	#endif
#endif  /* defined(PROFILING) */

#if DPLASMA_SMART_SCHEDULING
		/* After use context, GPUs will not know which gpu_array id they are up to */
		/* I will put lifoid for them !! */
                gpu_device->lifoid = i;
		dplasma_atomic_lifo_push( &(gpu_array[i].gpu_devices), (dplasma_list_item_t*)gpu_device);
		printf("\t\tPush context into list[%d] for GPU[%d]\n",i,gpu_array[i].gpu_id);
#else
		dplasma_atomic_lifo_push( &(gpu_devices), (dplasma_list_item_t*)gpu_device );
#endif

            }
        }

#if defined(DPLASMA_PROFILING)
        dplasma_profiling_add_dictionary_keyword( "movein", "fill:#33FF33",
                                                  &movein_key_start, &movein_key_end);
        dplasma_profiling_add_dictionary_keyword( "compute", "fill:#ff33cc",
                                                  &compute_key_start, &compute_key_end);
        dplasma_profiling_add_dictionary_keyword( "moveout", "fill:#ffff66",
                                                  &moveout_key_start, &moveout_key_end);
#endif  /* defined(PROFILING) */

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
    gpu_elem_t* gpu_elem;
    gpu_device_t* gpu_device;
    int total = 0, *gpu_counter, i, overlap_counter, active_devices = 0;
    uint64_t *transferred_in, *transferred_out, total_data_in = 0, total_data_out = 0;
    uint64_t *required_in, *required_out;
    float gtotal = 0.0, best_data_in, best_data_out;
    char *data_in_unit, *data_out_unit;

    if (ndevices <= 0)
        return 0;

    /* GPU counter for GEMM / each */
    gpu_counter = (int*)calloc(ndevices, sizeof(int));
    transferred_in  = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    transferred_out = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    required_in     = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
    required_out    = (uint64_t*)calloc(ndevices, sizeof(uint64_t));
#if DPLASMA_SMART_SCHEDULING	
    /* RUN INTO gpu_array[ 0 -> ndevices - 1] */
    for(i = 0; i < ndevices ; i++)
	    while( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&(gpu_array[i].gpu_devices))) ) {
#else
    while( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&gpu_devices)) ) {
#endif
        status = cuCtxPushCurrent( gpu_device->ctx );
        DPLASMA_CUDA_CHECK_ERROR( "(FINI) cuCtxPushCurrent ", status,
                                  {continue;} );
        status = cuCtxSynchronize();
        DPLASMA_CUDA_CHECK_ERROR( "cuCtxSynchronize", status,
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
        active_devices++;
    }
    /* No active devices */
    if( 0 == active_devices )
        return 0;

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

#if DPLASMA_SMART_SCHEDULING
    /* use new method*/
    if( NULL != (gpu_device = get_best_gpu(0)) ) {
#else
    if( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&gpu_devices)) ) {
#endif
        CUstream stream;
        cudaError_t status;
        float alpha = -1.0, beta = 1.0;

        status = cuCtxPushCurrent(gpu_device->ctx);
        DPLASMA_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                  {goto return_error;} );
        tile_size = ddescA.mb*ddescA.nb*sizeof(float);

#if defined(DPLASMA_PROFILING)
        dplasma_profiling_trace( gpu_device->profiling, movein_key_start, 0 );
#endif  /* defined(PROFILING) */
        /*cuStreamCreate(&stream, 0);*/
        on_gpu = dplasma_data_is_on_gpu(gpu_device, &ddescA, DPLASMA_READ, n, k, &gpu_elem_A);
        gpu_elem_A->memory_elem->memory = A;
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
        gpu_elem_B->memory_elem->memory = B;
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
        gpu_elem_C->memory_elem->memory = C;
        gpu_device->required_data_in += tile_size;
        if( !on_gpu ) {
            /* Push C into the GPU */
            status = cuMemcpyHtoD( d_C, C, tile_size );
            DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyHtoD to device (d_C) ", status,
                                      {printf("<<%p>>\n", (void*)(long)d_C); goto release_and_return_error;} );
            gpu_device->transferred_data_in += tile_size;
        }
#if defined(DPLASMA_PROFILING)
        dplasma_profiling_trace( gpu_device->profiling, movein_key_end, 0 );
#endif  /* defined(PROFILING) */
        /* Wait until all data are on the GPU */
        /*status = cuStreamSynchronize(stream);
          DPLASMA_CUDA_CHECK_ERROR( "cuStreamSynchronize", status,
          {goto release_and_return_error;} );*/

#if defined(DPLASMA_PROFILING)
        dplasma_profiling_trace( gpu_device->profiling, compute_key_start, 1 );
#endif  /* defined(PROFILING) */
        offset = 0;
        CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_B );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
        CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_A );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
        CU_PUSH_POINTER( gpu_device->hcuFunction, offset, d_C );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
        CU_PUSH_INT(     gpu_device->hcuFunction, offset, ddescA.nb );
        CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, alpha );
        CU_PUSH_FLOAT(   gpu_device->hcuFunction, offset, beta );
        cuParamSetSize( gpu_device->hcuFunction, offset );
CUevent start, stop;
cuEventCreate(&start, 0);
cuEventCreate(&stop, 0);
cuEventRecord(start, 0);
        // cuLaunch: we kick off the CUDA
        status = cuLaunchGrid( gpu_device->hcuFunction,
                               ddescA.nb / 64,
                               ddescA.nb / 16 );
        if ( CUDA_SUCCESS != status ) {
            printf( "cuLaunch failed %d\n", status );
            return -1;
        }
        status = cuCtxSynchronize();
        DPLASMA_CUDA_CHECK_ERROR( "cuCtxSynchronize", status,
                                  {goto release_and_return_error;} );

cuEventRecord(stop, 0);
cuEventSynchronize(stop);
float elapsedTime;
cuEventElapsedTime(&elapsedTime, start, stop);
/*printf("%d: Compute: %.6f ms\n",gpu_device->id,elapsedTime);*/


#if defined(DPLASMA_PROFILING)
        dplasma_profiling_trace( gpu_device->profiling, compute_key_end, 1 );
#endif  /* defined(PROFILING) */

        /* Pop C from the GPU */
        gpu_device->required_data_out += tile_size;
        if( (n == k+1) ) {
#if defined(DPLASMA_PROFILING)
            dplasma_profiling_trace( gpu_device->profiling, moveout_key_start, 2 );
#endif  /* defined(PROFILING) */
            /* Pop C from the GPU */
            status = cuMemcpyDtoH( C, d_C, tile_size );
            DPLASMA_CUDA_CHECK_ERROR( "cuMemcpyDtoH from device (d_C) ", status,
                                      {printf("<<%p>>\n", d_C); goto release_and_return_error;} );
            gpu_device->transferred_data_out += tile_size;
#if defined(DPLASMA_PROFILING)
            dplasma_profiling_trace( gpu_device->profiling, moveout_key_end, 2 );
#endif  /* defined(PROFILING) */
        }

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
#if DPLASMA_SMART_SCHEDULING
	/* push it back in right place */
	dplasma_atomic_lifo_push(&(gpu_array[gpu_device->lifoid].gpu_devices), (dplasma_list_item_t*)gpu_device);
	gpu_array[gpu_device->lifoid].working = 0;
#else
        dplasma_atomic_lifo_push(&gpu_devices, (dplasma_list_item_t*)gpu_device);
#endif
        /*free(tempArray);*/
        return return_code;
    }
    dplasma_atomic_inc_32b(&cpu_counter);
    return -1;  /* fails to atomically get the ownership of the device */
}
#if DPLASMA_SMART_SCHEDULING
gpu_device_t* get_best_gpu(int priority){
	/* priority made for supporting 2 many functions , default is 0 */
	int i,j,diff,still = 0;
	gpu_device_t* gpu_device;
	/* get assigned gpu before waiting for that GPU */
	int assigned,x,w;
	int a_future = 1; /* if plus [1] is future                */
	/* if plus [2] is future + being working*/
	int b_future = 1;
	/* a : this card */
	/* b : next card */
	if(priority != 0){
		/* HOW ?? */
		/* if GPU is being used less than 2 ,  let's go !!! , just my first idea :) */
		for(i = 0; i < ndevices ; i++){ /*  if(gpu_array[i].func1_current == 0 && gpu_array[i].func2_current < 2 && gpu_array[i].working == 0){ */
			if(gpu_array[i].func1_current == 0 && gpu_array[i].func2_current == 0){
				dplasma_atomic_inc_32b(&(gpu_array[i].func2_current));
				assigned = i;
				/*                  printf("choose gpu %d\n",i);*/
				goto catchseat;                                                                                                                                      }
		}
		return NULL;
	}
	for(i = 0; i < ndevices; i++){
		if(gpu_array[i].working == 1)
			a_future = 2;
		else
			a_future = 1;
		if(gpu_array[i+1].working == 1 )
			b_future = 2;
		else
			b_future = 1;
		/* comparing with next gpu (choose if this GPU will give result faster */
		if(i < ndevices - 1){
			if(gpu_array[i].func1_usage * (gpu_array[i].func1_current + a_future) < gpu_array[i+1].func1_usage * ( gpu_array[i+1].func1_current + b_future) ){
				dplasma_atomic_inc_32b(&(gpu_array[i].func1_current));
				assigned = i;
				/*  printf("choose gpu %d\n",i);*/
				goto catchseat;
			}
		}else{
			if(gpu_array[i].func1_usage * (gpu_array[i].func1_current + a_future ) < cpu_usage){
				/* if(gpu_array[i].current >= 0 ){
				 * 			 * 				 *     printf("slow gpu already have %d task\n",gpu_array[i].current);
				 * 			 			 * 				 				 *  }*/
				dplasma_atomic_inc_32b(&(gpu_array[i].func1_current));
				assigned = i;
				/*                              printf("choose gpu %d\n",i);*/
				goto catchseat;
			}
		}
	}

	/* compare every GPU with CPU !! */
	for(i = 0; i < ndevices; i++){
		if(gpu_array[i].working == 1)
			a_future = 2;
		else
			a_future = 1;
		
		if(gpu_array[i].func1_usage * (gpu_array[i].func1_current + a_future) < cpu_usage){
			dplasma_atomic_inc_32b(&(gpu_array[i].func1_current));
			assigned = i;
			/*printf("choose gpu %d ANYWAY !!\n",i);*/
			goto catchseat;
		}
	}
	
/*	printf("choose CPU !!!!\n");*/
	/* hadn't better to use GPU */
	return NULL;
catchseat:
w = 9;

if(dplasma_atomic_cas(&(gpu_array[assigned].waiting[w]),0,1) == 1){
	while(w > 0){
			if(dplasma_atomic_cas(&(gpu_array[assigned].waiting[w-1]),0,1) == 1){
				if( w-1 > 0){
					gpu_array[assigned].waiting[w] = 0;
					w--;
				}else if(w-1 == 0){
					gpu_array[assigned].waiting[w] = 0;
					w--;
				}
				if(w == 0){
					goto catchgpu;
				}
			}else{
				continue;
			}
		}
	}
catchgpu:
	for(;;){
		if( NULL != (gpu_device = (gpu_device_t*)dplasma_atomic_lifo_pop(&(gpu_array[assigned].gpu_devices)))){
			/* unleash seat -- concurreny !! to sgemm function now !*/
			gpu_array[assigned].waiting[w] = 0;
			if(priority == 0)
				dplasma_atomic_dec_32b(&(gpu_array[assigned].func1_current));
			else
				dplasma_atomic_dec_32b(&(gpu_array[assigned].func2_current));
			gpu_array[assigned].working = 1;
			return gpu_device;
		}
	}
	
}
#endif

