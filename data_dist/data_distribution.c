#include "include/dague_config.h"
#include "data_dist/data_distribution.h"

static int using_gpu = 0;

#if defined(DAGUE_CUDA_SUPPORT)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/lifo.h"
#include "include/gpu_data.h"
#endif  /* defined(DAGUE_CUDA_SUPPORT) */

void* dague_allocate_data(size_t matrix_size)
{
    void* mat = NULL;
#if defined(DAGUE_CUDA_SUPPORT)
    if( using_gpu ) {
        CUresult status;
        gpu_device_t* gpu_device;

        gpu_device = (gpu_device_t*)dague_atomic_lifo_pop(&gpu_devices);
        if( NULL != gpu_device ) {
            status = cuCtxPushCurrent( gpu_device->ctx );
            DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuCtxPushCurrent ", status,
                                    { 
                                        fprintf(stderr, 
                                                "Unable to allocate GPU-compatible data as requested.\n"
                                                "  You might want to check that the number of preallocated buffers is\n"
                                                "  not too large (not enough memory to satisfy request)\n"
                                                "  and not too small (cannot allocate memory during execution of code in the GPU)\n");
                                        assert(0);
                                        exit(2);
                                    } );

            status = cuMemHostAlloc( (void**)&mat, matrix_size, CU_MEMHOSTALLOC_PORTABLE);
            if( CUDA_SUCCESS != status ) {
                DAGUE_CUDA_CHECK_ERROR( "(dague_allocate_matrix) cuMemHostAlloc failed ", status,
                                          { 
                                              fprintf(stderr, 
                                                      "Unable to allocate GPU-compatible data as requested.\n"
                                                      "  You might want to check that the number of preallocated buffers is\n"
                                                      "  not too large (not enough memory to satisfy request)\n"
                                                      "  and not too small (cannot allocate memory during execution of code in the GPU)\n");
                                              assert(0);
                                              exit(2);
                                          } );
                mat = NULL;
            }
            status = cuCtxPopCurrent(NULL);
            DAGUE_CUDA_CHECK_ERROR( "cuCtxPushCurrent ", status,
                                      {} );
            dague_atomic_lifo_push(&gpu_devices, (dague_list_item_t*)gpu_device);
        }
    }
#endif  /* defined(DAGUE_CUDA_SUPPORT) */
    /* If nothing else worked so far, allocate the memory using malloc */
    if( NULL == mat ) {
        mat = malloc( matrix_size );
    }

    if( NULL == mat ) {
        printf("memory allocation of %lu\n", (unsigned long) matrix_size);
        perror("matrix allocation failed");
        return NULL;
    }
    return mat;
}

/**
 * Enable GPU-compatible memory if possible
 */
void dague_data_enable_gpu( void )
{
#if defined(DAGUE_CUDA_SUPPORT)
    using_gpu = 1;
#else
    fprintf(stderr, "Requesting GPU-enabled memory, although no CUDA support\n");
#endif
}

/**
 * free a buffer allocated by dague_allocate_data
 */
void dague_free_data(void *dta)
{
#if defined(DAGUE_CUDA_SUPPORT)
    if( using_gpu )
        cuMemFreeHost( dta );
    else
#endif
        free( dta );
}
