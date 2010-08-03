/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DPLASMA_GPU_DATA_H_HAS_BEEN_INCLUDED
#define DPLASMA_GPU_DATA_H_HAS_BEEN_INCLUDED

#include "dplasma_config.h"
#include "data_management.h"
#include "linked_list.h"
#include "profiling.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#define DPLASMA_SMART_SCHEDULING 0

typedef struct _gpu_device {
    dplasma_list_item_t item;
    CUcontext ctx;
    CUmodule hcuModule;
    CUfunction hcuFunction;
    int id;
    int executed_tasks;
    int major;
    int minor;
#if DPLASMA_SMART_SCHEDULING
    int lifoid;
#endif
    uint64_t transferred_data_in;
    uint64_t transferred_data_out;
    uint64_t required_data_in;
    uint64_t required_data_out;
    dplasma_linked_list_t* gpu_mem_lru;
#if defined(DPLASMA_PROFILING)
    dplasma_thread_profiling_t *profiling;
#endif  /* defined(PROFILING) */
} gpu_device_t;

#if DPLASMA_SMART_SCHEDULING
typedef struct _gpu_item {
	int gpu_id;
	int func1_usage;
	int func1_current;
	int func2_usage;
	int func2_current;
	int working;
	volatile int32_t *waiting;
	dplasma_atomic_lifo_t gpu_devices;
} gpu_item_t;
gpu_item_t* gpu_array;
#endif

typedef struct _memory_elem memory_elem_t;
typedef struct _gpu_elem gpu_elem_t;

struct _gpu_elem {
    dplasma_list_item_t item;
    int lock;
    CUdeviceptr gpu_mem;
    memory_elem_t* memory_elem;
    int gpu_version;
};
 	
struct _memory_elem {
    int memory_version;
    int readers;
    int writer;
    int row;
    int col;
    void* memory;
    gpu_elem_t* gpu_elems[1];
};

typedef enum {
    DPLASMA_READ,
    DPLASMA_WRITE
} dplasma_data_usage_type_t;

extern int dplasma_mark_data_usage( DPLASMA_desc* data, int type, int col, int row );
extern int dplasma_data_is_on_gpu( gpu_device_t* gpu_device,
                                   DPLASMA_desc* data,
                                   int type, int col, int row,
                                   gpu_elem_t **gpu_elem);
extern int dplasma_data_map_init( gpu_device_t* gpu_device,
                                  DPLASMA_desc* data );

#define DPLASMA_CUDA_CHECK_ERROR( STR, ERROR, CODE )                    \
    {                                                                   \
        cudaError_t cuda_error = (ERROR);                               \
        if( cudaSuccess != cuda_error ) {                               \
            printf( "%s:%d %s%s\n", __FILE__, __LINE__,                 \
                    (STR), cudaGetErrorString(cuda_error) );            \
            CODE;                                                       \
        }                                                               \
    }

#endif  /* DPLASMA_GPU_DATA_H_HAS_BEEN_INCLUDED */
