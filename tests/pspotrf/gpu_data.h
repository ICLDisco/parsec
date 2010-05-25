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

#include <cuda.h>
#include <cuda_runtime_api.h>

typedef struct _gpu_device {
    dplasma_list_item_t item;
    CUcontext ctx;
    CUmodule hcuModule;
    CUfunction hcuFunction;
    int id;
    int executed_tasks;
    uint64_t transferred_data_in;
    uint64_t transferred_data_out;
    dplasma_linked_list_t* gpu_mem_lru;
} gpu_device_t;

typedef struct _gpu_elem {
    dplasma_list_item_t item;
    int lock;
    CUdeviceptr gpu_mem;
    int col;
    int row;
    void* memory;
    int gpu_version;
    int memory_version;
    int readers;
    int writer;
} gpu_elem_t;

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

#endif  /* DPLASMA_GPU_DATA_H_HAS_BEEN_INCLUDED */
