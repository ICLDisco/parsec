/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _gpu_gemm_h
#define _gpu_gemm_h

#include "dague_config.h"
#include "gpu_data.h"
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "list.h"
#include "fifo.h"

#define GEMM_KEY(M, N) (uint32_t)((NULL == gpu_data.tiled_matrix) ? \
                                  0 : (M) * gpu_data.tiled_matrix->lmt + (N))

int gpu_sgemm( dague_execution_unit_t* eu_context,
               dague_execution_context_t* this_task,
               int uplo );

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

#include "data_dist/matrix/matrix.h"

/**
 * Data coherency protocol based on MOESI.
 */
#define    DAGUE_DATA_INVALID    ((uint8_t)0x0)
#define    DAGUE_DATA_OWNED      ((uint8_t)0x1)
#define    DAGUE_DATA_EXCLUSIVE  ((uint8_t)0x2)
#define    DAGUE_DATA_SHARED     ((uint8_t)0x4)

typedef uint8_t                    dague_data_coherency_t;
typedef struct _dague_device_elem  dague_device_elem_t;
typedef struct _memory_elem        memory_elem_t;
typedef struct _gpu_elem           gpu_elem_t;

/**
 * Generic type for all the devices.
 */
struct _dague_device_elem {
    dague_list_item_t      item;
    dague_data_coherency_t coherency_state;
    uint16_t               readers;
    uint32_t               version;
    memory_elem_t*         memory_elem;
};

/**
 * A memory element targets a specific data. It can be found
 * based on a unique key.
 */
struct _memory_elem {
    uint32_t               key;
    dague_data_coherency_t coherency_state;
    uint16_t               device_owner;
    uint32_t               version;
    void*                  main_memory;
    dague_device_elem_t*   device_elem[1];
};

typedef struct __dague_gpu_data_map {
    tiled_matrix_desc_t*  tiled_matrix;
    memory_elem_t** data_map;
} dague_gpu_data_map_t;

/**
 * Particular overloading of the generic device type
 * for GPUs.
 */
struct _gpu_elem {
    dague_device_elem_t    generic;
    CUdeviceptr            gpu_mem;
};


typedef enum {
    DAGUE_READ       = ACCESS_READ,
    DAGUE_WRITE      = ACCESS_WRITE,
    DAGUE_READ_DONE  = 0x4,
    DAGUE_WRITE_DONE = 0x8
} dague_data_usage_type_t;

int sgemm_cuda_init( dague_context_t* context, tiled_matrix_desc_t *tileA );
int sgemm_cuda_fini( dague_context_t* dague_context );

int sgemm_cuda_ndevices(void);

int gpu_data_map_init( gpu_device_t* gpu_device,
                       tiled_matrix_desc_t* data,
                       dague_gpu_data_map_t* gpu_map);
int gpu_data_map_fini( dague_gpu_data_map_t* gpu_map );

int gpu_data_tile_write_owner( dague_gpu_data_map_t* gpu_map,
                               uint32_t key );
int gpu_data_get_tile( dague_gpu_data_map_t* gpu_map,
                       uint32_t key,
                       memory_elem_t **pmem_elem );

int dague_update_data_version( dague_gpu_data_map_t* gpu_map, uint32_t key );

#endif
