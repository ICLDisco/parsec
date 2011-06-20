/*
 * Copyright (c) 2010      The University of Tennessee and The University
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

int gpu_sgemm( dague_execution_unit_t* eu_context,
               dague_execution_context_t* exec_context,
               int uplo );

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

#include "data_distribution.h"

typedef struct _memory_elem memory_elem_t;
typedef struct _gpu_elem gpu_elem_t;

struct _gpu_elem {
    dague_list_item_t item;
    int lock;
    int type;
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
    DAGUE_READ,
    DAGUE_WRITE
} dague_data_usage_type_t;

#include "data_dist/matrix/matrix.h"

int gpu_mark_data_usage( tiled_matrix_desc_t* data, int type, int col, int row );

int sgemm_cuda_init( dague_context_t* context, tiled_matrix_desc_t *tileA );
int sgemm_cuda_fini( dague_context_t* dague_context );

int gpu_data_map_init( gpu_device_t* gpu_device,
                       tiled_matrix_desc_t* data );
int gpu_data_tile_write_owner( tiled_matrix_desc_t* data,
                               int col, int row );
int gpu_data_get_tile( tiled_matrix_desc_t* data,
                       int col, int row,
                       memory_elem_t **pmem_elem );
int gpu_data_is_on_gpu( gpu_device_t* gpu_device,
                        tiled_matrix_desc_t* data,
                        int type, int col, int row,
                        gpu_elem_t **pgpu_elem);

#endif
