#ifndef _gpu_stsmqr_h
#define _gpu_stsmqr_h

#include "dague_config.h"
#include "gpu_data.h"
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"

int gpu_stsmqr( dague_execution_unit_t* eu_context,
                dague_execution_context_t* exec_context );

/****************************************************
 ** GPU-DATA that is QR Specific Starts Here **
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

int gpu_qr_mark_data_usage( int matrixIsT, tiled_matrix_desc_t* data, int type, int col, int row );

int stsmqr_cuda_init( dague_context_t* context,
                      tiled_matrix_desc_t *tileA,
                      tiled_matrix_desc_t *tileT );
int stsmqr_cuda_fini( dague_context_t* context );

int gpu_qr_data_map_init( int matrixIsT, 
                          gpu_device_t* gpu_device,
                          tiled_matrix_desc_t* data );
int gpu_qr_data_tile_write_owner( int matrixIsT, 
                                  tiled_matrix_desc_t* data,
                                  int col, int row );
int gpu_qr_data_get_tile( int matrixIsT, 
                          tiled_matrix_desc_t* data,
                          int col, int row,
                          memory_elem_t **pmem_elem );
int gpu_qr_data_is_on_gpu( int matrixIsT, 
                           gpu_device_t* gpu_device,
                           tiled_matrix_desc_t* data,
                           int type, int col, int row,
                           gpu_elem_t **pgpu_elem);

#endif
