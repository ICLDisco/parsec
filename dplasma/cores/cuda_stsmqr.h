/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _gpu_stsmqr_h
#define _gpu_stsmqr_h

#include "dague_config.h"
#include "gpu_data.h"
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
/**
 * Import the generics for the GPU handling.
 */
#include "cuda_sgemm.h"
int gpu_stsmqr( dague_execution_unit_t* eu_context,
                dague_execution_context_t* this_task );

/****************************************************
 ** GPU-DATA that is QR Specific Starts Here **
 ****************************************************/

int gpu_qr_mark_data_usage( int matrixIsT, const tiled_matrix_desc_t* data, int type, int col, int row );

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
