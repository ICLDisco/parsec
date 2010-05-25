/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "gpu_data.h"
#include "data_management.h"
#include "linked_list.h"

gpu_elem_t** data_map = NULL;

int dplasma_mark_data_usage( DPLASMA_desc* data, int type, int col, int row )
{
    gpu_elem_t* this_data;

    if( (NULL == data_map) || (NULL == (this_data = data_map[col * data->lnt + row])) ) {
        /* Data not on the GPU. Nothing to do */
        return 0;
    }
    if( type & DPLASMA_WRITE ) {
        this_data->memory_version++;
        this_data->writer++;
    }
    if( type & DPLASMA_READ ) {
        this_data->readers++;
    }
    return 0;
}

int dplasma_data_map_init( gpu_device_t* gpu_device,
                           DPLASMA_desc* data )
{
    if( NULL == data_map ) {
        data_map = (gpu_elem_t**)calloc(data->lmt * data->lnt, sizeof(gpu_elem_t*));
    }
    gpu_device->gpu_mem_lru = (dplasma_linked_list_t*)malloc(sizeof(dplasma_linked_list_t));
    dplasma_linked_list_construct(gpu_device->gpu_mem_lru);
    return 0;
}

/**
 * This function check if the target tile is already on the GPU memory. If it is the case,
 * it check if the version on the GPU match with the one in memory. In all cases, it
 * propose a section in the GPU memory where the data should be transferred.
 *
 * It return 1 if no transfer should be initiated, a 0 if a transfer is
 * necessary, and a negative value if no memory is currently available on the GPU.
 */
int dplasma_data_is_on_gpu( gpu_device_t* gpu_device,
                            DPLASMA_desc* data,
                            int type, int col, int row,
                            gpu_elem_t **gpu_elem)
{
    gpu_elem_t* this_data;

    if( NULL == (this_data = data_map[col * data->lnt + row]) ) {
        this_data = (gpu_elem_t*)dplasma_linked_list_remove_head(gpu_device->gpu_mem_lru);
        if( NULL != this_data->memory ) {  /* remove the refs to the old location */
            data_map[this_data->col * data->lnt + this_data->row] = NULL;
        }
        this_data->col = col;
        this_data->row = row;
        this_data->gpu_version = 0;
        this_data->memory_version = 0;
        this_data->readers = 0;
        this_data->writer = 0;
        this_data->memory = dplasma_get_local_tile_s(data, col, row);
        data_map[col * data->lnt + row] = this_data;
        /* Get the LRU element on the GPU and transfer it to this new data */
        *gpu_elem = this_data;
        dplasma_linked_list_add_tail(gpu_device->gpu_mem_lru, (dplasma_list_item_t*)this_data);
    } else {
        dplasma_linked_list_remove_item(gpu_device->gpu_mem_lru, (dplasma_list_item_t*)this_data);
        dplasma_linked_list_add_tail(gpu_device->gpu_mem_lru, (dplasma_list_item_t*)this_data);
        *gpu_elem = this_data;
        if( this_data->memory_version == this_data->gpu_version ) {
            /* The GPU version of the data matches the one in memory. We're done */
            return 1;
        }
        if( -1 == this_data->gpu_version ) {
            /* No mapping to GPU memory. We have to allocate one */
            goto allocate_gpu_memory;
        }
        /* The version on the GPU doesn't match the one in memory. Let the
         * upper level know a transfer is required.
         */
        return 0;
    }
 allocate_gpu_memory:
    /* No memory on the GPU. Get the least recently used tile on the GPU and
     * attach it.
     */
    return 0;
}

