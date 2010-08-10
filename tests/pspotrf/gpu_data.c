/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "gpu_data.h"
#include "data_management.h"
#include "linked_list.h"

static memory_elem_t** data_map = NULL;
extern int ndevices;

int dplasma_mark_data_usage( DPLASMA_desc* data, int type, int col, int row )
{
    memory_elem_t* this_data;

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
        data_map = (memory_elem_t**)calloc(data->lmt * data->lnt, sizeof(memory_elem_t*));
    }
    gpu_device->gpu_mem_lru = (dplasma_linked_list_t*)malloc(sizeof(dplasma_linked_list_t));
    dplasma_linked_list_construct(gpu_device->gpu_mem_lru);
    return 0;
}

int dplasma_data_tile_write_owner( DPLASMA_desc* data,
                                   int col, int row )
{
    memory_elem_t* memory_elem;
    gpu_elem_t* gpu_elem;
    int i;

    if( NULL == (memory_elem = data_map[col * data->lnt + row]) ) {
        return -1;
    }
    for( i = 0; i < ndevices; i++ ) {
        gpu_elem = memory_elem->gpu_elems[i];
        if( NULL == gpu_elem )
            continue;
        if( gpu_elem->type & DPLASMA_WRITE )
            return i;
    }
    return -2;
}

int dplasma_data_get_tile( DPLASMA_desc* data,
                           int col, int row,
                           memory_elem_t **pmem_elem )
{
    memory_elem_t* memory_elem;
    int rc = 0;  /* the tile already existed */

    if( NULL == (memory_elem = data_map[col * data->lnt + row]) ) {
        memory_elem = (memory_elem_t*)calloc(1, sizeof(memory_elem_t) + (ndevices-1) * sizeof(gpu_elem_t*));
        memory_elem->col = col;
        memory_elem->row = row;
        memory_elem->memory_version = 0;
        memory_elem->readers = 0;
        memory_elem->writer = 0;
        memory_elem->memory = NULL;
        rc = 1;  /* the tile has just been created */
        if( 0 == dplasma_atomic_cas( &(data_map[col * data->lnt + row]), NULL, memory_elem ) ) {
            free(memory_elem);
            rc = 0;  /* the tile already existed */
            memory_elem = data_map[col * data->lnt + row];
        }
    }
    *pmem_elem = memory_elem;
    return rc;
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
                            gpu_elem_t **pgpu_elem)
{
    memory_elem_t* memory_elem;
    gpu_elem_t* gpu_elem;

    dplasma_data_get_tile( data, col, row, &memory_elem );

    if( NULL == (gpu_elem = memory_elem->gpu_elems[gpu_device->id]) ) {
        /* Get the LRU element on the GPU and transfer it to this new data */
        gpu_elem = (gpu_elem_t*)dplasma_linked_list_remove_head(gpu_device->gpu_mem_lru);
        if( memory_elem != gpu_elem->memory_elem ) {
            if( NULL != gpu_elem->memory_elem ) {
                memory_elem_t* old_mem = gpu_elem->memory_elem;
                old_mem->gpu_elems[gpu_device->id] = NULL;
            }
            gpu_elem->type = 0;
        }
        gpu_elem->type |= type;
        gpu_elem->memory_elem = memory_elem;
        memory_elem->gpu_elems[gpu_device->id] = gpu_elem;
        *pgpu_elem = gpu_elem;
        dplasma_linked_list_add_tail(gpu_device->gpu_mem_lru, (dplasma_list_item_t*)gpu_elem);
    } else {
        dplasma_linked_list_remove_item(gpu_device->gpu_mem_lru, (dplasma_list_item_t*)gpu_elem);
        dplasma_linked_list_add_tail(gpu_device->gpu_mem_lru, (dplasma_list_item_t*)gpu_elem);
        gpu_elem->type |= type;
        *pgpu_elem = gpu_elem;
        if( memory_elem->memory_version == gpu_elem->gpu_version ) {
            /* The GPU version of the data matches the one in memory. We're done */
            return 1;
        }
        /* The version on the GPU doesn't match the one in memory. Let the
         * upper level know a transfer is required.
         */
    }
    gpu_elem->gpu_version = memory_elem->memory_version;
    /* Transfer is required */
    return 0;
}

