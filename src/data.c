/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "data.h"
#include "lifo.h"
#include "gpu_data.h"

/* TODO: create a consistent common infrastructure for the devices */
uint32_t dague_supported_number_of_devices = 1;
static dague_lifo_t dague_data_lifo;
static dague_lifo_t dague_data_copies_lifo;

int dague_data_init(void)
{
    dague_lifo_construct(&dague_data_lifo);
    dague_lifo_construct(&dague_data_copies_lifo);
    dague_supported_number_of_devices = 1 + dague_active_gpu();
    return 0;
}

int dague_data_fini()
{
    dague_lifo_destruct(&dague_data_lifo);
    dague_lifo_destruct(&dague_data_copies_lifo);
    return 0;
}

/**
 *
 */
dague_data_t* dague_data_new(void)
{
    dague_data_t* item = (dague_data_t*)dague_lifo_pop(&dague_data_lifo);
    if( NULL == item ) {
        item = (dague_data_t*)malloc(sizeof(dague_data_t) +
                                     sizeof(dague_data_copy_t) * dague_supported_number_of_devices);
        if( NULL == item ) return NULL;
    }
    item->version          = 0;
    item->coherency_state  = DATA_COHERENCY_INVALID;
    item->owner_device     = -1;
    item->key              = 0;
    item->nb_elts          = 0;
    { uint32_t i;
        for( i = 0; i < dague_supported_number_of_devices; item->device_copies[i++] = NULL );
    }
    return item;
}

/**
 *
 */
void dague_data_delete(dague_data_t* data)
{
    DAGUE_LIST_ITEM_SINGLETON((dague_list_item_t*)data);
    dague_lifo_push(&dague_data_lifo, (dague_list_item_t*)data);
}

/**
 *
 */
dague_data_copy_t* dague_data_copy_new(dague_data_t* data, uint16_t device)
{
    dague_data_copy_t* item;
    dague_data_t* master = NULL;

    if( NULL == data ) {
        master = dague_data_new();
        if( NULL == master ) return NULL;
        data = master;
    }
    item = (dague_data_copy_t*)dague_lifo_pop(&dague_data_copies_lifo);
    if( NULL == item ) {
        item = (dague_data_copy_t*)malloc(sizeof(dague_data_copy_t));
        if( NULL == item ) {
            free(master);  /* released the data if allocated here */
            return NULL;
        }
    }
    item->refcount        = 1;
    item->device_index    = device;
    item->flags           = 0;
    item->coherency_state = DATA_COHERENCY_INVALID;
    item->readers         = 1;
    item->version         = data->version;
    item->older           = NULL;
    item->original        = data;
    item->device_private  = NULL;
    /* Atomically set the device copy */
    if( !dague_atomic_cas(&data->device_copies[device], NULL, item) ) {
        free(item);  /* another thread succeeded before us */
        item = data->device_copies[device];
    }
    return item;
}

