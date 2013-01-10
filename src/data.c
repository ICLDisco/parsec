/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "data.h"
#include "lifo.h"
#include "gpu_data.h"
#include <dague/constants.h>
#include "data.h"

/* TODO: create a consistent common infrastructure for the devices */
uint32_t dague_supported_number_of_devices = 1;
static dague_lifo_t dague_data_lifo;
static dague_lifo_t dague_data_copies_lifo;

static void dague_data_copy_construct(dague_data_copy_t* obj)
{
    obj->device_index     = 0;
    obj->flags            = 0;
    obj->coherency_state  = DATA_COHERENCY_INVALID;
    obj->readers          = 0;
    obj->version          = 0;
    obj->older            = NULL;
    obj->original         = NULL;
    obj->device_private   = NULL;

}

#if defined(DAGUE_DEBUG_ENABLE)
static void dague_data_copy_destruct(dague_data_copy_t* obj)
{
    assert(NULL == obj->original);  /* make sure we are not attached to a data */
}
#endif  /* defined(DAGUE_DEBUG_ENABLE) */

OBJ_CLASS_INSTANCE(dague_data_copy_t, dague_list_item_t,
                   dague_data_copy_construct,
#if defined(DAGUE_DEBUG_ENABLE)
                   dague_data_copy_destruct
#else
                   NULL
#endif  /* defined(DAGUE_DEBUG_ENABLE) */
                   );

static void dague_data_construct(dague_data_t* obj )
{
    obj->version          = 0;
    obj->coherency_state  = DATA_COHERENCY_INVALID;
    obj->owner_device     = -1;
    obj->key              = 0;
    obj->nb_elts          = 0;
    for( uint32_t i = 0; i < dague_supported_number_of_devices;
         obj->device_copies[i] = NULL, i++ );
}

static void dague_data_destruct(dague_data_t* obj )
{
    for( uint32_t i = 0; i < dague_supported_number_of_devices; i++ )
        assert(NULL == obj->device_copies[i]);
}

OBJ_CLASS_INSTANCE(dague_data_t, dague_list_item_t,
                   dague_data_construct,
#if defined(DAGUE_DEBUG_ENABLE)
                   dague_data_destruct
#else
                   NULL
#endif  /* defined(DAGUE_DEBUG_ENABLE) */
                   );

int dague_data_init(void)
{
    OBJ_CONSTRUCT(&dague_data_lifo, dague_lifo_t);
    OBJ_CONSTRUCT(&dague_data_copies_lifo, dague_lifo_t);
    dague_supported_number_of_devices = 1 + dague_active_gpu();
    /**
     * This is a trick. Now that we know the number of available devices
     * we can update the size of the dague_data_t class to the correct value.
     */
    dague_data_t_class.cls_sizeof += sizeof(dague_data_copy_t*) * dague_supported_number_of_devices;
    return 0;
}

int dague_data_fini()
{
    OBJ_DESTRUCT(&dague_data_lifo);
    OBJ_DESTRUCT(&dague_data_copies_lifo);
    return 0;
}

/**
 *
 */
dague_data_t* dague_data_new(void)
{
    dague_data_t* item = (dague_data_t*)dague_lifo_pop(&dague_data_lifo);
    if( NULL == item ) {
        item = OBJ_NEW(dague_data_t);
        if( NULL == item ) return NULL;
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

inline int
dague_data_copy_attach(dague_data_t* data,
                       dague_data_copy_t* copy,
                       uint16_t device)
{
    copy->device_index    = device;
    copy->original        = data;
    /* Atomically set the device copy */
    if( !dague_atomic_cas(&data->device_copies[device], NULL, copy) ) {
        return DAGUE_ERROR;
    }
    return DAGUE_SUCCESS;
}

int dague_data_copy_detach(dague_data_t* data,
                           dague_data_copy_t* copy,
                           uint16_t device)
{
    dague_data_copy_t* obj = data->device_copies[device];

    if( obj != copy )
        return DAGUE_ERR_NOT_FOUND;
    /* Atomically set the device copy */
    if( !dague_atomic_cas(&data->device_copies[device], copy, NULL) ) {
        return DAGUE_ERROR;
    }
    copy->device_index    = 0;
    copy->original        = NULL;
    return DAGUE_SUCCESS;
}

/**
 *
 */
dague_data_copy_t* dague_data_copy_new(dague_data_t* data, uint16_t device)
{
    dague_data_copy_t* copy;

    copy = (dague_data_copy_t*)dague_lifo_pop(&dague_data_copies_lifo);
    if( NULL == copy ) {
        copy = OBJ_NEW(dague_data_copy_t);
        if( NULL == copy ) {
            return NULL;
        }
    }
    OBJ_CONSTRUCT(&copy->super, dague_list_item_t);
    dague_data_copy_construct(copy);
    if( DAGUE_SUCCESS != dague_data_copy_attach(data, copy, device) ) {
        OBJ_RELEASE(copy);
        return NULL;
    }
    return copy;
}

int dague_data_copy_ownership_to_device(dague_data_t* data,
                                        uint16_t device,
                                        uint8_t access_mode)
{
    dague_data_copy_t* copy;
    int i, transfer_required = 0;

    copy = data->device_copies[device];
    assert( NULL != copy );

    if( ACCESS_READ & access_mode ) copy->readers++;

    if( DATA_COHERENCY_INVALID == copy->coherency_state ) {
        if( ACCESS_READ & access_mode ) transfer_required = -1;
        /* Update the coherency state of the others versions */
        if( ACCESS_WRITE & access_mode ) {
            //assert( DATA_COHERENCY_OWNED != data->coherency_state ); /* 2 writters on the same data: wrong JDF */
            data->coherency_state = DATA_COHERENCY_OWNED;
            data->owner_device = (uint16_t)device;
            for( i = 0; i < dague_supported_number_of_devices; i++ ) {
                if( NULL == data->device_copies[i] ) continue;
                data->device_copies[i]->coherency_state = DATA_COHERENCY_INVALID;
            }
            copy->coherency_state = DATA_COHERENCY_OWNED;
            copy->version = data->version;
        }
        else if( ACCESS_READ & access_mode ) {
            if( DATA_COHERENCY_OWNED == data->coherency_state ) {
                transfer_required = 1; /* TODO: is this condition making sense? */
            }
            data->coherency_state = DATA_COHERENCY_SHARED;
            copy->coherency_state = DATA_COHERENCY_SHARED;
        }
    }
    else { /* !DATA_COHERENCY_INVALID */
        if( DATA_COHERENCY_OWNED == copy->coherency_state ) {
            assert( device == data->owner_device ); /* memory is owned, better be me otherwise 2 writters: wrong JDF */
        }
        else {
            if( ACCESS_WRITE & access_mode ) {
                copy->coherency_state = DATA_COHERENCY_OWNED;
                data->owner_device = (uint16_t)device;
                /* Update the coherency state of the others versions */
            } else {
                /* The data is shared or exclusive and I'm doing a read */
            }
        }
    }

    assert( data->version >= copy->version );
    /* The version on the GPU doesn't match the one in memory. Let the
     * upper level know a transfer is required.
     */
    transfer_required = transfer_required || (data->version > copy->version);
    if( transfer_required )
        copy->version = data->version;
    return transfer_required;
}

static char dump_coherency_codex(dague_data_coherency_t state)
{
    if( DATA_COHERENCY_INVALID   == state ) return 'I';
    if( DATA_COHERENCY_OWNED     == state ) return 'O';
    if( DATA_COHERENCY_EXCLUSIVE == state ) return 'E';
    if( DATA_COHERENCY_SHARED    == state ) return 'S';
    return 'X';
}

void dague_dump_data_copy(dague_data_copy_t* copy)
{
    dague_data_t* data = copy->original;

    printf("data %p key %x owner %d state %c version %d\n"
           "-  %d: copy %p state %c readers %d version %d\n",
           data, data->key, data->owner_device, dump_coherency_codex(data->coherency_state), data->version,
           (int)copy->device_index, copy, dump_coherency_codex(copy->coherency_state), copy->readers, copy->version);
}
