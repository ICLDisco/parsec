/*
 * Copyright (c) 2012-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "data.h"
#include "lifo.h"
#include <dague/constants.h>
#include <dague/devices/device.h>
#include <dague/utils/output.h>
#include "data.h"
#include "arena.h"

/* TODO: create a consistent common infrastructure for the devices */
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
    obj->arena_chunk      = NULL;
    obj->data_transfer_status      = DATA_STATUS_NOT_TRANSFER;
    obj->push_task        = NULL;
}

static void dague_data_copy_destruct(dague_data_copy_t* obj)
{
    if( obj->flags & DAGUE_DATA_FLAG_ARENA ) {
        /* It is an arena that is now unused.
         * give the chunk back to the arena memory management.
         * This detaches obj from obj->original, and frees everything */
        dague_arena_release(obj);
    }
  //  assert(NULL == obj->original);  /* make sure we are not attached to a data */
}

OBJ_CLASS_INSTANCE(dague_data_copy_t, dague_list_item_t,
                   dague_data_copy_construct,
                   dague_data_copy_destruct);

static void dague_data_construct(dague_data_t* obj )
{
    obj->owner_device     = -1;
    obj->key              = 0;
    obj->nb_elts          = 0;
    for( uint32_t i = 0; i < dague_nb_devices;
         obj->device_copies[i] = NULL, i++ );
}

static void dague_data_destruct(dague_data_t* obj )
{
    for( uint32_t i = 0; i < dague_nb_devices; i++ ) {
        dague_data_copy_t *copy = NULL;

        while( (copy = obj->device_copies[i]) != NULL )
        {
            dague_data_copy_detach( obj, copy, i );
            OBJ_RELEASE( copy );
        }
        assert(NULL == obj->device_copies[i]);
    }
}

OBJ_CLASS_INSTANCE(dague_data_t, dague_object_t,
                   dague_data_construct,
                   dague_data_destruct
                   );

int dague_data_init(dague_context_t* context)
{
    OBJ_CONSTRUCT(&dague_data_lifo, dague_lifo_t);
    OBJ_CONSTRUCT(&dague_data_copies_lifo, dague_lifo_t);
    /**
     * This is a trick. Now that we know the number of available devices
     * we can update the size of the dague_data_t class to the correct value.
     */
    if( !dague_devices_freezed(context) ) {
        dague_output(0, "Canot configure the data infrastructure as the devices layer has not yet been freezed\n");
        return DAGUE_ERROR;
    }
    dague_data_t_class.cls_sizeof += sizeof(dague_data_copy_t*) * dague_nb_devices;
    return 0;
}

int dague_data_fini(dague_context_t* context)
{
    OBJ_DESTRUCT(&dague_data_lifo);
    OBJ_DESTRUCT(&dague_data_copies_lifo);
    (void)context;
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
    } else {
        OBJ_CONSTRUCT(item, dague_object_t);
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
                       uint8_t device)
{
    copy->device_index    = device;
    copy->original        = data;
    /* Atomically set the device copy */
    copy->older = data->device_copies[device];
    if( !dague_atomic_cas(&data->device_copies[device], copy->older, copy) ) {
        copy->older = NULL;
        return DAGUE_ERROR;
    }
    return DAGUE_SUCCESS;
}

int dague_data_copy_detach(dague_data_t* data,
                           dague_data_copy_t* copy,
                           uint8_t device)
{
    dague_data_copy_t* obj = data->device_copies[device];

    if( obj != copy )
        return DAGUE_ERR_NOT_FOUND;
    /* Atomically set the device copy */
    if( !dague_atomic_cas(&data->device_copies[device], copy, copy->older) ) {
        return DAGUE_ERROR;
    }
    copy->device_index    = 0;
    copy->original        = NULL;
    return DAGUE_SUCCESS;
}

/**
 *
 */
dague_data_copy_t* dague_data_copy_new(dague_data_t* data, uint8_t device)
{
    dague_data_copy_t* copy;

    copy = (dague_data_copy_t*)dague_lifo_pop(&dague_data_copies_lifo);
    if( NULL == copy ) {
        copy = OBJ_NEW(dague_data_copy_t);
        if( NULL == copy ) {
            return NULL;
        }
    } else {
        OBJ_CONSTRUCT(copy, dague_data_copy_t);
    }
    if( DAGUE_SUCCESS != dague_data_copy_attach(data, copy, device) ) {
        OBJ_RELEASE(copy);
        return NULL;
    }
    return copy;
}

/**
 * Find the corresponding copy of the data on the requested device. If the
 * copy is not available in the access mode requested, a new version will
 * be created. If no tranfer is required the correct version will be set
 * into dest and the function returns 0. If a transfer is necessary, dest
 * will contain the pointer to the copy to be tranferred to, and the function
 * will return 1. All other cases should be considered as errors, and a
 * negative value must be returned (corresponding to a specific DAGUE_
 * error code.
 */
int dague_data_get_device_copy(dague_data_copy_t* source,
                               dague_data_copy_t** dest,
                               uint8_t device,
                               uint8_t access_mode)
{
    dague_data_copy_t* copy, *newer;
    dague_data_t* original;
    int transfer = 0;

    if( device == source->device_index ) {
        *dest = source;
        return 0;
    }
    original = source->original;
    /* lock the original data */
    newer = copy = original->device_copies[device];
    while( NULL != copy ) {
        if( source->version == copy->version )
            break;
        copy = copy->older;
    }
    if( NULL == copy ) {
        *dest = copy = dague_data_copy_new(original, device);
        transfer = 1;
    } else if( source->version == copy->version ) {
        *dest = copy;
    }
    /* unlock the original data */

    return transfer;
}

/**
 * Beware: Before calling this function the owner of the data must be
 * saved in order to know where to transfer the data from. Once this
 * function returns, the ownership is transfered based on the access
 * mode and the knowledge about the location of the most up-to-date
 * version of the data is lost.
 */
int dague_data_transfer_ownership_to_copy(dague_data_t* data,
                                          uint8_t device,
                                          uint8_t access_mode)
{
    dague_data_copy_t* copy;
    uint32_t i;
    int valid_copy, transfer_required = 0;

    copy = data->device_copies[device];
    assert( NULL != copy );

    if( ACCESS_READ & access_mode ) copy->readers++;
    valid_copy = data->owner_device;

    if( DATA_COHERENCY_INVALID == copy->coherency_state ) {
        if( ACCESS_READ & access_mode ) transfer_required = 1;
        /* Update the coherency state of the others versions */
        if( ACCESS_WRITE & access_mode ) {
            valid_copy = data->owner_device = (uint8_t)device;
            for( i = 0; i < dague_nb_devices; i++ ) {
                if( NULL == data->device_copies[i] ) continue;
                data->device_copies[i]->coherency_state = DATA_COHERENCY_INVALID;
            }
            copy->coherency_state = DATA_COHERENCY_OWNED;
        }
        else if( ACCESS_READ & access_mode ) {
            transfer_required = 1; /* TODO: is this condition making sense? */
            copy->coherency_state = DATA_COHERENCY_SHARED;
        }
    }
    else { /* ! DATA_COHERENCY_INVALID */
        /* I either own the data or have one of its valid copies */
        if( DATA_COHERENCY_OWNED == copy->coherency_state ) {
            assert( device == data->owner_device ); /* memory is owned, better be me otherwise 2 writters: wrong JDF */
        }
        else {  /* I have one of the valid copies */
            if( ACCESS_WRITE & access_mode ) {
                /* Update the coherency state of the others versions */
                for( i = 0; i < dague_nb_devices; i++ ) {
                    if( NULL == data->device_copies[i] ) continue;
                    data->device_copies[i]->coherency_state = DATA_COHERENCY_INVALID;
                }
                copy->coherency_state = DATA_COHERENCY_OWNED;
                valid_copy = data->owner_device = (uint8_t)device;
            } else {
                /* The data is shared or exclusive and I'm doing a read */
            }
        }
    }
    if(valid_copy >= 0) {
        assert( data->device_copies[valid_copy]->version >= copy->version );
        /* The version on the GPU doesn't match the one in memory. Let the
         * upper level know a transfer is required.
         */
        transfer_required = transfer_required || (data->device_copies[valid_copy]->version > copy->version);
    }
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
    printf("-  [%d]: copy %p state %c readers %d version %u\n",
           (int)copy->device_index, copy, dump_coherency_codex(copy->coherency_state), copy->readers, copy->version);
}

void dague_dump_data(dague_data_t* data)
{
    printf("data %p key %x owner %d\n", data, data->key, data->owner_device);

    for( uint32_t i = 0; i < dague_nb_devices; i++ ) {
        if( NULL != data->device_copies[i])
            dague_dump_data_copy(data->device_copies[i]);
    }
}
