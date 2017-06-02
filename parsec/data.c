/*
 * Copyright (c) 2012-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#include "parsec/class/lifo.h"
#include "parsec/constants.h"
#include "parsec/devices/device.h"
#include "parsec/utils/output.h"
#include "parsec/data_internal.h"
#include "parsec/arena.h"

static parsec_lifo_t parsec_data_lifo;
static parsec_lifo_t parsec_data_copies_lifo;

static void parsec_data_copy_construct(parsec_data_copy_t* obj)
{
    obj->device_index         = 0;
    obj->flags                = 0;
    obj->coherency_state      = DATA_COHERENCY_INVALID;
    obj->readers              = 0;
    obj->version              = 0;
    obj->older                = NULL;
    obj->original             = NULL;
    obj->device_private       = NULL;
    obj->arena_chunk          = NULL;
    obj->data_transfer_status = DATA_STATUS_NOT_TRANSFER;
    obj->push_task            = NULL;
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Allocate data copy %p", obj);
}

static void parsec_data_copy_destruct(parsec_data_copy_t* obj)
{
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Destruct data copy %p (attached to %p)", obj, obj->original);

    /* If the copy is still attached to a data we should detach it first */
    if( NULL != obj->original) {
        parsec_data_copy_detach(obj->original, obj, obj->device_index);
        assert( NULL == obj->original );
    }

    if( obj->flags & PARSEC_DATA_FLAG_ARENA ) {
        /* It is an arena that is now unused.
         * give the chunk back to the arena memory management.
         * This detaches obj from obj->original, and frees everything */
        parsec_arena_release(obj);
    }
}

OBJ_CLASS_INSTANCE(parsec_data_copy_t, parsec_list_item_t,
                   parsec_data_copy_construct,
                   parsec_data_copy_destruct);

static void parsec_data_construct(parsec_data_t* obj )
{
    obj->owner_device     = -1;
    obj->key              = 0;
    obj->nb_elts          = 0;
    for( uint32_t i = 0; i < parsec_nb_devices;
         obj->device_copies[i] = NULL, i++ );
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Allocate data %p", obj);
}

static void parsec_data_destruct(parsec_data_t* obj )
{
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Release data %p", obj);
    for( uint32_t i = 0; i < parsec_nb_devices; i++ ) {
        parsec_data_copy_t *copy = NULL;
        parsec_device_t *device = parsec_devices_get(i);
        assert(NULL != device);
        while( (copy = obj->device_copies[i]) != NULL )
        {
            assert(obj->super.obj_reference_count > 1);
            parsec_data_copy_detach( obj, copy, i );
            if ( !(device->type & PARSEC_DEV_CUDA) ){
                /**
                 * GPU copies are normally stored in LRU lists, and must be
                 * destroyed by the release list to free the memory on the device
                 */
                OBJ_RELEASE( copy );
            }
        }
        assert(NULL == obj->device_copies[i]);
    }
}

OBJ_CLASS_INSTANCE(parsec_data_t, parsec_object_t,
                   parsec_data_construct,
                   parsec_data_destruct
                   );

int parsec_data_init(parsec_context_t* context)
{
    OBJ_CONSTRUCT(&parsec_data_lifo, parsec_lifo_t);
    OBJ_CONSTRUCT(&parsec_data_copies_lifo, parsec_lifo_t);
    /**
     * This is a trick. Now that we know the number of available devices
     * we can update the size of the parsec_data_t class to the correct value.
     */
    if( !parsec_devices_freezed(context) ) {
        parsec_warning("Cannot configure the data infrastructure as the devices layer has not yet been froze.");
        return PARSEC_ERROR;
    }
    parsec_data_t_class.cls_sizeof += sizeof(parsec_data_copy_t*) * parsec_nb_devices;
    return 0;
}

int parsec_data_fini(parsec_context_t* context)
{
    OBJ_DESTRUCT(&parsec_data_lifo);
    OBJ_DESTRUCT(&parsec_data_copies_lifo);
    (void)context;
    return 0;
}

/**
 *
 */
parsec_data_t* parsec_data_new(void)
{
    parsec_data_t* item = (parsec_data_t*)parsec_lifo_pop(&parsec_data_lifo);
    if( NULL == item ) {
        item = OBJ_NEW(parsec_data_t);
        if( NULL == item ) return NULL;
    } else {
        OBJ_CONSTRUCT(item, parsec_object_t);
    }
    return item;
}

/**
 *
 */
void parsec_data_delete(parsec_data_t* data)
{
    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)data);
    parsec_lifo_push(&parsec_data_lifo, (parsec_list_item_t*)data);
}

inline int
parsec_data_copy_attach(parsec_data_t* data,
                       parsec_data_copy_t* copy,
                       uint8_t device)
{
    assert(NULL == copy->original);
    assert(NULL == copy->older);

    copy->device_index    = device;
    copy->original        = data;
    /* Atomically set the device copy */
    copy->older = data->device_copies[device];
    if( !parsec_atomic_cas_ptr(&data->device_copies[device], copy->older, copy) ) {
        copy->older = NULL;
        return PARSEC_ERROR;
    }
    OBJ_RETAIN(data);
    return PARSEC_SUCCESS;
}

/**
 * In the current version only the latest copy of a data for each device can
 * be safely removed.
 */
int parsec_data_copy_detach(parsec_data_t* data,
                           parsec_data_copy_t* copy,
                           uint8_t device)
{
    parsec_data_copy_t* obj = data->device_copies[device];

    if( obj != copy )
        return PARSEC_ERR_NOT_FOUND;
    /* Atomically set the device copy */
    if( !parsec_atomic_cas_ptr(&data->device_copies[device], copy, copy->older) ) {
        return PARSEC_ERROR;
    }
    copy->original     = NULL;
    copy->older        = NULL;
    OBJ_RELEASE(data);
    return PARSEC_SUCCESS;
}

/**
 * Allocate a data copy and attach it as a device specific copy. The data must
 * not be NULL in order for this operation to be relevant.
 */
parsec_data_copy_t* parsec_data_copy_new(parsec_data_t* data, uint8_t device)
{
    parsec_data_copy_t* copy;

    copy = (parsec_data_copy_t*)parsec_lifo_pop(&parsec_data_copies_lifo);
    if( NULL == copy ) {
        copy = OBJ_NEW(parsec_data_copy_t);
        if( NULL == copy ) {
            return NULL;
        }
    } else {
        OBJ_CONSTRUCT(copy, parsec_data_copy_t);
    }
    if( PARSEC_SUCCESS != parsec_data_copy_attach(data, copy, device) ) {
        OBJ_RELEASE(copy);
        return NULL;
    }
    return copy;
}


#if 0
/*
 * WARNING: Is this function usefull or should it be removed ?
 */
/**
 * Find the corresponding copy of the data on the requested device. If the
 * copy is not available in the access mode requested, a new version will
 * be created. If no tranfer is required the correct version will be set
 * into dest and the function returns 0. If a transfer is necessary, dest
 * will contain the pointer to the copy to be tranferred to, and the function
 * will return 1. All other cases should be considered as errors, and a
 * negative value must be returned (corresponding to a specific PARSEC_
 * error code.
 */
int parsec_data_get_device_copy(parsec_data_copy_t* source,
                               parsec_data_copy_t** dest,
                               uint8_t device,
                               uint8_t access_mode)
{
    parsec_data_copy_t* copy;
    parsec_data_t* original;
    int transfer = 0;

    if( device == source->device_index ) {
        *dest = source;
        return 0;
    }
    original = source->original;
    /* lock the original data */
    copy = original->device_copies[device];
    while( NULL != copy ) {
        if( source->version == copy->version )
            break;
        copy = copy->older;
    }
    if( NULL == copy ) {
        *dest = copy = parsec_data_copy_new(original, device);
        transfer = 1;
    } else if( source->version == copy->version ) {
        *dest = copy;
    }
    /* unlock the original data */

    return transfer;
}
#endif

/**
 * Beware: Before calling this function the owner of the data must be
 * saved in order to know where to transfer the data from. Once this
 * function returns, the ownership is transfered based on the access
 * mode and the knowledge about the location of the most up-to-date
 * version of the data is lost.
 */
int parsec_data_transfer_ownership_to_copy(parsec_data_t* data,
                                          uint8_t device,
                                          uint8_t access_mode)
{
    uint32_t i;
    int transfer_required = 0;
    int valid_copy = data->owner_device;
    parsec_data_copy_t* copy = data->device_copies[device];
    assert( NULL != copy );

    switch( copy->coherency_state ) {
    case DATA_COHERENCY_INVALID:
        transfer_required = 1;
        if( -1 == valid_copy ) {
            for( i = 0; i < parsec_nb_devices; i++ ) {
                if( NULL == data->device_copies[i] ) continue;
                if( DATA_COHERENCY_INVALID == data->device_copies[i]->coherency_state ) continue;
                assert( DATA_COHERENCY_EXCLUSIVE == data->device_copies[i]->coherency_state
                     || DATA_COHERENCY_SHARED == data->device_copies[i]->coherency_state );
                valid_copy = i;
            }
        }
        break;

    case DATA_COHERENCY_SHARED:
        for( i = 0; i < parsec_nb_devices; i++ ) {
            if( NULL == data->device_copies[i] ) continue;
            if( DATA_COHERENCY_OWNED == data->device_copies[i]->coherency_state
             && data->device_copies[i]->version > copy->version ) {
                assert( (int)i == valid_copy );
                transfer_required = 1;
            }
#if defined(PARSEC_DEBUG_PARANOID)
            else {
                assert( DATA_COHERENCY_INVALID == data->device_copies[i]->coherency_state
                     || DATA_COHERENCY_SHARED == data->device_copies[i]->coherency_state
                     || data->device_copies[i]->version == copy->version
                     || copy->data_transfer_status );
            }
#endif  /* defined(PARSEC_DEBUG_PARANOID) */
        }
        break;

    case DATA_COHERENCY_EXCLUSIVE:
#if defined(PARSEC_DEBUG_PARANOID)
        for( i = 0; i < parsec_nb_devices; i++ ) {
            if( device == i || NULL == data->device_copies[i] ) continue;
            assert( DATA_COHERENCY_INVALID == data->device_copies[i]->coherency_state );
        }
#endif  /* defined(PARSEC_DEBUG_PARANOID) */
        break;

    case DATA_COHERENCY_OWNED:
        assert( device == data->owner_device ); /* memory is owned, better be me otherwise 2 writters: wrong JDF */
#if defined(PARSEC_DEBUG_PARANOID)
        for( i = 0; i < parsec_nb_devices; i++ ) {
            if( device == i || NULL == data->device_copies[i] ) continue;
            assert( DATA_COHERENCY_INVALID == data->device_copies[i]->coherency_state
                 || DATA_COHERENCY_SHARED == data->device_copies[i]->coherency_state );
            assert( copy->version >= data->device_copies[i]->version );
        }
#endif  /* defined(PARSEC_DEBUG_PARANOID) */
        break;
    }

    if( FLOW_ACCESS_READ & access_mode ) {
        for( i = 0; i < parsec_nb_devices; i++ ) {
            if( device == i || NULL == data->device_copies[i] ) continue;
            if( DATA_COHERENCY_INVALID == data->device_copies[i]->coherency_state ) continue;
            if( DATA_COHERENCY_OWNED == copy->coherency_state
             && !(FLOW_ACCESS_WRITE & access_mode) ) {
                 if( data->device_copies[i]->version < copy->version ) {
                     data->device_copies[i]->coherency_state = DATA_COHERENCY_INVALID;
                 }
                 data->owner_device = -1;
            }
            if( DATA_COHERENCY_EXCLUSIVE == data->device_copies[i]->coherency_state ) {
                data->device_copies[i]->coherency_state = DATA_COHERENCY_SHARED;
            }
        }
        copy->readers++;
        copy->coherency_state = DATA_COHERENCY_SHARED;
    }
    else transfer_required = 0; /* finally we'll just overwrite w/o read */

    if( FLOW_ACCESS_WRITE & access_mode ) {
        for( i = 0; i < parsec_nb_devices; i++ ) {
            if( NULL == data->device_copies[i] ) continue;
            if( DATA_COHERENCY_INVALID == data->device_copies[i]->coherency_state ) continue;
            data->device_copies[i]->coherency_state = DATA_COHERENCY_SHARED;
        }
        data->owner_device = (uint8_t)device;
        copy->coherency_state = DATA_COHERENCY_OWNED;
    }

    if( !transfer_required ) {
        return -1;
    }
    assert( -1 != valid_copy );
    assert( data->device_copies[valid_copy]->version >= copy->version );
    return valid_copy;
}

static char dump_coherency_codex(parsec_data_coherency_t state)
{
    if( DATA_COHERENCY_INVALID   == state ) return 'I';
    if( DATA_COHERENCY_OWNED     == state ) return 'O';
    if( DATA_COHERENCY_EXCLUSIVE == state ) return 'E';
    if( DATA_COHERENCY_SHARED    == state ) return 'S';
    return 'X';
}

void parsec_dump_data_copy(parsec_data_copy_t* copy)
{
    printf("-  [%d]: copy %p state %c readers %d version %u\n",
           (int)copy->device_index, copy, dump_coherency_codex(copy->coherency_state), copy->readers, copy->version);
}

void parsec_dump_data(parsec_data_t* data)
{
    printf("data %p key %x owner %d\n", data, data->key, data->owner_device);

    for( uint32_t i = 0; i < parsec_nb_devices; i++ ) {
        if( NULL != data->device_copies[i])
            parsec_dump_data_copy(data->device_copies[i]);
    }
}

parsec_data_copy_t*
parsec_data_get_copy(parsec_data_t* data, uint32_t device)
{
    return PARSEC_DATA_GET_COPY(data, device);
}

void parsec_data_copy_release(parsec_data_copy_t* copy)
{
    /* TODO: Move the copy back to the CPU before destroying it */
    PARSEC_DATA_COPY_RELEASE(copy);
}

void* parsec_data_copy_get_ptr(parsec_data_copy_t* data)
{
    return PARSEC_DATA_COPY_GET_PTR(data);
}

/* Return the pointer on the selected device */
void* parsec_data_get_ptr(parsec_data_t* data, uint32_t device)
{
    parsec_data_copy_t *copy = parsec_data_get_copy( data, device );
    return PARSEC_DATA_COPY_GET_PTR(copy);
}

parsec_data_t *
parsec_data_create( parsec_data_t **holder,
                   parsec_ddesc_t *desc,
                   parsec_data_key_t key, void *ptr, size_t size )
{
    parsec_data_t *data = *holder;

    if( NULL == data ) {
        parsec_data_copy_t* data_copy = OBJ_NEW(parsec_data_copy_t);
        data = OBJ_NEW(parsec_data_t);

        data_copy->coherency_state = DATA_COHERENCY_OWNED;
        data_copy->device_private = ptr;

        data->owner_device = 0;
        data->key = key;
        data->ddesc = desc;
        data->nb_elts = size;
        parsec_data_copy_attach(data, data_copy, 0);

        if( !parsec_atomic_cas_ptr(holder, NULL, data) ) {
            parsec_data_copy_detach(data, data_copy, 0);
            OBJ_RELEASE(data_copy);
            data = *holder;
        }
    } else {
        /* Do we have a copy of this data */
        if( NULL == data->device_copies[0] ) {
            parsec_data_copy_t* data_copy = parsec_data_copy_new(data, 0);
            data_copy->device_private = ptr;
        }
    }
    assert( data->key == key );
    return data;
}

void
parsec_data_destroy( parsec_data_t *data )
{
    if (data != NULL)
    {
        /*
         * Need to call destruct before release due to circular
         * dependency between the parsec_data_copy_t and the parsec_data_t
         */
        OBJ_DESTRUCT(data);
#if defined(PARSEC_DEBUG_PARANOID)
        ((parsec_object_t *)(data))->obj_magic_id = PARSEC_OBJ_MAGIC_ID;
#endif
        OBJ_RELEASE(data);
    }
}
