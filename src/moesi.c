/*
 * Copyright (c) 2012       The University of Tennessee and The University
 *                          of Tennessee Research Foundation.  All rights
 *                          reserved.
 */

#include "dague_config.h"
#include "moesi.h"
#include "debug.h"
#include "atomic.h"

void moesi_map_create(moesi_map_t** pmap, int nmasters, int ndevices) {
    moesi_map_t* map = *pmap;
    if( NULL != map ) {
        assert(nmasters <= map->nmasters);
        assert(ndevices <= map->ndevices);
        DEBUG3(("  Moesi:\tMap %p already exists (m=%d, d=%d): it does not need to be updated to hold (m=%d, d=%d)\n",
                pmap, map->nmasters, map->ndevices, nmasters, ndevices));
        return;
    }
    else {
        assert( nmasters > 0 );
        assert( ndevices < UINT16_MAX );
        map = calloc(1, sizeof(moesi_map_t) + (nmasters-1)*sizeof(moesi_master_t*));
        map->nmasters = nmasters;
        map->ndevices = (uint16_t)ndevices;
        *pmap = map;
        DEBUG3(("  Moesi:\tMap %p created (m=%d, d=%d)\n", pmap, nmasters, ndevices));
    }
}

void moesi_map_destroy(moesi_map_t** pmap) {
    moesi_map_t* map = *pmap;
    if( NULL != map ) {
        int i;
        for( i = 0; i < map->nmasters; i++ ) {
            if( NULL != map->masters[i] ) {
#ifdef DAGUE_DEBUG
                int d;
                for( d = 0; d < map->ndevices; d++ ) {
                    if( NULL != map->masters[i]->device_copies[d] ) {
                        WARNING(("  Moesi:\tpossible memory leak, moesi_copy_t %p is still in the moesi map %p but it is deallocated\n", map->masters[i]->device_copies[d], map));
                    }
                }
#endif
                free(map->masters[i]);
            }
        }
        free(map);
        DEBUG3(("  Moesi:\tMap %p destroyed\n", pmap));
        *pmap = NULL;
    }
    else {
        DEBUG3(("  Moesi:\tMap %p already destroyed (or never initialized)\n", pmap));
    }
}


int moesi_locate_device_with_valid_copy(moesi_map_t* map, moesi_key_t key) {
    moesi_master_t* master;
    moesi_copy_t* copy;
    int i;

    if( (NULL == map) || (NULL == (master = map->masters[key])) )
        return -1;

    for( i = 0; i < map->ndevices; i++ ) {
        if( NULL == (copy = master->device_copies[i]) )
            continue;
        if( MOESI_INVALID == copy->coherency_state )
            continue;
        return i;
    }
    return -2;
}

int moesi_prepare_transfer_to_device(moesi_map_t* map, moesi_key_t key, int device, uint8_t access_mode) {
    moesi_master_t* master;
    moesi_copy_t* copy;
    int i, transfer_required = 0;

    assert( NULL != map );
    assert( NULL != map->masters[key] );
    master = map->masters[key];
    assert( UINT16_MAX > device );
    assert( NULL != master->device_copies[device] );
    copy = master->device_copies[device];

    if( FLOW_ACCESS_READ & access_mode ) copy->readers++;

    if( MOESI_INVALID == copy->coherency_state ) {
        if( FLOW_ACCESS_READ & access_mode ) transfer_required = -1;
        /* Update the coherency state of the others versions */
        if( FLOW_ACCESS_WRITE & access_mode ) {
            //assert( MOESI_OWNED != master->coherency_state ); /* 2 writters on the same data: wrong JDF */
            master->coherency_state = MOESI_OWNED;
            master->owner_device = (uint16_t)device;
            for( i = 0; i < map->ndevices; i++ ) {
                if( NULL == master->device_copies[i] ) continue;
                master->device_copies[i]->coherency_state = MOESI_INVALID;
            }
            copy->coherency_state = MOESI_OWNED;
            copy->version = master->version;
        }
        else if( FLOW_ACCESS_READ & access_mode ) {
            if( MOESI_OWNED == master->coherency_state ) {
                transfer_required = 1; /* TODO: is this condition making sense? */
            }
            master->coherency_state = MOESI_SHARED;
            copy->coherency_state = MOESI_SHARED;
        }
    }
    else { /* !MOESI_INVALID */
        if( MOESI_OWNED == copy->coherency_state ) {
            assert( device == master->owner_device ); /* memory is owned, better be me otherwise 2 writters: wrong JDF */
        }
        else {
            if( FLOW_ACCESS_WRITE & access_mode ) {
                copy->coherency_state = MOESI_OWNED;
                master->owner_device = (uint16_t)device;
                /* Update the coherency state of the others versions */
            } else {
                /* The data is shared or exclusive and I'm doing a read */
            }
        }
    }

    assert( master->version >= copy->version );
    /* The version on the GPU doesn't match the one in memory. Let the
     * upper level know a transfer is required.
     */
    transfer_required = transfer_required || (master->version > copy->version);
    if( transfer_required )
        copy->version = master->version;
    return transfer_required;
}


int moesi_get_master(moesi_map_t* map, moesi_key_t key, moesi_master_t** pmaster) {
    moesi_master_t **from, *master = NULL;
    int rc = 0; /* the tile already existed */

    from = &(map->masters[key]);
    if( NULL == (master = *from) ) {
        master = (moesi_master_t*)calloc(1, sizeof(moesi_master_t) + (map->ndevices-1)*sizeof(moesi_copy_t*));
        master->map             = map;
        master->key             = key;
        master->mem_ptr         = NULL;
        master->owner_device    = -1;
        master->coherency_state = MOESI_INVALID;
        rc = 1;  /* the tile has just been created */
        if( 0 == dague_atomic_cas(from, NULL, master) ) {
            free(master);
            rc = 0;  /* the entry has been created by some other thread */
        }
    }
    *pmaster = *from;
    return rc;
}


int moesi_master_update(moesi_map_t *map, moesi_key_t key) {
    moesi_master_t* master;

    if( (NULL == map) || (NULL == (master = map->masters[key])) )
        return 0;

    if( MOESI_SHARED == master->coherency_state ) {
        int i;
        for( i = 0; i < map->ndevices; i++ ) {
            if( NULL == master->device_copies[i] ) continue;
            master->device_copies[i]->coherency_state = MOESI_INVALID;
        }
        master->coherency_state = MOESI_OWNED;
        master->owner_device = -1;
    }
    master->version++;
    return 0;
}

static char dump_moesi_codex(moesi_coherency_t state)
{
    if( MOESI_INVALID   == state ) return 'I';
    if( MOESI_OWNED     == state ) return 'O';
    if( MOESI_EXCLUSIVE == state ) return 'E';
    if( MOESI_SHARED    == state ) return 'S';
    return 'X';
}

void moesi_dump_moesi_copy( moesi_copy_t* copy )
{
    moesi_master_t* master = copy->master;

    printf("device_private %p coherency %c readers %d version %u\n"
           "  master %p [mem_ptr %p map %p key %u coherency %c owner %d version %u]\n",
           copy->device_private, dump_moesi_codex(copy->coherency_state), copy->readers, copy->version,
           copy->master, master->mem_ptr, master->map, master->key, dump_moesi_codex(master->coherency_state), master->owner_device, master->version);
}
