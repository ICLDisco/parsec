/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_MOESI_H_HAS_BEEN_INCLUDED
#define DAGUE_MOESI_H_HAS_BEEN_INCLUDED

/**
 * Data coherency protocol based on MOESI.
 */

#include "dague_config.h"

typedef uint8_t moesi_coherency_t;
#define    MOESI_INVALID   ((uint8_t)0x0)
#define    MOESI_OWNED     ((uint8_t)0x1)
#define    MOESI_EXCLUSIVE ((uint8_t)0x2)
#define    MOESI_SHARED    ((uint8_t)0x4)

typedef struct _moesi_copy      moesi_copy_t;
typedef struct _moesi_map       moesi_map_t;

typedef uint32_t moesi_key_t;

#include "dague_internal.h"

/**
 * A moesi map contains an array of pointers to all the moesi master
 * representing the locally (on rank) stored blocks.
 * Blocks that are not locally stored have a NULL master
 */
struct _moesi_map {
    int      nmasters;
    uint16_t ndevices;
    moesi_master_t* masters[1];
};

/**
 * A moesi master represents a specific data block (and links to all it copies).
 * It can be found based on a unique key.
 */
struct _moesi_master {
    void*               mem_ptr;
    moesi_map_t*        map;
    moesi_key_t         key;
    moesi_coherency_t   coherency_state;
    uint16_t            owner_device;
    uint32_t            version;
    moesi_copy_t*       device_copies[1]; /* this array allocated according to the number of devices */
};

/**
 * a moesi copy represent a specific copy of a master block on a
 * particular device.
 */
struct _moesi_copy {
    void*               device_private;
    moesi_master_t*     master;
    moesi_coherency_t   coherency_state;
    int16_t             readers;
    uint32_t            version;
};

void moesi_map_create(moesi_map_t** map, int nmasters, int ndevices);
void moesi_map_destroy(moesi_map_t** map);

/**
 * Return (and create if necessary) the master entry used for handling
 * MOESI protocol on a specific data block.
 * Devices will have to add their own entries for copies they make in the
 * moesi_copies array.
 */
int moesi_get_master(moesi_map_t* map, moesi_key_t key, moesi_master_t** pmaster);

/**
 * Return the device index of a device that contains an up-to-date
 * version of the data block.
 * If the returned value is negative, the master copy is authoritative.
 */
int moesi_locate_device_with_valid_copy(moesi_map_t* map, moesi_key_t key);

/**
 * Prepares the transfer of a data (refered by key in the moesi map) to the device.
 * The function returns !0 if the data needs to be staged in.
 * The state of the MOESI map is updated accordingly.
 * The moesi_copy must be filled for the target device.
 */
int moesi_prepare_transfer_to_device(moesi_map_t* map, moesi_key_t key, int device,
                                     uint8_t access_mode);

/**
 * The master copy is accessed in WRITE mode, invalidate all shared copies
 */
int moesi_master_update(moesi_map_t *map, moesi_key_t key);

/**
 * Debugging functions.
 */
extern void moesi_dump_moesi_copy( moesi_copy_t* copy );

#endif
