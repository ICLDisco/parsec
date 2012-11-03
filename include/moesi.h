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

typedef uint8_t dague_moesi_coherency_t;
#define    MOESI_INVALID   ((uint8_t)0x0)
#define    MOESI_OWNED     ((uint8_t)0x1)
#define    MOESI_EXCLUSIVE ((uint8_t)0x2)
#define    MOESI_SHARED    ((uint8_t)0x4)

typedef struct dague_moesi_copy_s  dague_moesi_copy_t;
typedef struct dague_moesi_map_s   dague_moesi_map_t;

typedef uint32_t dague_moesi_key_t;

#include "dague_internal.h"

/**
 * A moesi map contains an array of pointers to all the moesi master
 * representing the locally (on rank) stored blocks.
 * Blocks that are not locally stored have a NULL master
 */
struct dague_moesi_map_s {
    int      nmasters;
    uint16_t ndevices;
    dague_moesi_master_t* masters[1];
};

/**
 * A moesi master represents a specific data block (and links to all it copies).
 * It can be found based on a unique key.
 */
struct dague_moesi_master_s {
    void                    *mem_ptr;
    dague_moesi_map_t       *map;
    dague_moesi_key_t        key;
    dague_moesi_coherency_t  coherency_state;
    uint8_t                  owner_device;
    uint16_t                 version;
    dague_moesi_copy_t      *device_copies[1]; /* this array allocated according to the number of devices */
};

/**
 * a moesi copy represent a specific copy of a master block on a
 * particular device.
 */
struct dague_moesi_copy_s {
    void                    *device_private;
    dague_moesi_master_t    *master;
    dague_moesi_coherency_t  coherency_state;
    uint8_t                  owner_device;
    int16_t                  readers;
    uint32_t                 version;
};

void moesi_map_create(dague_moesi_map_t** map, int nmasters, int ndevices);
void moesi_map_destroy(dague_moesi_map_t** map);

/**
 * Return (and create if necessary) the master entry used for handling
 * MOESI protocol on a specific data block.
 * Devices will have to add their own entries for copies they make in the
 * moesi_copies array.
 */
int moesi_get_master(dague_moesi_map_t* map,
                     dague_moesi_key_t key,
                     dague_moesi_master_t** pmaster);

/**
 * Return the device index of a device that contains an up-to-date
 * version of the data block.
 * If the returned value is negative, the master copy is authoritative.
 */
int moesi_locate_device_with_valid_copy(dague_moesi_map_t* map, dague_moesi_key_t key);

/**
 * Prepares the transfer of a data (refered by key in the moesi map) to the device.
 * The function returns !0 if the data needs to be staged in.
 * The state of the MOESI map is updated accordingly.
 * The moesi_copy must be filled for the target device.
 */
int moesi_prepare_transfer_to_device(dague_moesi_map_t* map,
                                     dague_moesi_key_t key, int device,
                                     uint8_t access_mode);

/**
 * The master copy is accessed in WRITE mode, invalidate all shared copies
 */
int moesi_master_update(dague_moesi_map_t *map,
                        dague_moesi_key_t key);

/**
 * Debugging functions.
 */
extern void moesi_dump_moesi_copy(dague_moesi_copy_t* copy);

#endif
