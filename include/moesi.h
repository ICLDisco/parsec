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
#include "dague_internal.h"
#include "list_item.h"

typedef uint8_t moesi_coherency_t;
#define    MOESI_INVALID   ((uint8_t)0x0)
#define    MOESI_OWNED     ((uint8_t)0x1)
#define    MOESI_EXCLUSIVE ((uint8_t)0x2)
#define    MOESI_SHARED    ((uint8_t)0x4)

typedef struct _moesi_master    moesi_master_t;
typedef struct _moesi_copy      moesi_copy_t;
typedef struct _moesi_map       moesi_map_t;

typedef uint32_t moesi_key_t;

/**
 * A moesi map contains an array of pointers to all the moesi master 
 * representing the locally (on rank) stored blocks. 
 * Blocks that are not locally stored have a NULL master
 */
struct _moesi_map { 
    uint16_t nmasters;
    uint16_t ndevices;
    moesi_master_t* masters[1];
};

/**
 * A moesi master represents a specific data block (and links to all it copies).
 * It can be found based on a unique key.
 */
struct _moesi_master {
    void*               master_ptr;
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
    dague_list_item_t   item;
    moesi_map_t*        map;
    moesi_coherency_t   coherency_state;
    int16_t             readers;
    uint32_t            version;
    void*               device_private;
};



void moesi_map_create(moesi_map_t** map, int nmasters, int ndevices);
void moesi_map_destroy(moesi_map_t** map);

int moesi_write_owner(moesi_map_t* map, moesi_key_t key);
int moesi_get_master(moesi_map_t* map, moesi_key_t key, moesi_master_t** master);
int moesi_update_version(moesi_map_t* map, moesi_key_t key);

#endif
