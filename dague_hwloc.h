/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef HWLOC_H_HAS_BEEN_INCLUDED
#define HWLOC_H_HAS_BEEN_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <hwloc.h>

typedef struct {
            int lvl;
            int processor_id;
            int master_id;
            int id1;
            int id2;
            int set;
} hwloc_info;

/**
 * Find the master for the processor_id at n level
 *
 */
extern int DAGuE_hwloc_master_id( int level, int processor_id );
 
/**
 * Find the number of core for master_id at n level 
 *
 */
extern unsigned int DAGuE_hwloc_nb_cores( int level, int master_id );
 
/**
 * Find the number of level from the computer architectur
 *
 */
extern int DAGuE_hwloc_nb_levels( void );
    
/**
 * Find the cache size for master at n level
 *
 */
extern size_t DAGuE_hwloc_cache_size( int level, int master_id );

/**
 * Find the distance between id1 and id2
 *
 */
extern int DAGuE_hwloc_distance( int id1, int id2 );

/**
 * load the HWLOC topology.
 */
extern int DAGuE_hwloc_init(void);

/**
 * unload the HWLOC topology.
 */
extern int DAGuE_hwloc_fini(void);

#endif  /* HWLOC_H_HAS_BEEN_INCLUDED */
