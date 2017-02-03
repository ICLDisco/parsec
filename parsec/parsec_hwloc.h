/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef HWLOC_H_HAS_BEEN_INCLUDED
#define HWLOC_H_HAS_BEEN_INCLUDED

/** @addtogroup parsec_internal_binding
 *  @{
 */

#include <stdio.h>
#include <stdlib.h>

BEGIN_C_DECLS

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
extern int parsec_hwloc_master_id( int level, int processor_id );

/**
 * Find the number of core for master_id at n level
 *
 */
extern unsigned int parsec_hwloc_nb_cores( int level, int master_id );

/**
 * Find the number of level from the computer architecture
 *
 */
extern int parsec_hwloc_nb_levels( void );

/**
 * Find the cache size for master at n level
 *
 */
extern size_t parsec_hwloc_cache_size( unsigned int level, int master_id );

/**
 * Find the distance between id1 and id2
 *
 */
extern int parsec_hwloc_distance( int id1, int id2 );

/**
 * load the HWLOC topology.
 */
extern int parsec_hwloc_init(void);

/**
 * unload the HWLOC topology.
 */
extern int parsec_hwloc_fini(void);

/**
 * Find the number of core of the architecture.
 *
 */
extern int parsec_hwloc_nb_real_cores();

/**
 * Bind the current thread on the core of index cpu_index.
 *
 */
int parsec_hwloc_bind_on_core_index(int cpu_index, int ht_index);

/**
 * Gives a readable representation of the cpuset the thread is bound to
 * Result has to be freed.
 */
char *parsec_hwloc_get_binding(void);

/**
 * Return the logical socket index for a core index (hwloc numbering).
 */
int parsec_hwloc_socket_id(int core_id);

/**
 * Return the logical NUMA node index for a core index (hwloc numbering).
 */
int parsec_hwloc_numa_id(int core_id);

/**
 * Return the depth of the first core hardware ancestor: NUMA node or socket.
 */
int parsec_hwloc_core_first_hrwd_ancestor_depth();

/**
 * Return the number of hwloc objects at the "level" depth.
 */
int parsec_hwloc_get_nb_objects(int level);

/**
 * Return the number of hwloc objects at the "level" depth.
 */
int parsec_hwloc_get_nb_objects(int level);


/**
 * Find the number of core under the object number index at the topology depth level.
 */
unsigned int parsec_hwloc_nb_cores_per_obj( int level, int index );

/**
 * Exports the loaded topology to an XML buffer.
 * @param [OUT] buflen: the size of the buffer as allocated by the function
 * @param [OUT] xmlbuffer: the buffer containing an XML representation.
 *              this buffer should then be freed using parsec_hwloc_free_xml_buffer
 *
 * @return -1 if an error
 */
int parsec_hwloc_export_topology(int *buflen, char **xmlbuffer);

/**
 * Frees memory allocated by parsec_hwloc_export_topology
 * @param [IN] xmlbuffer: the buffer to free.
 */
void parsec_hwloc_free_xml_buffer(char *xmlbuffer);

/**
 * Bind the current thread according the mask of index mask_index.
 *
 */
#if defined(PARSEC_HAVE_HWLOC)
#include <hwloc.h>
#else
typedef int hwloc_cpuset_t;
#endif
int parsec_hwloc_bind_on_mask_index(hwloc_cpuset_t mask_index);

/**
 * Allow serial thread binding per core to use the SMT/HT capabilities of the processor 
 *
 */
int parsec_hwloc_allow_ht(int htnb);
int parsec_hwloc_get_ht(void);

END_C_DECLS

/** @} */

#endif  /* HWLOC_H_HAS_BEEN_INCLUDED */
