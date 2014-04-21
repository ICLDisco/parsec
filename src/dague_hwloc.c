/*
 * Copyright (c) 2010-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "debug.h"

#include "dague_hwloc.h"
#if defined(HAVE_HWLOC)
#include <hwloc.h>
#endif  /* defined(HAVE_HWLOC) */
#include <stdio.h>
#include <stdlib.h>

#if defined(HAVE_HWLOC)
static hwloc_topology_t topology;
static int first_init = 1;
#endif  /* defined(HAVE_HWLOC) */
static int ht = 1;

#if defined(HAVE_HWLOC_PARENT_MEMBER)
#define HWLOC_GET_PARENT(OBJ)  (OBJ)->parent
#else
#define HWLOC_GET_PARENT(OBJ)  (OBJ)->father
#endif  /* defined(HAVE_HWLOC_PARENT_MEMBER) */

#define MAX(x, y) ( (x)>(y)?(x):(y) )

int dague_hwloc_init(void)
{
#if defined(HAVE_HWLOC)
    if ( first_init ) {
        hwloc_topology_init(&topology);
        hwloc_topology_load(topology);
        first_init = 0;
    }
#endif  /* defined(HAVE_HWLOC) */
    return 0;
}

int dague_hwloc_fini(void)
{
#if defined(HAVE_HWLOC)
    hwloc_topology_destroy(topology);
    first_init = 1;
#endif  /* defined(HAVE_HWLOC) */
    return 0;
}

int dague_hwloc_export_topology(int *buflen, char **xmlbuffer)
{
#if defined(HAVE_HWLOC)
    if( first_init == 0 ) {
        return hwloc_topology_export_xmlbuffer(topology, xmlbuffer, buflen);
    } else {
        *buflen = 0;
        *xmlbuffer = NULL;
        return -1;
    }
#else
    *buflen = 0;
    *xmlbuffer = NULL;
    return -1;
#endif
}

void dague_hwloc_free_xml_buffer(char *xmlbuffer)
{
    if( NULL == xmlbuffer )
        return;

#if defined(HAVE_HWLOC)
    if( first_init == 0 ) {
        hwloc_free_xmlbuffer(topology, xmlbuffer);
    }
#endif
}

int dague_hwloc_distance( int id1, int id2 )
{
#if defined(HAVE_HWLOC)
    int count = 0;

    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, id1);
    hwloc_obj_t obj2 = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, id2);

    while( obj && obj2) {
        if(obj == obj2 ) {
            return count*2;
        }
        obj = HWLOC_GET_PARENT(obj);
        obj2 = HWLOC_GET_PARENT(obj2);
        count++;
    }
#endif  /* defined(HAVE_HWLOC) */
    (void)id1;(void)id2;
    return 0;
}

/* Previously use for the vpmap initialisation form hardware.
 * Should be obsolete as we now rely on the native hwloc topology to extrat the vp dedistribution.
 */
int dague_hwloc_master_id( int level, int processor_id )
{
#if defined(HAVE_HWLOC)
    int count = 0, div = 0, real_cores, cores;
    unsigned int i;

    real_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
    cores = real_cores;
    div = cores;

    if( 0 < (processor_id / cores) ) {
        while(processor_id) {
            if( (processor_id % div) == 0) {
                processor_id = count;
                break;
            }
            count++;
            div++;
            if( real_cores == count ) count = 0;
        }
    }

    for(i = 0; i < hwloc_get_nbobjs_by_depth(topology, level); i++) {
        hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, level, i);

#if !defined(HAVE_HWLOC_BITMAP)
        if(hwloc_cpuset_isset(obj->cpuset, processor_id)) {
            return hwloc_cpuset_first(obj->cpuset);
        }
#else
        if(hwloc_bitmap_isset(obj->cpuset, processor_id)) {
            return hwloc_bitmap_first(obj->cpuset);
        }
#endif
    }
#endif  /* defined(HAVE_HWLOC) */
    (void)level; (void)processor_id;
    return -1;
}

/* Previously use for the vpmap initialisation form hardware.
 * Should be obsolete as we now rely on the native hwloc topology to extrat the vp dedistribution.
 */
unsigned int dague_hwloc_nb_cores( int level, int master_id )
{
#if defined(HAVE_HWLOC)
    unsigned int i;

    for(i = 0; i < hwloc_get_nbobjs_by_depth(topology, level); i++){
        hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, level, i);
#if !defined(HAVE_HWLOC_BITMAP)
        if(hwloc_cpuset_isset(obj->cpuset, master_id)){
            return hwloc_cpuset_weight(obj->cpuset);
        }
#else
        if(hwloc_bitmap_isset(obj->cpuset, master_id)){
            return hwloc_bitmap_weight(obj->cpuset);
        }
#endif
    }
#endif  /* defined(HAVE_HWLOC) */
    (void)level; (void)master_id;
    return 0;
}


size_t dague_hwloc_cache_size( unsigned int level, int master_id )
{
#if defined(HAVE_HWLOC)
#if defined(HAVE_HWLOC_OBJ_PU) || 1
    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, master_id);
#else
    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PROC, master_id);
#endif  /* defined(HAVE_HWLOC_OBJ_PU) */

    while (obj) {
        if(obj->depth == level){
            if(obj->type == HWLOC_OBJ_CACHE){
#if defined(HAVE_HWLOC_CACHE_ATTR)
                return obj->attr->cache.size;
#else
                return obj->attr->cache.memory_kB;
#endif  /* defined(HAVE_HWLOC_CACHE_ATTR) */
            }
            return 0;
        }
        obj = HWLOC_GET_PARENT(obj);
    }
#endif  /* defined(HAVE_HWLOC) */
    (void)level; (void)master_id;
    return 0;
}

int dague_hwloc_nb_real_cores(void)
{
#if defined(HAVE_HWLOC)
    return hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
#else
    return -1;
#endif
}


int dague_hwloc_core_first_hrwd_ancestor_depth(void)
{
#if defined(HAVE_HWLOC)
    int level = MAX(hwloc_get_type_depth(topology, HWLOC_OBJ_NODE),hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET));
    assert(level < hwloc_get_type_depth(topology, HWLOC_OBJ_CORE));
    return level;
#endif  /* defined(HAVE_HWLOC) */
    return -1;
}

int dague_hwloc_get_nb_objects(int level)
{
#if defined(HAVE_HWLOC)
    return hwloc_get_nbobjs_by_depth(topology, level);
#endif  /* defined(HAVE_HWLOC) */
    (void)level;
    return -1;
}


int dague_hwloc_socket_id(int core_id )
{
#if defined(HAVE_HWLOC)
    hwloc_obj_t core =  hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, core_id);
    hwloc_obj_t socket = NULL;
    if ((socket = hwloc_get_ancestor_obj_by_type(topology , HWLOC_OBJ_SOCKET, core)) != NULL)
    {
        return socket->logical_index;

    }else{
        return -1;
    }
#endif  /* defined(HAVE_HWLOC) */
    (void)core_id;
    return -1;
}

int dague_hwloc_numa_id(int core_id )
{
#if defined(HAVE_HWLOC)
    hwloc_obj_t core =  hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, core_id);
    hwloc_obj_t node = NULL;
    if ((node = hwloc_get_ancestor_obj_by_type(topology , HWLOC_OBJ_NODE, core)) != NULL)
    {
        return node->logical_index;

    }else{
        return -1;
    }
#endif  /* defined(HAVE_HWLOC) */
    (void)core_id;
    return -1;
}

unsigned int dague_hwloc_nb_cores_per_obj( int level, int index )
{
#if defined(HAVE_HWLOC)
    hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, level, index);
    assert( obj != NULL );
    return hwloc_get_nbobjs_inside_cpuset_by_type(topology, obj->cpuset, HWLOC_OBJ_CORE);
#else
    (void)level; (void)index;
    return -1;
#endif  /* defined(HAVE_HWLOC) */
}

int dague_hwloc_nb_levels(void)
{
#if defined(HAVE_HWLOC)
    return hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
#else
    return -1;
#endif  /* defined(HAVE_HWLOC) */
}


int dague_hwloc_bind_on_core_index(int cpu_index, int local_ht_index)
{
    (void)cpu_index; (void)local_ht_index;
#if defined(HAVE_HWLOC)
    hwloc_obj_t      obj, core;      /* Hwloc object    */
    hwloc_cpuset_t   cpuset;   /* Hwloc cpuset    */

    /* Get the core of index cpu_index */
    core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, cpu_index);
    if (!core) {
        WARNING(("dague_hwloc: unable to get the core of index %i (nb physical cores = %i )\n",
                 cpu_index,  dague_hwloc_nb_real_cores()));
        return -1;
    }
   /* Get the cpuset of the core if not using SMT/HyperThreading,
    * get the cpuset of the designated child object (PU) otherwise */
    if ( local_ht_index > -1) {
        if ( (uint32_t)local_ht_index < core->arity )
            obj = core->children[local_ht_index];
        else
            obj = core->children[0];

        if (!obj) {
            WARNING(("dague_hwloc: unable to get the core of index %i, HT %i (nb cores = %i)\n",
                     cpu_index, local_ht_index, dague_hwloc_nb_real_cores()));
            return -1;
        }
    }
    else
        obj = core;

    /* Get a copy of its cpuset that we may modify.  */
#if !defined(HAVE_HWLOC_BITMAP)
    cpuset = hwloc_cpuset_dup(obj->cpuset);
    hwloc_cpuset_singlify(cpuset);
#else
    cpuset = hwloc_bitmap_dup(obj->cpuset);
    hwloc_bitmap_singlify(cpuset);
#endif

    /* And try to bind ourself there.  */
    if (hwloc_set_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD)) {
        char *str = NULL;
#if !defined(HAVE_HWLOC_BITMAP)
        hwloc_cpuset_asprintf(&str, obj->cpuset);
#else
        hwloc_bitmap_asprintf(&str, obj->cpuset);
#endif
        WARNING(("dague_hwloc: couldn't bind to cpuset %s\n", str));
        free(str);

        /* Free our cpuset copy */
#if !defined(HAVE_HWLOC_BITMAP)
        hwloc_cpuset_free(cpuset);
#else
        hwloc_bitmap_free(cpuset);
#endif
        return -1;
    }
    DEBUG2(("Thread bound on core index %i, [HT %i ]\n", cpu_index, local_ht_index));

    /* Get the number at Proc level*/
    cpu_index = obj->os_index;

    /* Free our cpuset copy */
#if !defined(HAVE_HWLOC_BITMAP)
    hwloc_cpuset_free(cpuset);
#else
    hwloc_bitmap_free(cpuset);
#endif
    return cpu_index;
#else
    return -1;
#endif
}

int dague_hwloc_bind_on_mask_index(hwloc_cpuset_t cpuset)
{
#if defined(HAVE_HWLOC) && defined(HAVE_HWLOC_BITMAP)
    unsigned cpu_index;
    int first_free;
    hwloc_obj_t obj;
    hwloc_cpuset_t binding_mask=hwloc_bitmap_alloc();

    /* For each index in the mask, get the associated cpu object and use its cpuset to add it to the binding mask */
    hwloc_bitmap_foreach_begin(cpu_index, cpuset)
    {
        /* Get the core of index cpu */
        obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, cpu_index);
        if (!obj) {
            DEBUG3(("dague_hwloc_bind_on_mask_index: unable to get the core of index %i\n", cpu_index));
        } else {
            hwloc_bitmap_or(binding_mask, binding_mask, obj->cpuset);
        }
    }
    hwloc_bitmap_foreach_end();

    if (hwloc_set_cpubind(topology, binding_mask, HWLOC_CPUBIND_THREAD)) {
        char *str = NULL;
        hwloc_bitmap_asprintf(&str, binding_mask);
        WARNING(("Couldn't bind to cpuset %s\n", str));
        free(str);
        return -1;
    }

#if (DAGUE_DEBUG_VERBOSE != 0) && defined(HAVE_HWLOC_BITMAP)
    {
        char *str = NULL;
        hwloc_bitmap_asprintf(&str,  binding_mask);
        DEBUG2(("Thread bound on the cpuset  %s\n", str));
        free(str);
    }
#endif /* DAGUE_DEBUG_VERBOSE */

    first_free = hwloc_bitmap_first(binding_mask);
    hwloc_bitmap_free(binding_mask);
    return first_free;
#else
    (void) cpuset;
    return -1;
#endif /* HAVE_HWLOC && HAVE_HWLOC_BITMAP */
}

int dague_hwloc_allow_ht(int htnb)
{
    assert( htnb > 0 );
#if defined(HAVE_HWLOC) && defined(HAVE_HWLOC_BITMAP)
    dague_hwloc_init();

    /* Check the validity of the parameter. Correct otherwise  */
    if (htnb > 1){
        int pu_per_core = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU)/hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
        if( htnb > pu_per_core){
            printf("Warning:: HyperThreading:: There not enought logical processors to consider %i HyperThreads per core (set up to %i)\n", htnb,  pu_per_core);
            ht=pu_per_core;
        }else{
            ht=htnb;
        }
    }
    ht=htnb;
    return ht;
#else
    /* Without hwloc, trust your user to give a correct parameter */
    ht = htnb;
    return ht;
#endif
}

int dague_hwloc_get_ht(void)
{
    return ht;
}

