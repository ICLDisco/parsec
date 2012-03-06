/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague.h"

#include "dague_hwloc.h"
#if defined(HAVE_HWLOC)
#include <hwloc.h>
#endif  /* defined(HAVE_HWLOC) */
#include <stdio.h>
#include <stdlib.h>

#if defined(HAVE_HWLOC)
static hwloc_topology_t topology;
#endif  /* defined(HAVE_HWLOC) */

#if defined(HAVE_HWLOC_PARENT_MEMBER)
#define HWLOC_GET_PARENT(OBJ)  (OBJ)->parent
#else
#define HWLOC_GET_PARENT(OBJ)  (OBJ)->father
#endif  /* defined(HAVE_HWLOC_PARENT_MEMBER) */

int dague_hwloc_init(void)
{
#if defined(HAVE_HWLOC)
    hwloc_topology_init(&topology);
    hwloc_topology_ignore_type_keep_structure(topology, HWLOC_OBJ_NODE);
    hwloc_topology_ignore_type_keep_structure(topology, HWLOC_OBJ_SOCKET);
    hwloc_topology_load(topology);
#endif  /* defined(HAVE_HWLOC) */
    return 0;
}

int dague_hwloc_fini(void)
{
#if defined(HAVE_HWLOC)
    hwloc_topology_destroy(topology);
#endif  /* defined(HAVE_HWLOC) */
    return 0;
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
    return 0;
}

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

    return -1;
}


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
    return 0;
}

int dague_hwloc_nb_levels(void)
{
#if defined(HAVE_HWLOC)
    return hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
#else
    return -1;
#endif  /* defined(HAVE_HWLOC) */
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
    return 0;
}

int dague_hwloc_nb_real_cores()
{
#if defined(HAVE_HWLOC)
    return hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
#else
    return -1;
#endif
}


int dague_hwloc_bind_on_core_index(int cpu)
{
#if defined(HAVE_HWLOC)
    hwloc_obj_t      obj;      /* Hwloc object    */
    hwloc_cpuset_t   cpuset;   /* HwLoc cpuset    */

    /* Get the core of index cpu */
    obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, cpu);
    if (!obj) {
        WARNING(("dague_hwloc: unable to get the core of index %i\n", cpu));
        return -1;
    }

    /* Get a copy of its cpuset that we may modify.  */
    /* Get only one logical processor (in case the core is SMT/hyperthreaded).  */
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
    }else{
#if defined(DAGUE_DEBUG_VERBOSE3) && defined(HAVE_HWLOC_BITMAP)
        {
            char *str = NULL;
            hwloc_bitmap_zero(cpuset);
            hwloc_get_cpubind(topology, cpuset, 0);
            hwloc_bitmap_asprintf(&str, cpuset);
            printf("thread bound on core %i\n", hwloc_bitmap_first(cpuset));
            free(str);
        }
#endif /* DAGUE_DEBUG_VERBOSE2 */
    }

    /* Get the number at Proc level ( We don't want to use HyperThreading ) */
    cpu = obj->children[0]->os_index;

    /* Free our cpuset copy */
#if !defined(HAVE_HWLOC_BITMAP)
    hwloc_cpuset_free(cpuset);
#else
    hwloc_bitmap_free(cpuset);
#endif
    return cpu;
#else
    return -1;
#endif
}


int dague_hwloc_bind_on_mask_index(hwloc_cpuset_t cpuset)
{
    // STEPH:: DO the appropriate modifications
#if defined(HAVE_HWLOC)
    if (hwloc_set_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD)) {
        char *str = NULL;
#if !defined(HAVE_HWLOC_BITMAP)
        hwloc_cpuset_asprintf(&str, cpuset);
#else
        hwloc_bitmap_asprintf(&str, cpuset);
#endif
        WARNING(("Couldn't bind to cpuset %s\n", str));
        free(str);
        return -1;
    }
    return hwloc_bitmap_first(cpuset);
#else
    return -1;
#endif /* defined(HAVE_HWLOC) */

}
