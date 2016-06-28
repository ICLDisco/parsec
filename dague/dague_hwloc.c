/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "dague/debug.h"
#include "dague/utils/output.h"

#include "dague/dague_hwloc.h"
#if defined(DAGUE_HAVE_HWLOC)
#include <hwloc.h>
#endif  /* defined(DAGUE_HAVE_HWLOC) */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#if defined(DAGUE_HAVE_HWLOC)
static hwloc_topology_t topology;
static int first_init = 1;
#endif  /* defined(DAGUE_HAVE_HWLOC) */
static int hyperth_per_core = 1;

#if defined(DAGUE_HAVE_HWLOC_PARENT_MEMBER)
#define HWLOC_GET_PARENT(OBJ)  (OBJ)->parent
#else
#define HWLOC_GET_PARENT(OBJ)  (OBJ)->father
#endif  /* defined(DAGUE_HAVE_HWLOC_PARENT_MEMBER) */

#define MAX(x, y) ( (x)>(y)?(x):(y) )

#if defined(DAGUE_HAVE_HWLOC)
#if defined(DAGUE_HAVE_HWLOC_BITMAP)
#define HWLOC_ASPRINTF hwloc_bitmap_asprintf
#define HWLOC_ISSET    hwloc_bitmap_isset
#define HWLOC_FIRST    hwloc_bitmap_first
#define HWLOC_WEIGHT   hwloc_bitmap_weight
#define HWLOC_ALLOC    hwloc_bitmap_alloc
#define HWLOC_DUP      hwloc_bitmap_dup
#define HWLOC_SINGLIFY hwloc_bitmap_singlify
#define HWLOC_FREE     hwloc_bitmap_free
#else
#define HWLOC_ASPRINTF hwloc_cpuset_asprintf
#define HWLOC_ISSET    hwloc_cpuset_isset
#define HWLOC_FIRST    hwloc_cpuset_first
#define HWLOC_WEIGHT   hwloc_cpuset_weight
#define HWLOC_ALLOC    hwloc_cpuset_alloc
#define HWLOC_DUP      hwloc_cpuset_dup
#define HWLOC_SINGLIFY hwloc_cpuset_singlify
#define HWLOC_FREE     hwloc_cpuset_free
#endif  /* defined(DAGUE_HAVE_HWLOC_BITMAP) */
#endif  /* defined(DAGUE_HAVE_HWLOC) */

/**
 * Print the cpuset as a string prefaced with the provided message.
 */
static void dague_hwloc_print_cpuset(int verb, char* msg, hwloc_cpuset_t cpuset)
{
#if defined(DAGUE_HAVE_HWLOC)
    char *str = NULL;
    HWLOC_ASPRINTF(&str, cpuset);
    if( 1 == verb ) dague_warning("%s %s", msg, str);
    else if( 2 == verb ) dague_inform("%s %s", msg, str);
    else dague_debug_verbose(verb, dague_debug_output, "%s %s", msg, str);
    free(str);
#else
    (void)cpuset;
    dague_debug_verbose(3, dague_debug_output, "%s compiled without HWLOC support", msg);
#endif  /* defined(DAGUE_HAVE_HWLOC) */
}

int dague_hwloc_init(void)
{
#if defined(DAGUE_HAVE_HWLOC)
    if ( first_init ) {
        hwloc_topology_init(&topology);
        hwloc_topology_load(topology);
        first_init = 0;
    }
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    return 0;
}

int dague_hwloc_fini(void)
{
#if defined(DAGUE_HAVE_HWLOC)
    hwloc_topology_destroy(topology);
    first_init = 1;
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    return 0;
}

int dague_hwloc_export_topology(int *buflen, char **xmlbuffer)
{
#if defined(DAGUE_HAVE_HWLOC)
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

#if defined(DAGUE_HAVE_HWLOC)
    if( first_init == 0 ) {
        hwloc_free_xmlbuffer(topology, xmlbuffer);
    }
#endif
}

int dague_hwloc_distance( int id1, int id2 )
{
#if defined(DAGUE_HAVE_HWLOC)
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
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    (void)id1;(void)id2;
    return 0;
}

/**
 *
 */
int dague_hwloc_master_id( int level, int processor_id )
{
#if defined(DAGUE_HAVE_HWLOC)
    unsigned int i;
    int ncores;

    ncores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);

    /* If we are using hyper-threads */
    processor_id = processor_id % ncores;

    for(i = 0; i < hwloc_get_nbobjs_by_depth(topology, level); i++) {
        hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, level, i);

        if(HWLOC_ISSET(obj->cpuset, processor_id)) {
            return HWLOC_FIRST(obj->cpuset);
        }
    }
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    (void)level; (void)processor_id;
    return -1;
}

/**
 *
 */
unsigned int dague_hwloc_nb_cores( int level, int master_id )
{
#if defined(DAGUE_HAVE_HWLOC)
    unsigned int i;

    for(i = 0; i < hwloc_get_nbobjs_by_depth(topology, level); i++){
        hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, level, i);
        if(HWLOC_ISSET(obj->cpuset, master_id)){
            return HWLOC_WEIGHT(obj->cpuset);
        }
    }
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    (void)level; (void)master_id;
    return 0;
}


size_t dague_hwloc_cache_size( unsigned int level, int master_id )
{
#if defined(DAGUE_HAVE_HWLOC)
#if defined(DAGUE_HAVE_HWLOC_OBJ_PU) || 1
    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, master_id);
#else
    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PROC, master_id);
#endif  /* defined(DAGUE_HAVE_HWLOC_OBJ_PU) */

    while (obj) {
        if(obj->depth == level){
            if(obj->type == HWLOC_OBJ_CACHE){
#if defined(DAGUE_HAVE_HWLOC_CACHE_ATTR)
                return obj->attr->cache.size;
#else
                return obj->attr->cache.memory_kB;
#endif  /* defined(DAGUE_HAVE_HWLOC_CACHE_ATTR) */
            }
            return 0;
        }
        obj = HWLOC_GET_PARENT(obj);
    }
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    (void)level; (void)master_id;
    return 0;
}

int dague_hwloc_nb_real_cores(void)
{
#if defined(DAGUE_HAVE_HWLOC)
    return hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
#else
    int nb_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if(nb_cores == -1) {
        perror("sysconf(_SC_NPROCESSORS_ONLN). Expect at least one.\n");
        nb_cores = 1;
    }
    return nb_cores;
#endif
}


int dague_hwloc_core_first_hrwd_ancestor_depth(void)
{
#if defined(DAGUE_HAVE_HWLOC)
    int level = MAX( hwloc_get_type_depth(topology, HWLOC_OBJ_NODE),
                     hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET) );
    assert(level < hwloc_get_type_depth(topology, HWLOC_OBJ_CORE));
    return level;
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    return -1;
}

int dague_hwloc_get_nb_objects(int level)
{
#if defined(DAGUE_HAVE_HWLOC)
    return hwloc_get_nbobjs_by_depth(topology, level);
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    (void)level;
    return -1;
}


int dague_hwloc_socket_id(int core_id )
{
#if defined(DAGUE_HAVE_HWLOC)
    hwloc_obj_t core =  hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, core_id);
    hwloc_obj_t socket = NULL;
    if( NULL == core ) return -1;  /* protect against NULL objects */
    if( NULL != (socket = hwloc_get_ancestor_obj_by_type(topology,
                                                         HWLOC_OBJ_SOCKET, core)) ) {
        return socket->logical_index;
    }
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    (void)core_id;
    return -1;
}

int dague_hwloc_numa_id(int core_id )
{
#if defined(DAGUE_HAVE_HWLOC)
    hwloc_obj_t core =  hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, core_id);
    hwloc_obj_t node = NULL;
    if( NULL == core ) return -1;  /* protect against NULL objects */
    if( NULL != (node = hwloc_get_ancestor_obj_by_type(topology , HWLOC_OBJ_NODE, core)) ) {
        return node->logical_index;
    }
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    (void)core_id;
    return -1;
}

unsigned int dague_hwloc_nb_cores_per_obj( int level, int index )
{
#if defined(DAGUE_HAVE_HWLOC)
    hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, level, index);
    assert( obj != NULL );
    return hwloc_get_nbobjs_inside_cpuset_by_type(topology, obj->cpuset, HWLOC_OBJ_CORE);
#else
    (void)level; (void)index;
    return -1;
#endif  /* defined(DAGUE_HAVE_HWLOC) */
}

int dague_hwloc_nb_levels(void)
{
#if defined(DAGUE_HAVE_HWLOC)
    return hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
#else
    return -1;
#endif  /* defined(DAGUE_HAVE_HWLOC) */
}

char *dague_hwloc_get_binding(void)
{
#if defined(DAGUE_HAVE_HWLOC)
    char *binding;
    hwloc_cpuset_t cpuset;

    cpuset = HWLOC_ALLOC();
    HWLOC_SINGLIFY(cpuset);

    /** No need to check for return code: the set will be unchanged (0x0)
     *  if get_cpubind fails */
    hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD);

    HWLOC_ASPRINTF(&binding, cpuset);
    HWLOC_FREE(cpuset);
    return binding;
#endif
    return strdup("No_Binding_Information");
}

int dague_hwloc_bind_on_core_index(int cpu_index, int local_ht_index)
{
#if !defined(DAGUE_HAVE_HWLOC)
    (void)cpu_index; (void)local_ht_index;
    return -1;
#else
    hwloc_obj_t      obj, core;      /* HWLOC object */
    hwloc_cpuset_t   cpuset;         /* HWLOC cpuset */

    /* Get the core of index cpu_index */
    obj = core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, cpu_index);
    if (!core) {
        dague_warning("dague_hwloc: unable to get the core of index %i (nb physical cores = %i )",
                 cpu_index,  dague_hwloc_nb_real_cores());
        return -1;
    }
    /* Get the cpuset of the core if not using SMT/HyperThreading,
     * get the cpuset of the designated child object (PU) otherwise */
    if( local_ht_index > -1) {
        obj = core->children[local_ht_index % core->arity];
        if(!obj) {
            dague_warning("dague_hwloc: unable to get the core of index %i, HT %i (nb cores = %i)",
                     cpu_index, local_ht_index, dague_hwloc_nb_real_cores());
            return -1;
        }
    }

    /* Get a copy of its cpuset that we may modify.  */
    cpuset = HWLOC_DUP(obj->cpuset);
    HWLOC_SINGLIFY(cpuset);

    /* And try to bind ourself there.  */
    if (hwloc_set_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD)) {
        dague_hwloc_print_cpuset(1, "dague_hwloc: couldn't bind to cpuset", obj->cpuset );
        cpu_index = -1;
        goto free_and_return;
    }
    dague_debug_verbose(3, dague_debug_output, "Thread bound on core index %i, [HT %i ]", cpu_index, local_ht_index);

    /* Get the number at Proc level*/
    cpu_index = obj->os_index;

  free_and_return:
    /* Free our cpuset copy */
    HWLOC_FREE(cpuset);
    return cpu_index;
#endif  /* !defined(DAGUE_HAVE_HWLOC) */
}

int dague_hwloc_bind_on_mask_index(hwloc_cpuset_t cpuset)
{
#if defined(DAGUE_HAVE_HWLOC) && defined(DAGUE_HAVE_HWLOC_BITMAP)
    unsigned cpu_index;
    int first_free;
    hwloc_obj_t obj;
    hwloc_cpuset_t binding_mask = hwloc_bitmap_alloc();

    /* For each index in the mask, get the associated cpu object and use its cpuset to add it to the binding mask */
    hwloc_bitmap_foreach_begin(cpu_index, cpuset) {
        /* Get the core of index cpu */
        obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, cpu_index);
        if (!obj) {
            DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "dague_hwloc_bind_on_mask_index: unable to get the core of index %i", cpu_index);
        } else {
            hwloc_bitmap_or(binding_mask, binding_mask, obj->cpuset);
        }
    } hwloc_bitmap_foreach_end();

    if (hwloc_set_cpubind(topology, binding_mask, HWLOC_CPUBIND_THREAD)) {
        dague_hwloc_print_cpuset(1, "Couldn't bind to cpuset ", binding_mask);
        return -1;
    }

    dague_hwloc_print_cpuset(9, "Thread binding: cpuset binding [BEFORE]: ", cpuset);
    dague_hwloc_print_cpuset(4, "Thread binding: cpuset binding [ACTUAL]: ", binding_mask);

    first_free = hwloc_bitmap_first(binding_mask);
    hwloc_bitmap_free(binding_mask);
    return first_free;
#else
    (void) cpuset;
    return -1;
#endif /* DAGUE_HAVE_HWLOC && DAGUE_HAVE_HWLOC_BITMAP */
}

/*
 * Define the number of hyper-threads accepted per core.
 */
int dague_hwloc_allow_ht(int htnb)
{
    assert( htnb > 0 );

#if defined(DAGUE_HAVE_HWLOC) && defined(DAGUE_HAVE_HWLOC_BITMAP)
    /* Check the validity of the parameter. Correct otherwise  */
    if (htnb > 1) {
        int pu_per_core = (hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU) /
                           hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE));
        if( htnb > pu_per_core){
            dague_warning("HyperThreading:: There not enought logical processors to consider %i HyperThreads per core (set up to %i)", htnb,  pu_per_core);
            htnb = pu_per_core;
        }
    }
#endif
    /* Without hwloc, trust your user to give a correct parameter */
    hyperth_per_core = htnb;
    return hyperth_per_core;
}

int dague_hwloc_get_ht(void)
{
    return hyperth_per_core;
}

