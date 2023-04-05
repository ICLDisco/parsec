/*
 * Copyright (c) 2010-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/output.h"

#include "parsec/parsec_hwloc.h"
#if defined(PARSEC_HAVE_HWLOC)
#include <hwloc.h>
#endif  /* defined(PARSEC_HAVE_HWLOC) */
#include <stdio.h>
#include <stdlib.h>
#if defined(PARSEC_HAVE_UNISTD_H)
#include <unistd.h>
#endif  /* defined(PARSEC_HAVE_UNISTD_H) */
#if defined(__WINDOWS__)
#include <windows.h>
#endif  /* defined(__WINDOWS__) */
#if defined(PARSEC_HAVE_HWLOC)
static hwloc_topology_t topology;
static int parsec_hwloc_first_init = 1;
#endif  /* defined(PARSEC_HAVE_HWLOC) */
static int hyperth_per_core = 1;
static int parsec_available_binding_resources = 1;

#if defined(PARSEC_HAVE_HWLOC_PARENT_MEMBER)
#define HWLOC_GET_PARENT(OBJ)  (OBJ)->parent
#else
#define HWLOC_GET_PARENT(OBJ)  (OBJ)->father
#endif  /* defined(PARSEC_HAVE_HWLOC_PARENT_MEMBER) */

#define MAX(x, y) ( (x)>(y)?(x):(y) )

/**
 * @brief The original cpuset as provide by the program manager. This cpuset will
 * not be altered (such that we always have the original bindings) but it will be
 * used as a base to reflect the additional restrictions from the PaRSEC binding
 * parameter, and become parsec_cpuset_restricted.
 */
hwloc_cpuset_t parsec_cpuset_original;
/**
 * @brief The set of resources we are allowed to bind onto. If the MCA parameter for
 * ignoring the binding is provided this cpuset will then contain all possible cores
 * on the node.
 */
hwloc_cpuset_t parsec_cpuset_restricted;

static hwloc_cpuset_t parsec_hwloc_cpuset_convert_to_system(hwloc_cpuset_t cpuset);

char* parsec_hwloc_convert_cpuset(int convert_to_system, hwloc_cpuset_t cpuset)
{
    char *str = NULL;

    if( convert_to_system ) {
        hwloc_cpuset_t binding_mask;
        binding_mask = parsec_hwloc_cpuset_convert_to_system(cpuset);
        HWLOC_ASPRINTF(&str, binding_mask);
        hwloc_bitmap_free(binding_mask);
    } else {
        HWLOC_ASPRINTF(&str, cpuset);
    }
    return str;
}

/**
 * Print the cpuset as a string prefaced with the provided message.
 */
static void parsec_hwloc_print_cpuset(int verb, int convert_to_system, char* msg, hwloc_cpuset_t cpuset)
{
#if defined(PARSEC_HAVE_HWLOC)
    char *str = NULL;

    str = parsec_hwloc_convert_cpuset(convert_to_system, cpuset);

    if( 1 == verb ) parsec_warning("%s %s", msg, str);
    else if( 2 == verb ) parsec_inform("%s %s", msg, str);
    else parsec_debug_verbose(verb, parsec_debug_output, "%s %s", msg, str);
    free(str);
#else
    (void)cpuset;(void)verb;
    parsec_debug_verbose(3, parsec_debug_output, "%s compiled without HWLOC support", msg);
#endif  /* defined(PARSEC_HAVE_HWLOC) */
}

int parsec_hwloc_init(void)
{
#if defined(PARSEC_HAVE_HWLOC)
    if ( !parsec_hwloc_first_init ) {
        return PARSEC_SUCCESS;
    }
#if HWLOC_API_VERSION >= 0x00020000
    /* headers are recent */
    if (hwloc_get_api_version() < 0x20000) {
        parsec_fatal("Compile headers and runtime hwloc libraries are not compatible (headers %x ; lib %x)", HWLOC_API_VERSION, hwloc_get_api_version());
    }
#else
    /* headers are pre-2.0 */
    if (hwloc_get_api_version() >= 0x20000) {
        parsec_fatal("Compile headers and runtime hwloc libraries are not compatible (headers %x ; lib %x)", HWLOC_API_VERSION, hwloc_get_api_version());
    }
#endif
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    int binding_unit = (parsec_runtime_allow_ht ? HWLOC_OBJ_PU : HWLOC_OBJ_CORE);

    parsec_cpuset_original = HWLOC_ALLOC();
    parsec_cpuset_restricted = HWLOC_ALLOC();
    /* save the original process binding */
    int rc = hwloc_get_cpubind(topology, parsec_cpuset_original, HWLOC_CPUBIND_PROCESS );
    if( 0 != rc ) {
        /* We are on a system without support for process/thread binding. */
        parsec_available_binding_resources = hwloc_get_nbobjs_by_type(topology, binding_unit);
        hwloc_bitmap_set_range(parsec_cpuset_original, 0, parsec_available_binding_resources-1);
        parsec_runtime_ignore_bindings = 1;  /* ignore all bindings provided by the user */
    }

    if( parsec_runtime_ignore_bindings ) {
        /* We are running unrestricted, so we need to build the cpuset that reflects this.
         * Keep in mind that this remains subject to the batch scheduler restrictions, the
         * loaded topology might not be the real hardware topology, but restricted to what
         * this process can access.
         */
        parsec_available_binding_resources = hwloc_get_nbobjs_by_type(topology, binding_unit);
        hwloc_bitmap_set_range(parsec_cpuset_restricted, 0, parsec_available_binding_resources-1);
    } else {
        hwloc_bitmap_copy(parsec_cpuset_restricted, parsec_cpuset_original);
        /** No need to check for return code: the set will be unchanged (0x0)
         *  if get_cpubind fails */
        hwloc_topology_dup(&parsec_hwloc_restricted_topology, parsec_hwloc_loaded_topology);
        hwloc_topology_restrict(parsec_hwloc_restricted_topology, parsec_cpuset_restricted, 0);
        parsec_available_binding_resources = hwloc_get_nbobjs_inside_cpuset_by_type(parsec_hwloc_restricted_topology,
                                                                                    parsec_cpuset_original, binding_unit);
    }

    if( 0 == parsec_available_binding_resources ) {  /* bozo case: something wrong let's just assume single core */
        parsec_warning("Could not identify the CPU sets to be used. Fall back on safety more, single core");
        parsec_available_binding_resources = 1;
        parsec_runtime_allow_ht = 1;
    }

    if( parsec_runtime_allow_ht > 1 ) {
        int hyperth_per_core = (hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU) /
                                hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE));
        if( parsec_runtime_allow_ht > hyperth_per_core) {
            parsec_warning("HyperThreading:: There not enough logical processors to consider %i HyperThreads "
                           "per core (set up to %i)", parsec_runtime_allow_ht,  hyperth_per_core);
            parsec_runtime_allow_ht = hyperth_per_core;
        }
    }

    parsec_hwloc_first_init = 0;
    return PARSEC_SUCCESS;
#else
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif  /* defined(PARSEC_HAVE_HWLOC) */
}

int parsec_hwloc_fini(void)
{
#if defined(PARSEC_HAVE_HWLOC)
    hwloc_topology_destroy(topology);
    parsec_hwloc_first_init = 1;
#endif  /* defined(PARSEC_HAVE_HWLOC) */
    return PARSEC_SUCCESS;
}

int parsec_hwloc_export_topology(int *buflen, char **xmlbuffer)
{
#if defined(PARSEC_HAVE_HWLOC)
    if( parsec_hwloc_first_init == 0 ) {
#if HWLOC_API_VERSION >= 0x20000
        return hwloc_topology_export_xmlbuffer(topology, xmlbuffer, buflen, 0 /*HWLOC_TOPOLOGY_EXPORT_XML_FLAG_V1*/);
#else
        return hwloc_topology_export_xmlbuffer(topology, xmlbuffer, buflen);
#endif
    }
#endif
    *buflen = 0;
    *xmlbuffer = NULL;
    return PARSEC_ERR_NOT_IMPLEMENTED;
}

void parsec_hwloc_free_xml_buffer(char *xmlbuffer)
{
    if( NULL == xmlbuffer )
        return;

#if defined(PARSEC_HAVE_HWLOC)
    if( parsec_hwloc_first_init == 0 ) {
        hwloc_free_xmlbuffer(topology, xmlbuffer);
    }
#endif
}

int parsec_hwloc_distance( int id1, int id2 )
{
#if defined(PARSEC_HAVE_HWLOC)
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
#endif  /* defined(PARSEC_HAVE_HWLOC) */
    (void)id1;(void)id2;
    return 0;
}

/**
 *
 */
int parsec_hwloc_master_id( int level, int processor_id )
{
#if defined(PARSEC_HAVE_HWLOC)
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
#endif  /* defined(PARSEC_HAVE_HWLOC) */
    (void)level; (void)processor_id;
    return PARSEC_ERR_NOT_IMPLEMENTED;
}

/**
 *
 */
unsigned int parsec_hwloc_nb_cores( int level, int master_id )
{
#if defined(PARSEC_HAVE_HWLOC)
    unsigned int i;

    for(i = 0; i < hwloc_get_nbobjs_by_depth(topology, level); i++){
        hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, level, i);
        if(HWLOC_ISSET(obj->cpuset, master_id)){
            return HWLOC_WEIGHT(obj->cpuset);
        }
    }
#endif  /* defined(PARSEC_HAVE_HWLOC) */
    (void)level; (void)master_id;
    return PARSEC_ERR_NOT_IMPLEMENTED;
}


size_t parsec_hwloc_cache_size( unsigned int level, int master_id )
{
#if defined(PARSEC_HAVE_HWLOC)
#if defined(PARSEC_HAVE_HWLOC_OBJ_PU) || 1
    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, master_id);
#else
    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PROC, master_id);
#endif  /* defined(PARSEC_HAVE_HWLOC_OBJ_PU) */

    while (obj) {
#if HWLOC_API_VERSION >= 0x00020000
        if((int)level == hwloc_get_type_depth(topology, obj->type)) {
            if(hwloc_obj_type_is_cache(obj->type)) {
#else
        if(obj->depth == level){
            if(obj->type == HWLOC_OBJ_CACHE){
#endif
#if defined(PARSEC_HAVE_HWLOC_CACHE_ATTR)
                return obj->attr->cache.size;
#else
                return obj->attr->cache.memory_kB;
#endif  /* defined(PARSEC_HAVE_HWLOC_CACHE_ATTR) */
            }
            return 0;
        }
        obj = HWLOC_GET_PARENT(obj);
    }
#endif  /* defined(PARSEC_HAVE_HWLOC) */
    (void)level; (void)master_id;
    return PARSEC_ERR_NOT_IMPLEMENTED;
}

int parsec_hwloc_nb_real_cores(void)
{
#if defined(PARSEC_HAVE_HWLOC)
    return parsec_available_binding_resources;
#elif defined(__WINDOWS__)
    SYSTEM_INFO systemInfo;
    GetSystemInfo(&systemInfo);
    return systemInfo.dwNumberOfProcessors;
#else
    int nb_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if(nb_cores == -1) {
        perror("sysconf(_SC_NPROCESSORS_ONLN). Expect at least one.\n");
        nb_cores = 1;
    }
    return nb_cores;
#endif
}


int parsec_hwloc_core_first_hrwd_ancestor_depth(void)
{
#if defined(PARSEC_HAVE_HWLOC)
    int level = MAX( hwloc_get_type_depth(topology, HWLOC_OBJ_NODE),
                     hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET) );
    assert(level < hwloc_get_type_depth(topology, HWLOC_OBJ_CORE));
    return level;
#else
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif  /* defined(PARSEC_HAVE_HWLOC) */
}

int parsec_hwloc_get_nb_objects(int level)
{
#if defined(PARSEC_HAVE_HWLOC)
    return hwloc_get_nbobjs_by_depth(topology, level);
#else
    (void)level;
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif  /* defined(PARSEC_HAVE_HWLOC) */
}


int parsec_hwloc_socket_id(int core_id )
{
#if defined(PARSEC_HAVE_HWLOC)
    hwloc_obj_t core =  hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, core_id);
    hwloc_obj_t socket = NULL;
    if( NULL == core ) return PARSEC_ERR_NOT_FOUND;  /* protect against NULL objects */
    if( NULL != (socket = hwloc_get_ancestor_obj_by_type(topology,
                                                         HWLOC_OBJ_SOCKET, core)) ) {
        return socket->logical_index;
    }
#else
    (void)core_id;
#endif  /* defined(PARSEC_HAVE_HWLOC) */
    return PARSEC_ERR_NOT_IMPLEMENTED;
}

int parsec_hwloc_numa_id(int core_id )
{
#if defined(PARSEC_HAVE_HWLOC)
    hwloc_obj_t core =  hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, core_id);
    hwloc_obj_t node = NULL;
    if( NULL == core ) return PARSEC_ERR_NOT_FOUND;  /* protect against NULL objects */
    if( NULL != (node = hwloc_get_ancestor_obj_by_type(topology , HWLOC_OBJ_NODE, core)) ) {
        return node->logical_index;
    }
#else
    (void)core_id;
#endif  /* defined(PARSEC_HAVE_HWLOC) */
    return PARSEC_ERR_NOT_IMPLEMENTED;
}

unsigned int parsec_hwloc_nb_cores_per_obj( int level, int index )
{
#if defined(PARSEC_HAVE_HWLOC)
    hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, level, index);
    if(NULL == obj) return PARSEC_ERR_NOT_FOUND;
    return hwloc_get_nbobjs_inside_cpuset_by_type(topology, obj->cpuset, HWLOC_OBJ_CORE);
#else
    (void)level; (void)index;
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif  /* defined(PARSEC_HAVE_HWLOC) */
}

hwloc_cpuset_t parsec_hwloc_cpuset_per_obj(int level, int index)
{
    hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, level, index);
    if(NULL == obj) return NULL;
    return HWLOC_DUP(obj->cpuset);
}

int parsec_hwloc_nb_levels(void)
{
#if defined(PARSEC_HAVE_HWLOC)
    return hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
#else
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif  /* defined(PARSEC_HAVE_HWLOC) */
}

char *parsec_hwloc_get_binding(hwloc_cpuset_t* cpuset, int flag)
{
#if defined(PARSEC_HAVE_HWLOC)
    char *binding;
    hwloc_cpuset_t stack_cpuset;

    if ((flag != HWLOC_CPUBIND_PROCESS) && (flag != HWLOC_CPUBIND_THREAD)) {
        return NULL;
    }
    if( NULL == cpuset ) {
        stack_cpuset = HWLOC_ALLOC();
        HWLOC_SINGLIFY(stack_cpuset);

        /** No need to check for return code: the set will be unchanged (0x0)
         *  if get_cpubind fails */
        hwloc_get_cpubind(topology, stack_cpuset, flag );
        HWLOC_ASPRINTF(&binding, stack_cpuset);
        HWLOC_FREE(stack_cpuset);
    } else {
        hwloc_get_cpubind(topology, *cpuset, flag );
        HWLOC_ASPRINTF(&binding, *cpuset);
    }
    return binding;
#else
    return NULL;
#endif
}

int parsec_hwloc_bind_on_core_index(int cpu_index, int local_ht_index)
{
#if !defined(PARSEC_HAVE_HWLOC)
    (void)cpu_index; (void)local_ht_index;
    return PARSEC_ERR_NOT_IMPLEMENTED;
#else
    hwloc_obj_t      obj, core;      /* HWLOC object */
    hwloc_cpuset_t   cpuset;         /* HWLOC cpuset */

    /* If we were not initialized first, let's initialize */
    if( parsec_hwloc_first_init == 1 ) {
        parsec_hwloc_init();
    }

    /* Get the core of index cpu_index */
    obj = core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, cpu_index);
    if (!core) {
        parsec_warning("parsec_hwloc: unable to get the core of index %i (nb physical cores = %i )",
                 cpu_index,  parsec_hwloc_nb_real_cores());
        return PARSEC_ERR_NOT_FOUND;
    }
    /* Get the cpuset of the core if not using SMT/HyperThreading,
     * get the cpuset of the designated child object (PU) otherwise */
    if( local_ht_index > -1) {
        obj = core->children[local_ht_index % core->arity];
        if(!obj) {
            parsec_warning("parsec_hwloc: unable to get the core of index %i, HT %i (nb cores = %i)",
                     cpu_index, local_ht_index, parsec_hwloc_nb_real_cores());
            return PARSEC_ERR_NOT_FOUND;
        }
    }

    /* Get a copy of its cpuset that we may modify.  */
    cpuset = HWLOC_DUP(obj->cpuset);
    HWLOC_SINGLIFY(cpuset);

    /* And try to bind ourself there.  */
    if (hwloc_set_cpubind(parsec_hwloc_loaded_topology, cpuset, HWLOC_CPUBIND_THREAD)) {
        parsec_hwloc_print_cpuset(1, 0, "parsec_hwloc: couldn't bind to cpuset", obj->cpuset );
        cpu_index = PARSEC_ERR_NOT_FOUND;
        goto free_and_return;
    }
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Thread bound on core index %i, [HT %i ]",
                        cpu_index, local_ht_index);

    /* Get the number at Proc level*/
    cpu_index = obj->os_index;

  free_and_return:
    /* Free our cpuset copy */
    HWLOC_FREE(cpuset);
    return cpu_index;
#endif  /* !defined(PARSEC_HAVE_HWLOC) */
}

static hwloc_cpuset_t parsec_hwloc_cpuset_convert_to_system(hwloc_cpuset_t cpuset)
{
    unsigned cpu_index;
    hwloc_obj_t obj;
    hwloc_cpuset_t binding_mask;

    /* If we were not initialized first, let's initialize */
    if( parsec_hwloc_first_init == 1 ) {
        parsec_hwloc_init();
    }

    binding_mask = hwloc_bitmap_alloc();

    /* For each index in the mask, get the associated cpu object and use its cpuset to add it to the binding mask */
    hwloc_bitmap_foreach_begin(cpu_index, cpuset) {
        /* Get the core of index cpu */
        obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, cpu_index);
        if (!obj) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "parsec_hwloc_bind_on_mask_index: unable to get the core of index %i", cpu_index);
        } else {
            hwloc_bitmap_or(binding_mask, binding_mask, obj->cpuset);
        }
    } hwloc_bitmap_foreach_end();

    return binding_mask;
}
int parsec_hwloc_bind_on_mask_index(hwloc_cpuset_t cpuset)
{
#if defined(PARSEC_HAVE_HWLOC) && defined(PARSEC_HAVE_HWLOC_BITMAP)
    int first_free;
    hwloc_cpuset_t binding_mask;

    /* If we were not initialized first, let's initialize */
    if( parsec_hwloc_first_init == 1 ) {
        parsec_hwloc_init();
    }

    binding_mask = parsec_hwloc_cpuset_convert_to_system(cpuset);

    parsec_hwloc_print_cpuset(9, 0, "Thread binding: cpuset binding [LOGICAL ]: ", cpuset);
    parsec_hwloc_print_cpuset(4, 0, "Thread binding: cpuset binding [PHYSICAL]: ", binding_mask);

    first_free = hwloc_bitmap_first(binding_mask);
    hwloc_bitmap_free(binding_mask);
    return first_free;
#else
    (void) cpuset;
    return PARSEC_ERR_NOT_IMPLEMENTED;
#endif /* PARSEC_HAVE_HWLOC && PARSEC_HAVE_HWLOC_BITMAP */
}


int parsec_hwloc_get_ht(void)
{
    return hyperth_per_core;
}

