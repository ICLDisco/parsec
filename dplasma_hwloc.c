/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "dplasma.h"

#if defined(HAVE_HWLOC)
#include <hwloc.h>
#endif  /* defined(HAVE_HWLOC) */
#include <stdio.h>
#include <stdlib.h>
 
#if defined(HAVE_HWLOC)
static hwloc_topology_t topology;
#endif  /* defined(HAVE_HWLOC) */

int dplasma_hwloc_init(void)
{
#if defined(HAVE_HWLOC)
    hwloc_topology_init(&topology);
    hwloc_topology_ignore_type_keep_structure(topology, HWLOC_OBJ_NODE);
    hwloc_topology_ignore_type_keep_structure(topology, HWLOC_OBJ_SOCKET);
    hwloc_topology_load(topology);
#endif  /* defined(HAVE_HWLOC) */
    return 0;
}

int dplasma_hwloc_fini(void)
{
#if defined(HAVE_HWLOC)
    hwloc_topology_destroy(topology);
#endif  /* defined(HAVE_HWLOC) */
    return 0;
}

int dplasma_hwloc_distance( int id1, int id2 )
{
#if defined(HAVE_HWLOC)
	int count = 0;

	hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, id1);
	hwloc_obj_t obj2 = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, id2);
	 
	while( obj && obj2) {
		if(obj == obj2 ) {
            return count*2;
        }
		obj = obj->father;
		obj2 = obj2->father;
        count++;
	}
#endif  /* defined(HAVE_HWLOC) */
    return 0;
}

int dplasma_hwloc_master_id( int level, int processor_id )
{
#if defined(HAVE_HWLOC)
	int count = 0, i, div = 0, real_cores, cores, test = 0;
	        
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
					            
		if(hwloc_cpuset_isset(obj->cpuset, processor_id)) {
			return hwloc_cpuset_first(obj->cpuset);
        }
	}
#endif  /* defined(HAVE_HWLOC) */

	return -1;
}    
 
unsigned int dplasma_hwloc_nb_cores( int level, int master_id )
{
#if defined(HAVE_HWLOC)
	int i;
	     
	for(i = 0; i < hwloc_get_nbobjs_by_depth(topology, level); i++){	 
		
        hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, level, i);
				 
        if(hwloc_cpuset_isset(obj->cpuset, master_id)){
            return hwloc_cpuset_weight(obj->cpuset);
        }
	}
#endif  /* defined(HAVE_HWLOC) */

	return 0;
}
 
 
int dplasma_hwloc_nb_levels(void)
{
#if defined(HAVE_HWLOC)
    return hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);    
#endif  /* defined(HAVE_HWLOC) */
}
 
    
size_t dplasma_hwloc_cache_size( int level, int master_id )
{	    
#if defined(HAVE_HWLOC)
	hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PROC, master_id);
	     
	while (obj) {
		if(obj->depth == level){
            if(obj->type == HWLOC_OBJ_CACHE){
	        	return obj->attr->cache.memory_kB;
            }
            return 0;
        }
        obj = obj->father;
	}
#endif  /* defined(HAVE_HWLOC) */
	 
	return 0;
}
