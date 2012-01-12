#include "dague_config.h"
#include "bindthread.h"
#if defined(HAVE_HWLOC)
#  include <hwloc.h>
#elif defined(ARCH_COMPAQ)
#  include <sys/types.h>
#  include <sys/resource.h>
#  include <sys/processor.h> 
#  include <sys/sysinfo.h>
#  include <machine/hal_sysinfo.h>
#  define X_INCLUDE_CXML
#elif defined(HAVE_SCHED_SETAFFINITY)
#  include <linux/unistd.h>
#  include <sched.h>
#elif defined(MAC_OS_X)
#  include <mach/mach_init.h>
#  include <mach/thread_policy.h>
/**
 * Expose the hidden kernel interface.
 */
extern kern_return_t thread_policy_set( thread_t               thread,
                                        thread_policy_flavor_t flavor,
                                        thread_policy_t        policy_info,
                                        mach_msg_type_number_t count);
#endif  /* define(HAVE_HWLOC) */

int dague_bindthread(int cpu)
{
#ifdef MARCEL
    {
        marcel_vpset_t vpset = MARCEL_VPSET_ZERO;
        marcel_vpset_vp(&vpset, cpu);
        marcel_apply_vpset(&vpset); 
    }
    
#elif defined(HAVE_HWLOC)
    {
        hwloc_topology_t topology; /* Topology object */
        hwloc_obj_t      obj;      /* Hwloc object    */ 
        hwloc_cpuset_t   cpuset;   /* HwLoc cpuset    */
        
        /* Allocate and initialize topology object.  */
        hwloc_topology_init(&topology);
        
        /* Perform the topology detection.  */
        hwloc_topology_load(topology);
        
        /* Get last one.  */
        obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, cpu);
        if (!obj) {
            /* Destroy topology object.  */
            hwloc_topology_destroy(topology);  
            
            return 0;
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
            printf("Couldn't bind to cpuset %s\n", str);
            free(str);

            /* Free our cpuset copy */
#if !defined(HAVE_HWLOC_BITMAP)
            hwloc_cpuset_free(cpuset);
#else
            hwloc_bitmap_free(cpuset);
#endif
            /* Destroy topology object.  */
            hwloc_topology_destroy(topology);  
            return -1;
        }
        
        /* Get the number at Proc level ( We don't want to use HyperThreading ) */
        cpu = obj->children[0]->os_index;
        
        /* Free our cpuset copy */
#if !defined(HAVE_HWLOC_BITMAP)
        hwloc_cpuset_free(cpuset);
#else
        hwloc_bitmap_free(cpuset);
#endif
        /* Destroy topology object.  */
        hwloc_topology_destroy(topology);  
    }
#else /* We bind thread ourself in funtion of architecture */
    
#ifdef ARCH_PPC
    {
        tid_t self_ktid = thread_self ();
        bindprocessor(BINDTHREAD, self_ktid, cpu*2);
    }
#elif (defined ARCH_COMPAQ)
    {
        bind_to_cpu_id(getpid(), cpu, 0);
    }
#elif defined(HAVE_SCHED_SETAFFINITY)
    {  
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        
#ifdef HAVE_OLD_SCHED_SETAFFINITY
        if(sched_setaffinity(0,&mask) < 0)
#else /* HAVE_OLD_SCHED_SETAFFINITY */
        if(sched_setaffinity(0,sizeof(mask),&mask) < 0)
#endif /* HAVE_OLD_SCHED_SETAFFINITY */
            {
                return -1;
            }
    }
#elif (defined MAC_OS_X)
    {
        thread_affinity_policy_data_t ap;
        int                           ret;
        
        ap.affinity_tag = 1; /* non-null affinity tag */
        ret = thread_policy_set(
                                mach_thread_self(),
                                THREAD_AFFINITY_POLICY,
                                (integer_t*) &ap,
                                THREAD_AFFINITY_POLICY_COUNT
                                );
        if(ret != 0) {
            return -1;
        }
    }
#endif /* Architectures */
#endif /* WITH_HWLOC     */
    
    return cpu;
}


#if defined(HAVE_HWLOC)
int dague_bindthread_mask(hwloc_cpuset_t cpuset)
{
    hwloc_topology_t topology; /* Topology object */

    /* Allocate and initialize topology object.  */
    hwloc_topology_init(&topology);

    /* Perform the topology detection.  */
    hwloc_topology_load(topology);

    if (hwloc_set_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD)) {
	char *str = NULL;
#if !defined(HAVE_HWLOC_BITMAP)
	hwloc_cpuset_asprintf(&str, cpuset);
#else
	hwloc_bitmap_asprintf(&str, cpuset);
#endif
	printf("Couldn't bind to cpuset %s\n", str);
	free(str);

        /* Destroy topology object.  */
	hwloc_topology_destroy(topology);
	return -1;
    }

    /* Destroy topology object.  */
    hwloc_topology_destroy(topology);
    return 0;
}
#endif
