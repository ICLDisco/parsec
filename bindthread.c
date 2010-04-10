#define _GNU_SOURCE
#include <pthread.h>
#include "dplasma_config.h"

#ifdef DPLASMA_BIND_HWLOC
#  include <hwloc.h>
#else
#  if (defined ARCH_COMPAQ)
#    include <sys/types.h>
#    include <sys/resource.h>
#    include <sys/processor.h> 
#    include <sys/sysinfo.h>
#    include <machine/hal_sysinfo.h>
#    define X_INCLUDE_CXML
#  elif (defined ARCH_X86) || (defined ARCH_X86_64) || (defined ARCH_MAC_OS_X)
#    include <sched.h>
#    ifdef ARCH_MAC_OS_X
#      include <mach/thread_policy.h>
#    endif
#  endif
#endif

int dplasma_bindthread(int cpu)
{
#ifdef MARCEL

  {
    marcel_vpset_t vpset = MARCEL_VPSET_ZERO;
    marcel_vpset_vp(&vpset, cpu);
    marcel_apply_vpset(&vpset); 
  }

#elif (defined DPLASMA_BIND_HWLOC)
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
   if (!obj)
     return 0;
   
   /* Get a copy of its cpuset that we may modify.  */
   cpuset = hwloc_cpuset_dup(obj->cpuset);
    
   /* Get only one logical processor (in case the core is SMT/hyperthreaded).  */
   hwloc_cpuset_singlify(cpuset);
    
   printf("Try to bind sur le cpu %d\n", cpu);
   /* And try to bind ourself there.  */
   if (hwloc_set_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD)) {
     char *str = NULL;
     hwloc_cpuset_asprintf(&str, obj->cpuset);
     printf("Couldn't bind to cpuset %s\n", str);
     free(str);
   }
    
   /* Get the number at Proc level ( We don't want to use HyperThreading ) */
   cpu = obj->children[0]->os_index;
    
   /* Free our cpuset copy */
   hwloc_cpuset_free(cpuset);
    
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
#elif (defined ARCH_X86) || (defined ARCH_X86_64)
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
	 perror("sched_setaffinity");
	 EXIT(MOD_SOPALIN, INTERNAL_ERR);
       }
 }
#elif (defined ARCH_MAC_OS_X)
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
   if(ret != 0)
     {
       perror("thread_policy_set");
       EXIT(MOD_SOPALIN, INTERNAL_ERR);
     }
 }
#endif /* Acrhitectures */
#endif /* WITH_HWLOC     */

 return cpu;
}
