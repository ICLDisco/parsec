/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
/*  unset options that make debug.h unpure, we need bindthread to compile standalone for unit tests */
#undef PARSEC_HAVE_MPI
#undef PARSEC_DEBUG_HISTORY
#include "parsec/debug.h"
#include "parsec/bindthread.h"
#if defined(PARSEC_HAVE_HWLOC)
#include "parsec/parsec_hwloc.h"
#elif defined(ARCH_COMPAQ)
#  include <sys/types.h>
#  include <sys/resource.h>
#  include <sys/processor.h>
#  include <sys/sysinfo.h>
#  include <machine/hal_sysinfo.h>
#  define X_INCLUDE_CXML
#elif defined(PARSEC_HAVE_SCHED_SETAFFINITY)
#  include <linux/unistd.h>
#  include <sched.h>
#elif defined(PARSEC_OSX)
#  include <mach/mach_init.h>
#  include <mach/thread_policy.h>
/**
 * Expose the hidden kernel interface.
 */
extern kern_return_t thread_policy_set( thread_t               thread,
                                        thread_policy_flavor_t flavor,
                                        thread_policy_t        policy_info,
                                        mach_msg_type_number_t count);
#endif  /* define(PARSEC_HAVE_HWLOC) */

int parsec_bindthread(int cpu, int ht)
{
    if( -1 == cpu )  /* Don't try binding if not required */
        return -1;
#ifdef MARCEL
    {
        marcel_vpset_t vpset = MARCEL_VPSET_ZERO;
        marcel_vpset_vp(&vpset, cpu);
        marcel_apply_vpset(&vpset);
    }
#elif defined(PARSEC_HAVE_HWLOC)
    {
        cpu = parsec_hwloc_bind_on_core_index(cpu, ht);
        if(cpu == -1 ) {
            parsec_warning("Core binding on node %i failed", cpu);
            return -1;
        }
    }
#else /* We bind thread ourself in funtion of architecture */

#if defined(PARSEC_HAVE_SCHED_SETAFFINITY)
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);

#ifdef PARSEC_HAVE_OLD_SCHED_SETAFFINITY
        if(sched_setaffinity(0,&mask) < 0)
#else /* PARSEC_HAVE_OLD_SCHED_SETAFFINITY */
        if(sched_setaffinity(0,sizeof(mask),&mask) < 0)
#endif /* PARSEC_HAVE_OLD_SCHED_SETAFFINITY */
            {
                return -1;
            }
    }
#elif defined(PARSEC_ARCH_PPC)
    {
        tid_t self_ktid = thread_self ();
        bindprocessor(BINDTHREAD, self_ktid, cpu*2);
    }
#elif (defined ARCH_COMPAQ)
    {
        bind_to_cpu_id(getpid(), cpu, 0);
    }
#elif (defined PARSEC_OSX)
    {
        thread_affinity_policy_data_t ap;
        int                           ret;

        ap.affinity_tag = 1; /* non-null affinity tag */
        ret = thread_policy_set( mach_thread_self(),
                                 THREAD_AFFINITY_POLICY,
                                 (integer_t*) &ap,
                                 THREAD_AFFINITY_POLICY_COUNT );
        if(ret != 0) {
            return -1;
        }
    }
#endif /* Architectures */

#endif /* !defined(PARSEC_HAVE_HWLOC) */
    (void)ht;
    return cpu;
}


#if defined(PARSEC_HAVE_HWLOC)
int parsec_bindthread_mask(hwloc_cpuset_t cpuset)
{
    return parsec_hwloc_bind_on_mask_index(cpuset);
}
#endif

