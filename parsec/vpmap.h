/**
 * Copyright (c) 2009-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _VPMAP_H_
#define _VPMAP_H_

/**
 *  @defgroup parsec_internal_virtualprocess Virtual Processes
 *  @ingroup parsec_internal
 *    Virtual Processes allow to isolate groups of threads and avoid
 *    work stealing between threads belonging to different virtual
 *    processes.
 *  @addtogroup parsec_internal_virtualprocess
 *  @{
 */

#include <stdio.h> /* for FILE* */

BEGIN_C_DECLS

#include "parsec/parsec_hwloc.h"

/**
 * Initialize the VP amp based on the provided argument.
 */
int parsec_vpmap_init(char* optarg, int nb_cores);

/**
 * vpmap_fini
 * to be called when finalizing the virtual process map
 */
void parsec_vpmap_fini(void);

/* VPMAP_GET_*
 *   return information on the (local) virtual process map
 */

/**
 * @return the number of virtual processes on the local process
 */
int parsec_vpmap_get_nb_vp(void);

/**
 * @param vp: identifier of the virtual process
 *        0 <= vp < parsec_vpmap_get_nb_vp()
 * @return the number of threads in this VP, or -1 if vp is not correct
 */
extern int parsec_vpmap_get_vp_threads(int vp);

/**
 * @param vp: identifier of the virtual process
 *        0 <= vp < parsec_vpmap_get_nb_vp()
 * @param thread: identifier of the thread
 *        0 <= thread < vpmap_get_nb_threads_in_vp(vp)
 * @return the number of cores on which this thread of this vp can be bound
 *         -1 if vp or thread is not compatible.
 */
extern int parsec_vpmap_get_vp_thread_cores(int vp, int thread);

/**
 * @param vp: identifier of the virtual process
 *        0 <= vp < parsec_vpmap_get_nb_vp()
 * @param thread: identifier of the thread
 *        0 <= thread < vpmap_get_nb_threads_in_vp(vp)
 * @param cores: output array of cores identifier on which the thread
 *               thread of the virtual process vp can be bound.
 *               sizeof(cores) = sizeof(int) * vpmap_get_nb_cores_affinity(vp, thread)
 *        cores[*] is undefined if vp or thread is not compatible
 */
extern hwloc_cpuset_t parsec_vpmap_get_vp_thread_affinity(int vp, int thread, int *ht);

/**
 * Helping function: displays the virtual process map 
 */
void parsec_vpmap_display_map(void);

END_C_DECLS

/** @} */

#endif
