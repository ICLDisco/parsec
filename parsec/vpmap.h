/**
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _VPMAP_H_
#define _VPMAP_H_

#include <stdio.h> /* for FILE* */

BEGIN_C_DECLS
/**
 * vpmap_fini
 * to be called when finalizing the virtual process map
 */
void vpmap_fini(void);

/* VPMAP_INIT_*
 *  Different ways of initializing the vpmap.
 * Forall XXX, YYY,
 *     vpmap_init_XXX cannot be called after a succesful call to vpmap_init_YYY
 */

/**
 * Initialize the vpmap based on the HWLOC hardware locality information. Do not
 * initialize more than the expected number of cores.
 *   Create one thread per core
 *   Create one vp per socket
 *   Bind threads of the same vp on the different cores of the
 *     corresponding socket
 *   Uses hwloc
 * @return 0 if success; -1 if the initialization was not possible.
 */
int vpmap_init_from_hardware_affinity(int nbcores);

/**
 * initialize the vpmap using a simple nbvp x nbthreadspervp
 *   approach; and a round-robin distribution of threads among cores.
 */
int vpmap_init_from_parameters(int nbvp, int nbthreadspervp, int nbcores);

/**
 * initialize the vpmap using a very simple flat x nbcores approach
 */
int vpmap_init_from_flat(int nbcores);

/**
 * initialize the vpmap using a virtual process rank file
 *  Format of the rankfile:
 *  list of integers: cores of thread 0 of vp 0
 *  list of integers: cores of thread 1 of vp 0
 *  ...
 *  blank line: change the vp number
 *  list of integers: cores of thread 0 of vp 1
 *  ...
 */
int vpmap_init_from_file(const char *filename);

/* VPMAP_GET_*
 *   return information on the (local) virtual process map
 */

/**
 * @return the number of virtual processes on the local process
 */
int vpmap_get_nb_vp(void);

/**
 * @param vp: identifier of the virtual process
 *        0 <= vp < vpmap_get_nb_vp()
 * @return the number of threads in this VP, or -1 if vp is not correct
 */
typedef int (*vpmap_get_nb_threads_in_vp_t)(int vp);
extern vpmap_get_nb_threads_in_vp_t vpmap_get_nb_threads_in_vp;

/**
 * @param vp: identifier of the virtual process
 *        0 <= vp < vpmap_get_nb_vp()
 * @param thread: identifier of the thread
 *        0 <= thread < vpmap_get_nb_threads_in_vp(vp)
 * @return the number of cores on which this thread of this vp can be bound
 *         -1 if vp or thread is not compatible.
 */
typedef int (*vpmap_get_nb_cores_affinity_t)(int vp, int thread);
extern vpmap_get_nb_cores_affinity_t vpmap_get_nb_cores_affinity;

/**
 * @param vp: identifier of the virtual process
 *        0 <= vp < vpmap_get_nb_vp()
 * @param thread: identifier of the thread
 *        0 <= thread < vpmap_get_nb_threads_in_vp(vp)
 * @param cores: output array of cores identifier on which the thread
 *               thread of the virtual process vp can be bound.
 *               sizeof(cores) = sizeof(int) * vpmap_get_nb_cores_affinity(vp, thread)
 *        cores[*] is undefined if vp or thread is not compatible
 */
typedef void (*vpmap_get_core_affinity_t)(int vp, int thread, int *cores, int *ht);
extern vpmap_get_core_affinity_t vpmap_get_core_affinity;

/**
 * Returns the number of threads on which the current vpmap spans
 *  (sum of the get_nb_thread_in_vp for all vp)
 */
int vpmap_get_nb_total_threads(void);

/**
 * Helping function: displays the virtual process map 
 */
void vpmap_display_map(void);

END_C_DECLS
#endif
