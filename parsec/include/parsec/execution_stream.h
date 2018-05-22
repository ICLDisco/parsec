/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_EXECUTION_STREAM_H_HAS_BEEN_INCLUDED
#define PARSEC_EXECUTION_STREAM_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#ifdef PARSEC_HAVE_HWLOC
#include <hwloc.h>
#endif

#include <pthread.h>
#include <stdint.h>
#include "parsec/mempool.h"
#include "parsec/profiling.h"
#include "parsec/class/barrier.h"

#ifdef PINS_ENABLE
#include "parsec/mca/pins/pins.h"
#endif

#if defined(PARSEC_HAVE_GETRUSAGE) || !defined(__bgp__)
#include <sys/time.h>
#include <sys/resource.h>
#endif

BEGIN_C_DECLS

/**
 *  Computational Thread-specific structure
 */
struct parsec_execution_stream_s {
    int32_t   th_id;        /**< Internal thread identifier. A thread belongs to a vp */
    int core_id;            /**< Core on which the thread is bound (hwloc in order numbering) */
    int socket_id;          /**< Socket on which the thread is bound (hwloc in order numerotation) */

    pthread_t pthread_id;     /**< POSIX thread identifier. */

#if defined(PARSEC_PROF_TRACE)
    parsec_thread_profiling_t *es_profile;
#endif /* PARSEC_PROF_TRACE */

    void *scheduler_object;

#if defined(PARSEC_SIM)
    int largest_simulation_date;
#endif

#if defined(PINS_ENABLE)
    struct parsec_pins_next_callback_s pins_events_cb[PINS_FLAG_COUNT];
#endif  /* defined(PINS_ENABLE) */

#if defined(PARSEC_PROF_RUSAGE_EU)
#if defined(PARSEC_HAVE_GETRUSAGE) && !defined(__bgp__)
    struct rusage _es_rusage;
#endif /* PARSEC_HAVE_GETRUSAGE */
#endif

    struct parsec_vp_s      *virtual_process;   /**< Backlink to the virtual process that holds this thread */
    parsec_thread_mempool_t *context_mempool;   /**< When allocating new execution contexts, this mempool is used */
    parsec_thread_mempool_t *datarepo_mempools[MAX_PARAM_COUNT+1]; /**< When allocating new data repositories,
                                                                    *   we use these mempools */
    parsec_thread_mempool_t *dependencies_mempool; /**< If using hashtables to store dependencies
                                                    *   those are allocated using this mempool */
};

/**
 * Threads are grouped per virtual process
 */
struct parsec_vp_s {
    parsec_context_t *parsec_context; /**< backlink to the global context */
    int32_t vp_id;                  /**< virtual process identifier of this vp */
    int32_t nb_cores;               /**< number of cores for this vp */

    /* Mempools are allocated per VP, and used per execution_stream
     * The last eu of this VP will create the mempools for all eus of this VP
     * and each eu will point into the corresponding element
     */
    parsec_mempool_t         context_mempool;   /**< When allocating new execution contexts, this mempool is used */
    parsec_mempool_t         datarepo_mempools[MAX_PARAM_COUNT+1]; /**< When allocating new data repositories,
                                                                    *   we use these mempools */
    parsec_mempool_t         dependencies_mempool; /**< If using hashtables to store dependencies
                                                    *   those are allocated using this mempool */

    /* This field should always be the last one in the structure. Even if the
     * declared number of execution units is 1, when we allocate the memory
     * we will allocate more (as many as we need), so everything after this
     * field might be overwritten.
     */
    parsec_execution_stream_t* execution_streams[1];
};

/* The communication layer is up and running, or at least active */
#define PARSEC_CONTEXT_FLAG_COMM_ACTIVE    0x0001
/* All the PaRSEC threads associated with the context are up and running. */
#define PARSEC_CONTEXT_FLAG_CONTEXT_ACTIVE 0x0002

/**
 * All virtual processes belong to a single physical
 * process
 */
struct parsec_context_s {
    volatile int32_t __parsec_internal_finalization_in_progress;
    volatile int32_t __parsec_internal_finalization_counter;
    volatile int32_t active_taskpools;
    volatile int32_t flags;

    void*   comm_ctx;    /**< opaque communication context */
    int32_t nb_nodes;    /**< nb of physical processes */
    int32_t my_rank;     /**< rank of this physical process */

    parsec_barrier_t  barrier;

    size_t remote_dep_fw_mask_sizeof; /* Size of the remote dep fw mask */

    pthread_t *pthreads; /**< all POSIX threads used for computation are stored here in order
                          *   threads[0] is uninitialized, this is the user's thread
                          *   threads[1] = thread for vp=0, th=1, if vp[0]->nbcores > 1
                          *   threads[n] = thread(vp=1, th=0) if vp[0]->nb_cores = n
                          *   etc...
                          */

    int32_t nb_vp; /**< number of virtual processes in this physical process */

    int32_t taskpool_array_size; /**< size of array to save reference of dtd taskpools */
    int32_t taskpool_array_occupied; /**< count of dtd taskpools registered */
    parsec_taskpool_t **taskpool_array; /**< array of dtd taskpools registered with this context */

#if defined(PARSEC_SIM)
    int largest_simulation_date;
#endif

#ifdef PARSEC_HAVE_HWLOC
    int comm_th_core;  /* if specified on the MCA subsystem it holds the core where the
                        * communication thread is to be bound.
                        */
    /* Indicates the HWLOC bitmap of all hardware cores that are described by the MCA
     * thread location variables.
     */
    hwloc_cpuset_t cpuset_allowed_mask;
    /* Describe the HWLOC bitmap for all cores that are part of cpuset_allowed_mask
     * but were not used to currently bind computational threads.
     */
    hwloc_cpuset_t cpuset_free_mask;
#endif

    /* This field should always be the last one in the structure. Even if the
     * declared number of virtual processes is 1, when we allocate the memory
     * we will allocate more (as many as we need), so everything after this
     * field might be overwritten.
     */
    struct parsec_vp_s* virtual_processes[1];
};

#define PARSEC_THREAD_IS_MASTER(eu) ( ((eu)->th_id == 0) && ((eu)->virtual_process->vp_id == 0) )

END_C_DECLS

#endif  /* PARSEC_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED */
