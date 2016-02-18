/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED
#define DAGUE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

#ifdef DAGUE_HAVE_HWLOC
#include <hwloc.h>
#endif

#include <pthread.h>
#include <stdint.h>
#include "dague/hbbuffer.h"
#include "dague/mempool.h"
#include "dague/profiling.h"
#include "dague/class/barrier.h"

#ifdef PINS_ENABLE
#include "dague/mca/pins/pins.h"
#endif

#if defined(DAGUE_HAVE_GETRUSAGE) || !defined(__bgp__)
#include <sys/time.h>
#include <sys/resource.h>
#endif

/**
 *  Computational Thread-specific structure
 */
struct dague_execution_unit_s {
    int32_t   th_id;        /**< Internal thread identifier. A thread belongs to a vp */
    int core_id;            /**< Core on which the thread is bound (hwloc in order numbering) */
    int socket_id;          /**< Socket on which the thread is bound (hwloc in order numerotation) */

    pthread_t pthread_id;     /**< POSIX thread identifier. */

#if defined(DAGUE_PROF_TRACE)
    dague_thread_profiling_t *eu_profile;
#endif /* DAGUE_PROF_TRACE */

    void *scheduler_object;

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    uint32_t sched_nb_tasks_done;
#endif

#if defined(DAGUE_SIM)
    int largest_simulation_date;
#endif

#if defined(PINS_ENABLE)
    struct parsec_pins_next_callback_s pins_events_cb[PINS_FLAG_COUNT];
#endif  /* defined(PINS_ENABLE) */

#if defined(DAGUE_PROF_RUSAGE_EU)
#if defined(DAGUE_HAVE_GETRUSAGE) || !defined(__bgp__)
    int _eu_rusage_first_call;
    struct rusage _eu_rusage;
#endif /* DAGUE_HAVE_GETRUSAGE */
#endif

    struct dague_vp_s      *virtual_process;   /**< Backlink to the virtual process that holds this thread */
    /**
     * TODO: Why do we have the mempools both in the VP and in the execution unit?
     */
    dague_thread_mempool_t *context_mempool;
    dague_thread_mempool_t *datarepo_mempools[MAX_PARAM_COUNT+1];
};

/**
 * Threads are grouped per virtual process
 */
struct dague_vp_s {
    dague_context_t *dague_context; /**< backlink to the global context */
    int32_t vp_id;                  /**< virtual process identifier of this vp */
    int32_t nb_cores;               /**< number of cores for this vp */

    dague_mempool_t  context_mempool;
    dague_mempool_t  datarepo_mempools[MAX_PARAM_COUNT+1];

    /* This field should always be the last one in the structure. Even if the
     * declared number of execution units is 1, when we allocate the memory
     * we will allocate more (as many as we need), so everything after this
     * field might be overwritten.
     */
    dague_execution_unit_t* execution_units[1];
};

/* The communication layer is up and running, or at least active */
#define DAGUE_CONTEXT_FLAG_COMM_ACTIVE    0x0001
/* All the DAGuE threads associated with the context are up and running. */
#define DAGUE_CONTEXT_FLAG_CONTEXT_ACTIVE 0x0002

/**
 * All virtual processes belong to a single physical
 * process
 */
struct dague_context_s {
    volatile int32_t __dague_internal_finalization_in_progress;
    volatile int32_t __dague_internal_finalization_counter;
    volatile uint32_t active_objects;
    volatile uint32_t flags;

    void*   comm_ctx;    /**< opaque communication context */
    int32_t nb_nodes;    /**< nb of physical processes */
    int32_t my_rank;     /**< rank of this physical process */

    dague_barrier_t  barrier;

    size_t remote_dep_fw_mask_sizeof; /* Size of the remote dep fw mask */

    pthread_t *pthreads; /**< all POSIX threads used for computation are stored here in order
                          *   threads[0] is uninitialized, this is the user's thread
                          *   threads[1] = thread for vp=0, th=1, if vp[0]->nbcores > 1
                          *   threads[n] = thread(vp=1, th=0) if vp[0]->nb_cores = n
                          *   etc...
                          */

    int32_t nb_vp; /**< number of virtual processes in this physical process */

#if defined(DAGUE_SIM)
    int largest_simulation_date;
#endif

#ifdef DAGUE_HAVE_HWLOC
    int comm_th_core;
    hwloc_cpuset_t comm_th_index_mask;
    hwloc_cpuset_t index_core_free_mask;
#endif

    /* This field should always be the last one in the structure. Even if the
     * declared number of virtual processes is 1, when we allocate the memory
     * we will allocate more (as many as we need), so everything after this
     * field might be overwritten.
     */
    struct dague_vp_s* virtual_processes[1];
};

#define DAGUE_THREAD_IS_MASTER(eu) ( ((eu)->th_id == 0) && ((eu)->virtual_process->vp_id == 0) )

#endif  /* DAGUE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED */
