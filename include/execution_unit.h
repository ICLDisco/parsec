/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED
#define DAGUE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

#ifdef HAVE_HWLOC
#include <hwloc.h>
#endif

#include <pthread.h>
#include <stdint.h>
#include "hbbuffer.h"
#include "mempool.h"
#include "profiling.h"
#include "barrier.h"

/**
 *  Computational Thread-specific structure
 */
struct dague_execution_unit {
    int32_t   th_id;          /**< Internal thread identifier. A thread belongs to a vp */
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

    struct dague_vp        *virtual_process;   /**< Backlink to the virtual process that holds this thread */
    dague_thread_mempool_t *context_mempool;
    dague_thread_mempool_t *datarepo_mempools[MAX_PARAM_COUNT+1];
};

/**
 * Threads are grouped per virtual process
 */
struct dague_vp {
    dague_context_t *dague_context; /**< backlink to the global context */
    int32_t vp_id;                  /**< virtual process identifier of this vp */
    int32_t nb_cores;               /**< number of cores for this vp */

    dague_mempool_t context_mempool;
    dague_mempool_t datarepo_mempools[MAX_PARAM_COUNT+1];

    /* This field should always be the last one in the structure. Even if the
     * declared number of execution units is 1, when we allocate the memory
     * we will allocate more (as many as we need), so everything after this
     * field might be overwritten.
     */
    dague_execution_unit_t* execution_units[1];
};

/**
 * All virtual processes belong to a single physical
 * process
 */
struct dague_context_t {
    volatile int32_t __dague_internal_finalization_in_progress;
    volatile int32_t __dague_internal_finalization_counter;
    volatile uint32_t active_objects;
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

#ifdef HAVE_HWLOC
    int comm_th_core;
    hwloc_cpuset_t comm_th_index_mask;
    hwloc_cpuset_t index_core_free_mask;
#endif

    /* This field should always be the last one in the structure. Even if the
     * declared number of virtual processes is 1, when we allocate the memory
     * we will allocate more (as many as we need), so everything after this
     * field might be overwritten.
     */
    struct dague_vp* virtual_processes[1];
};

#define DAGUE_THREAD_IS_MASTER(eu) ( ((eu)->th_id == 0) && ((eu)->virtual_process->vp_id == 0) )

#endif  /* DAGUE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED */
