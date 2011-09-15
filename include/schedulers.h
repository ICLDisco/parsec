#ifndef schedulers_h
#define schedulers_h

#include "scheduling.h"

dague_scheduler_t sched_local_hier_queues;
dague_scheduler_t sched_global_dequeue;
dague_scheduler_t sched_local_flat_queues;
dague_scheduler_t sched_absolute_priorities;

#define DAGUE_SCHEDULER_LFQ 0
#define DAGUE_SCHEDULER_GD  1
#define DAGUE_SCHEDULER_LHQ 2
#define DAGUE_SCHEDULER_AP  3

#define NB_DAGUE_SCHEDULERS 4

dague_scheduler_t *dague_schedulers_array[NB_DAGUE_SCHEDULERS];

#endif
