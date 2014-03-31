/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef schedulers_h
#define schedulers_h

#include "scheduling.h"

BEGIN_C_DECLS

#define DAGUE_SCHEDULER_LFQ 0
#define DAGUE_SCHEDULER_GD  1
#define DAGUE_SCHEDULER_LHQ 2
#define DAGUE_SCHEDULER_AP  3
#define DAGUE_SCHEDULER_PBQ 4
#define DAGUE_SCHEDULER_LTQ 5
#define NB_DAGUE_SCHEDULERS 6

extern dague_scheduler_t *dague_schedulers_array[NB_DAGUE_SCHEDULERS];

END_C_DECLS

#endif
