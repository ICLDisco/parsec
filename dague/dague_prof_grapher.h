/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _dague_prof_grapher_h
#define _dague_prof_grapher_h

#include "dague_config.h"

#include "dague_internal.h"
#include "dague/execution_unit.h"

BEGIN_C_DECLS

void  dague_prof_grapher_init(const char *base_filename, int nbthreads);
void  dague_prof_grapher_task(const dague_execution_context_t *context, int thread_id, int vp_id, int task_hash);
void  dague_prof_grapher_dep(const dague_execution_context_t* from, const dague_execution_context_t* to,
                             int  dependency_activates_task,
                             const dague_flow_t* origin_flow, const dague_flow_t* dest_flow);
char *dague_prof_grapher_taskid(const dague_execution_context_t *context, char *tmp, int length);
void  dague_prof_grapher_fini(void);

END_C_DECLS

#endif /* _dague_prof_grapher_h */
