/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _dague_prof_grapher_h
#define _dague_prof_grapher_h

#include "dague_config.h"

#if defined(DAGUE_PROF_GRAPHER)

#include "dague_internal.h"
#include "execution_unit.h"

void  dague_prof_grapher_init(const char *base_filename, int nbthreads);
void  dague_prof_grapher_task(const dague_execution_context_t *context, int thread_id, int vp_id, int task_hash);
void  dague_prof_grapher_dep(const dague_execution_context_t* from, const dague_execution_context_t* to,
                             int  dependency_activates_task,
                             const dague_flow_t* origin_flow, const dague_flow_t* dest_flow);
char *dague_prof_grapher_taskid(const dague_execution_context_t *context, char *tmp, int length);
void  dague_prof_grapher_fini(void);


#else

#define dague_prof_grapher_init(f, n)           do {} while(0)
#define dague_prof_grapher_task(c, t, p, h)     do {} while(0)
#define dague_prof_grapher_dep(f, t, b, fp, tp) do {} while(0)
#define dague_prof_grapher_taskid(c, t, l)      do {} while(0)
#define dague_prof_grapher_fini()               do {} while(0)

#endif /* DAGUE_PROF_GRAPHER */

#endif /* _dague_prof_grapher_h */
