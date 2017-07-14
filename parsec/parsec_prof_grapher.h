/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _parsec_prof_grapher_h
#define _parsec_prof_grapher_h

/** 
 *  @addtogroup parsec_internal_profiling
 *  @{
 */

#include "parsec.h"
#include "parsec/parsec_description_structures.h"

BEGIN_C_DECLS

struct parsec_task_s;
struct parsec_flow_s;

void  parsec_prof_grapher_init(const char *base_filename, int nbthreads);
void  parsec_prof_grapher_task(const struct parsec_task_s *context, int thread_id, int vp_id, int task_hash);
void  parsec_prof_grapher_dep(const struct parsec_task_s* from, const struct parsec_task_s* to,
                             int  dependency_activates_task,
                             const struct parsec_flow_s* origin_flow, const struct parsec_flow_s* dest_flow);
char *parsec_prof_grapher_taskid(const struct parsec_task_s* context, char *tmp, int length);
void  parsec_prof_grapher_fini(void);

char *unique_color(int index, int colorspace);

END_C_DECLS

/** @} */

#endif /* _parsec_prof_grapher_h */
