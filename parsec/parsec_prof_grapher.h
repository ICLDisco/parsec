/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _parsec_prof_grapher_h
#define _parsec_prof_grapher_h

/**
 *  @addtogroup parsec_internal_profiling
 *  @{
 */

#include "parsec/runtime.h"

BEGIN_C_DECLS

struct parsec_task_s;
struct parsec_flow_s;
struct parsec_data_s;

void  parsec_prof_grapher_init(const char *base_filename, int nbthreads);
void  parsec_prof_grapher_task(const struct parsec_task_s *context, int thread_id, int vp_id, uint64_t task_hash);
void  parsec_prof_grapher_dep(const struct parsec_task_s* from, const struct parsec_task_s* to,
                             int  dependency_activates_task,
                             const struct parsec_flow_s* origin_flow, const struct parsec_flow_s* dest_flow);
void  parsec_prof_grapher_data_input(const struct parsec_data_s *data, const struct parsec_task_s *task, const struct parsec_flow_s *flow);
void  parsec_prof_grapher_data_output(const struct parsec_task_s *task, const struct parsec_data_s *data, const struct parsec_flow_s *flow);
char *parsec_prof_grapher_taskid(const struct parsec_task_s* context, char *tmp, int length);
void  parsec_prof_grapher_fini(void);

END_C_DECLS

/** @} */

#endif /* _parsec_prof_grapher_h */
