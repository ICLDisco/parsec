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

/**
 * Initialize the DAG grapher
 *
 * parsec_context is the context to graphe (currently, only one context at the time
 *                       is supported)
 * filename is a unique filename (the caller is responsible to ensure lack 
 *                       of collision between ranks)
 */
void  parsec_prof_grapher_init(const parsec_context_t *parsec_context, const char *filename);

/**
 * Log a task executing on this rank
 */
void  parsec_prof_grapher_task(const struct parsec_task_s *context, int thread_id, int vp_id, uint64_t task_hash);

/**
 * Log a dependency detected on this rank between task from and task to
 *   on flow origin_flow/dest_flow, dependency dep
 */
void  parsec_prof_grapher_dep(const struct parsec_task_s* from, const struct parsec_task_s* to,
                             int  dependency_activates_task,
                             const struct parsec_flow_s* origin_flow, const struct parsec_flow_s* dest_flow);

/**
 * Log a data input from task task, on data data, using flow flow.
 * direct_reference is 1 if this task is directly referring data,
 * 0 if the flow passed the data from another task
 */
void  parsec_prof_grapher_data_input(const struct parsec_data_s *data, const struct parsec_task_s *task, const struct parsec_flow_s *flow, int direct_reference);

/**
 * Log a direct data output, from task task on data data, using
 * flow flow
 */
void  parsec_prof_grapher_data_output(const struct parsec_task_s *task, const struct parsec_data_s *data, const struct parsec_flow_s *flow);

/**
 * Returns the identifier of task context used by the DAG grapher
 * writes up to length-1 bytes in tmp, and return tmp
 */
char *parsec_prof_grapher_taskid(const struct parsec_task_s* context, char *tmp, int length);

/**
 * Completes and close the output file
 */
void  parsec_prof_grapher_fini(void);

END_C_DECLS

/** @} */

#endif /* _parsec_prof_grapher_h */
