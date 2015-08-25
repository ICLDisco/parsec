/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _DAGUE_scheduling_h
#define _DAGUE_scheduling_h

#include "dague_internal.h"

BEGIN_C_DECLS

/**
 * Mark a execution context as being ready to be scheduled, i.e. all
 * input dependencies are resolved. The execution context can be
 * executed immediately or delayed until resources become available.
 *
 * @param [IN] The execution context to be executed. This include
 *             calling the attached hook (if any) as well as marking
 *             all dependencies as completed.
 *
 * @return  0 If the execution was succesful and all output dependencies
 *            has been correctly marked.
 * @return -1 If something went wrong.
 */
int __dague_schedule( dague_execution_unit_t*, dague_execution_context_t*);

int __dague_context_wait(dague_execution_unit_t* eu_context);

/**
 * Execute the body of the task associated to the context.
 */
int __dague_execute( dague_execution_unit_t*, dague_execution_context_t*);
/**
 * Signal the termination of the execution context to all dependencies of
 * its dependencies.
 *
 * @param [IN]  The execution context of the finished task.
 * @param [IN]  The task to be completed
 *
 * @return 0    If the dependencies have successfully been signaled.
 * @return -1   If something went wrong.
 */
int __dague_complete_execution( dague_execution_unit_t *eu_context,
                              dague_execution_context_t *exec_context );
/**
 * Mark a task belonging to dague_handle as complete, and update the
 * corresponding statuses. If it was the last task in the dague_handle,
 * trigger the completion callback and then update the main context.
 * Otherwise, just update the dague_handle.
 */
int __dague_complete_task(dague_handle_t *dague_handle, dague_context_t* context);

/**
 * When changing the number of local tasks, see if we need to call the
 * DAG complete_cb callback, and/or if we need to update the number of
 * active objects.
 *
 * remaining is the number of local tasks available, after updating it
 * using the appropriate atomic operation
 */
int dague_check_complete_cb(dague_handle_t *dague_handle, dague_context_t *context, int remaining);

/**
 * Loads the scheduler as selected using the MCA logic
 * You better not call this while computations are in progress,
 *  i.e. it should be safe to call this when the main thread is
 *  not yet inside dague_progress, but *before* any call to
 *  dague_progress...
 *
 *  @RETURN 1 if the new scheduler was succesfully installed
 *          0 if it failed. In this case, the previous scheduler
 *            is kept.
 */
int dague_set_scheduler( dague_context_t *dague );

/**
 *  Removes the current scheduler (cleanup)
 */
void dague_remove_scheduler( dague_context_t *dague );

struct dague_sched_module_s;
extern struct dague_sched_module_s *current_scheduler;

END_C_DECLS

#endif  /* _DAGUE_scheduling_h */
