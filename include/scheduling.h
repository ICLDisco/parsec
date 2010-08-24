/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _DAGUE_scheduling_h
#define _DAGUE_scheduling_h

#include "dague.h"

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
int dague_schedule( dague_context_t*, const dague_execution_context_t* );
int __dague_schedule( dague_execution_unit_t*, dague_execution_context_t*, int use_placeholder );

int dague_progress(dague_context_t* context);
void* __dague_progress(dague_execution_unit_t* eu_context);

void dague_register_nb_tasks(dague_context_t* context, int32_t n);



//#ifdef DEPRECATED
/**
 * Signal the termination of the execution context to all dependencies of 
 * its dependencies.  
 * 
 * @param [IN]  The exeuction context of the finished task.
 * @param [IN]  when forward_remote is 0, only local (in the sense of the 
 *              process grid predicates) dependencies are satisfied.
 *
 * @return 0    If the dependencies have successfully been signaled.
 * @return -1   If something went wrong. 
 */
int dague_trigger_dependencies( const struct dague_object *dague_object,
                                dague_execution_unit_t*,
                                const dague_execution_context_t*,
                                int forward_remote );
//#endif /* DEPRECATED */

#endif  /* _DAGUE_scheduling_h */

