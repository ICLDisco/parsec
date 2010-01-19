/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _dplasma_scheduling_h
#define _dplasma_scheduling_h

#include "dplasma.h"

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
int dplasma_schedule( dplasma_context_t*, const dplasma_execution_context_t* );
int __dplasma_schedule( dplasma_execution_unit_t*, const dplasma_execution_context_t* );


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
int dplasma_trigger_dependencies( dplasma_execution_unit_t*,
                                 const dplasma_execution_context_t*,
                                 int forward_remote );

int dplasma_progress(dplasma_context_t* context);
void* __dplasma_progress(dplasma_execution_unit_t* eu_context);

void dplasma_register_nb_tasks(dplasma_context_t* context, int32_t n);

#endif  /* _dplasma_scheduling_h */

