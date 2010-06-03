/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _DAGuE_scheduling_h
#define _DAGuE_scheduling_h

#include "DAGuE.h"

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
int DAGuE_schedule( DAGuE_context_t*, const DAGuE_execution_context_t* );
int __DAGuE_schedule( DAGuE_execution_unit_t*, DAGuE_execution_context_t*, int use_placeholder );

int DAGuE_progress(DAGuE_context_t* context);
void* __DAGuE_progress(DAGuE_execution_unit_t* eu_context);

void DAGuE_register_nb_tasks(DAGuE_context_t* context, int32_t n);



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
int DAGuE_trigger_dependencies( DAGuE_execution_unit_t*,
                                 const DAGuE_execution_context_t*,
                                 int forward_remote );
//#endif /* DEPRECATED */

#endif  /* _DAGuE_scheduling_h */

