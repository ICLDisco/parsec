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
int dplasma_schedule( const dplasma_execution_context_t* exec_context );

int dplasma_progress();

void dplasma_register_nb_tasks(int32_t n);

#endif  /* _dplasma_scheduling_h */

