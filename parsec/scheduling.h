/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_SCHEDULING_H_HAS_BEEN_INCLUDED
#define PARSEC_SCHEDULING_H_HAS_BEEN_INCLUDED

/** 
 *  @addtogroup parsec_internal_scheduling
 *  @{
 */


#include "parsec/runtime.h"

BEGIN_C_DECLS

/**
 * Mark tasks (ring) as being ready to be scheduled, i.e. all input
 * dependencies are resolved. These tasks can be progressed immediately
 * or delayed until resources become available.
 *
 * @param[in] es The execution stream where the task is to be proposed
 *             for scheduling. This is a hint, as the scheduling engine
 *             is free to push the task where it decides.
 * @param[in] task_ring A ring of tasks to be executed. The next step in each
 *            task execution will depend on the DSL engine.
 * @param[in] distance Suggested distance to the current state where the tasks
 *             are to be pushed. The larger the value (in absolute) the
 *             further away the tasks will be pushed. This is a hint
 *             that the schedulers are free to ignore.
 *
 * @return PARSEC_SUCCESS    If the execution was successful and all output
 *                           dependencies have been correctly marked.
 * @return less than PARSEC_SUCCESS  If something went wrong.
 */
int __parsec_schedule( parsec_execution_stream_t*,
                       parsec_task_t*,
                       int32_t distance);

/**
 * Schedule an array of rings of tasks with one entry per virtual
 * process. Each entry contains a ring of tasks similar to __parsec_schedule.
 * By default this version will save the highest priority task
 * (assuming the ring is ordered or the first task in the ring otherwise)
 * on the current execution stream virtual process as the next task to be
 * executed on the current execution stream. Everything else gets pushed
 * into the execution stream 0 of the corresponding virtual process.
 *
 * @param[in] es The execution stream where the tasks were discovered, or
 *             generated.
 * @param[in] tasks_ring An array of rings of tasks to be executed. The next
 *            step in each task execution will depend on the DSL engine, but
 *            the step will be executed on the virtual process corresponding
 *            to the location in the array of rings.
 * @param[in] distance Suggested distance to the current state where the tasks
 *             are to be pushed. The larger the value (in absolute) the
 *             further away the tasks will be pushed. This is a hint
 *             that the schedulers are free to ignore.
 *
 * @return PARSEC_SUCCESS    If the execution was successful and all output
 *                           dependencies have been correctly marked.
 * @return less than PARSEC_SUCCESS  If something went wrong.
 */
int __parsec_schedule_vp( parsec_execution_stream_t*,
                          parsec_task_t**,
                          int32_t distance);

int __parsec_schedule_activity( parsec_execution_stream_t *es,
                                parsec_task_t *task);

/**
 * @brief Reschedule a task on the most appropriate resource.
 *
 * @details The function reschedules a task, by trying to locate it as closer
 *          as possible to the current execution unit. If not available
 *          execution unit was found, the task is rescheduled on the same
 *          execution unit. To find the most appropriate execution unit
 *          we start from the next execution unit after the current one, and
 *          iterate over all existing execution units (in the current VP,
 *          then on the next VP and so on).
 *
 * @param [IN] es, the start execution_stream (normally it is the current one).
 * @param [IN] task, the task to be rescheduled.
 *
 * @return PARSEC_SUCCESS    If the execution was successful and all output
 *                           dependencies have been correctly marked.
 * @return less than PARSEC_SUCCESS  If something went wrong.
 */
int __parsec_reschedule(parsec_execution_stream_t* es,
                        parsec_task_t* task);

/**
 * @brief Enter the progress engine for this execution unit
 *
 * @details The function enters the progress engine for eu.
 *          If eu is a master execution unit, the function returns once
 *          all parsec handles scheduled on that node have been completed
 *          If eu is not a master execution unit, the function returns
 *          once the master execution unit has stopped the execution
 *          by calling parsec_fini.
 *
 * @param[in] eu_context the execution_unit that should start progressing.
 *
 * @return parsec scheduling return code
 */
int __parsec_context_wait(parsec_execution_stream_t* es);

/**
 * Execute the body of the task associated to the context.
 *
 * @return PARSEC_HOOK_RETURN_DONE  Task completed.
 * @return PARSEC_HOOK_RETURN_NEXT  Task not executed, try the next variant (if any).
 * @return PARSEC_HOOK_RETURN_ASYNC Task is not yet completed, it will be completed asynchronously later.
 * @return PARSEC_HOOK_RETURN_ERROR If something went wrong.
 */
int __parsec_execute( parsec_execution_stream_t*, parsec_task_t*);

/**
 * Signal the termination of the execution context to all dependencies of
 * its dependencies.
 *
 * @param[in]  eu_context The execution context of the finished task.
 * @param[in]  exec_context The task to be completed
 *
 * @return PARSEC_SUCCESS   If the dependencies have successfully been signaled.
 * @return less than PARSEC_SUCCESS If something went wrong.
 */
int __parsec_complete_execution( parsec_execution_stream_t *es,
                                 parsec_task_t *task);

/**
 * Signal the handle that a certain number of runtime bound activities have been
 * completed. Such activities includes network communications, other local data
 * transfers, and more generally any activity generated by the runtime itself
 * that is related to the handle. In addition, we assume that one extra activity
 * is to the capability of the upper level to generate tasks, activity that has
 * it's own counter, handled via parsec_taskpool_update_nbtask. Thus, once the
 * upper level knows no local tasks will be further generate it is it's
 * responsibility to update the runtime counter accordingly.
 *
 * @return 0 if the handle has not been completed.
 * @return 1 if the handle has been completed and it has been marked for release.
 */
int parsec_taskpool_update_runtime_nbtask(parsec_taskpool_t *tp, int32_t nb_tasks);

/**
 * @brief Scheduling-level termination detection callback
 *
 * @details This function must be called when a taskpool is complete.
 * This is usually done through a callback into the termination
 * detection object, but DSLs may want to do additional operations,
 * so they can call this function from their own callback.
 */
void parsec_taskpool_termination_detected(parsec_taskpool_t *tp);

/**
 * Loads the scheduler as selected using the MCA logic
 * You better not call this while computations are in progress,
 *  i.e. it should be safe to call this when the main thread is
 *  not yet inside parsec_progress, but *before* any call to
 *  parsec_progress...
 *
 *  @return 1 if the new scheduler was successfully installed
 *          0 if it failed. In this case, the previous scheduler
 *            is kept.
 */
int parsec_set_scheduler( parsec_context_t *parsec );

/**
 *  Removes the current scheduler (cleanup)
 */
void parsec_remove_scheduler( parsec_context_t *parsec );

struct parsec_sched_module_s;
extern struct parsec_sched_module_s *parsec_current_scheduler;


END_C_DECLS

/** @} */

#endif  /* PARSEC_SCHEDULING_H_HAS_BEEN_INCLUDED */
