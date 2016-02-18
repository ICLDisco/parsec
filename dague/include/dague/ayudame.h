/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#if defined(DAGUE_HAVE_AYUDAME)

#include <dague_internal.h>
#include <Ayudame.h>

#define AYU_INIT()                               \
  if( AYU_event ) {                              \
    enum ayu_runtime_t ayu_rt = AYU_RT_GOMP;     \
    AYU_event( AYU_PREINIT, 0, (void*)&ayu_rt ); \
  }

#define AYU_FINI()                    \
  if( AYU_event ) {                   \
    AYU_event( AYU_FINISH, 0, NULL ); \
  }

/**
 * Register a family of tasks.
 */
#define AYU_REGISTER_TASK(FCT) \
  if( AYU_event ) { \
    AYU_event( AYU_REGISTERFUNCTION, (FCT)->function_id, (void*)((FCT)->name) ); \
  }

/**
 * Advertise the creation of a new task. Once created a task will
 * be tracked by the system until the completion event is generated.
 */
#define AYU_ADD_TASK(TASK)                                 \
  if( AYU_event ) {                                        \
    int64_t _data[2] = {(TASK)->function->function_id, 1}; \
    AYU_event( AYU_ADDTASK, (int64_t)(TASK), _data );      \
  }

/**
 * Create the dependencies between tasks.
 */
#define AYU_ADD_TASK_DEP(TASK, ID) \
  do { \
    uintptr_t _data[3]; \
    _data[0] = (uintptr_t)(TASK)->data[(ID)].data_repo->generator; \
    _data[1] = (uintptr_t)(TASK)->data[(ID)].data; \
    _data[2] = (uintptr_t)(TASK)->data[(ID)].data; \
    AYU_event( AYU_ADDDEPENDENCY, (int64_t)(TASK), (void*)_data ); \
   } while (0)

#define AYU_ADD_TASK_DEPS(TASK) \
  if( AYU_event ) { \
    int _i; \
    for( _i = 0; _i < (TASK)->function->nb_flows; _i++ ) { \
      AYU_ADD_TASK_DEP(TASK, _i); \
    } \
  }

/*
 * The task is ready to be executed. Until computing units are
 * available it will stay in the queues.
 */
#define AYU_TASK_READY(THID, TASK) \
  if( AYU_event ) { \
    int _thid = (THID); \
    AYU_event( AYU_ADDTASKTOQUEUE, (int64_t)(TASK), &_thid ); \
  }

/**
 * The task has been selected from the waiting queues and it will be
 * executed by the thread.
 */
#define AYU_TASK_RUN(THID, TASK) \
  if( AYU_event ) { \
    int _thid = (THID); \
    AYU_event( AYU_RUNTASK, (int64_t)(TASK), &_thid ); \
  }

/**
 * The excution has been completed and the task will be removed from
 * the system.
 */
#define AYU_TASK_COMPLETE(TASK) \
  if( AYU_event ) { \
    AYU_event( AYU_REMOVETASK, (int64_t)(TASK), NULL ); \
  }

/**
 * Call this macro only when the task execution has failed. If the task
 * is resubmited, it should go through all the steps again.
 */
#define AYU_TASK_FAILED(TASK) \
  if( AYU_event ) { \
    AYU_event( AYU_RUNTASKFAILED, (int64_t)(TASK), NULL ); \
  }

#else

#define AYU_INIT()
#define AYU_FINI()
#define AYU_REGISTER_TASK(FCT)
#define AYU_ADD_TASK(TASK)
#define AYU_ADD_TASK_DEP(TASK, ID)
#define AYU_ADD_TASK_DEPS(TASK)
#define AYU_TASK_READY(THID, TASK)
#define AYU_TASK_RUN(THID, TASK)
#define AYU_TASK_COMPLETE(TASK)
#define AYU_TASK_FAILED(TASK)

#endif  /* defined(DAGUE_HAVE_AYUDAME) */
