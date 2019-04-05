/*
 * Copyright (c) 2012-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#if defined(PARSEC_HAVE_AYUDAME)

#include "parsec/parsec_internal.h"
#include <Ayudame.h>

#define PARSEC_AYU_INIT()                        \
  if( AYU_event ) {                              \
    enum ayu_runtime_t ayu_rt = AYU_RT_GOMP;     \
    AYU_event( AYU_PREINIT, 0, (void*)&ayu_rt ); \
  }

#define PARSEC_AYU_FINI()                    \
  if( AYU_event ) {                   \
    AYU_event( AYU_FINISH, 0, NULL ); \
  }

/**
 * Register a family of tasks.
 */
#define PARSEC_AYU_REGISTER_TASK(FCT) \
  if( AYU_event ) { \
    AYU_event( AYU_REGISTERFUNCTION, (FCT)->task_class_id, (void*)((FCT)->name) ); \
  }

/**
 * Advertise the creation of a new task. Once created a task will
 * be tracked by the system until the completion event is generated.
 */
#define PARSEC_AYU_ADD_TASK(TASK)                                 \
  if( AYU_event ) {                                        \
    int64_t _data[2] = {(TASK)->task_class->task_class_id, 1}; \
    AYU_event( AYU_ADDTASK, (int64_t)(TASK), _data );      \
  }

/**
 * Create the dependencies between tasks.
 */
#define PARSEC_AYU_ADD_TASK_DEP(TASK, ID) \
  do { \
    uintptr_t _data[3]; \
    _data[0] = (uintptr_t)(TASK)->data[(ID)].data_repo->generator; \
    _data[1] = (uintptr_t)(TASK)->data[(ID)].data_in->original; \
    _data[2] = (uintptr_t)(TASK)->data[(ID)].data_out->original; \
    AYU_event( AYU_ADDDEPENDENCY, (int64_t)(TASK), (void*)_data ); \
   } while (0)

#define PARSEC_AYU_ADD_TASK_DEPS(TASK) \
  if( AYU_event ) { \
    int _i; \
    for( _i = 0; _i < (TASK)->task_class->nb_flows; _i++ ) { \
      PARSEC_AYU_ADD_TASK_DEP(TASK, _i); \
    } \
  }

/*
 * The task is ready to be executed. Until computing units are
 * available it will stay in the queues.
 */
#define PARSEC_AYU_TASK_READY(THID, TASK) \
  if( AYU_event ) { \
    int _thid = (THID); \
    AYU_event( AYU_ADDTASKTOQUEUE, (int64_t)(TASK), &_thid ); \
  }

/**
 * The task has been selected from the waiting queues and it will be
 * executed by the thread.
 */
#define PARSEC_AYU_TASK_RUN(THID, TASK) \
  if( AYU_event ) { \
    int _thid = (THID); \
    AYU_event( AYU_RUNTASK, (int64_t)(TASK), &_thid ); \
  }

/**
 * The excution has been completed and the task will be removed from
 * the system.
 */
#define PARSEC_AYU_TASK_COMPLETE(TASK) \
  if( AYU_event ) { \
    AYU_event( AYU_REMOVETASK, (int64_t)(TASK), NULL ); \
  }

/**
 * Call this macro only when the task execution has failed. If the task
 * is resubmited, it should go through all the steps again.
 */
#define PARSEC_AYU_TASK_FAILED(TASK) \
  if( AYU_event ) { \
    AYU_event( AYU_RUNTASKFAILED, (int64_t)(TASK), NULL ); \
  }

#else

#define PARSEC_AYU_INIT()
#define PARSEC_AYU_FINI()
#define PARSEC_AYU_REGISTER_TASK(FCT)
#define PARSEC_AYU_ADD_TASK(TASK)
#define PARSEC_AYU_ADD_TASK_DEP(TASK, ID)
#define PARSEC_AYU_ADD_TASK_DEPS(TASK)
#define PARSEC_AYU_TASK_READY(THID, TASK)
#define PARSEC_AYU_TASK_RUN(THID, TASK)
#define PARSEC_AYU_TASK_COMPLETE(TASK)
#define PARSEC_AYU_TASK_FAILED(TASK)

#endif  /* defined(PARSEC_HAVE_AYUDAME) */
