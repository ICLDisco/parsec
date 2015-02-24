/*
 * Copyright (c) 2009-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_H_HAS_BEEN_INCLUDED
#define DAGUE_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

BEGIN_C_DECLS

typedef struct dague_handle_s            dague_handle_t;
typedef struct dague_execution_context_s dague_execution_context_t;
/**< The general context that holds all the threads of dague for this MPI process */
typedef struct dague_context_s           dague_context_t;

/**
 * TO BE REMOVED.
 */
typedef void* (*dague_data_allocate_t)(size_t matrix_size);
typedef void (*dague_data_free_t)(void *data);
extern dague_data_allocate_t dague_data_allocate;
extern dague_data_free_t     dague_data_free;

/**
 * CONTEXT MANIPULATION FUNCTIONS.
 */

/**
 * Create a new execution context, using the number of resources passed
 * with the arguments. Every execution happend in the context of such an
 * execution context. Several contextes can cohexist on disjoint resources
 * in same time.
 */
dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[]);

/**
 * Reset the remote_dep comm engine associated with @context, and use
 * the communication context @opaque_comm_ctx in the future
 * (typically an MPI communicator);
 *   dague_context_wait becomes collective accross nodes spanning on this
 *   communication context.
 */
int dague_remote_dep_set_ctx( dague_context_t* context, void* opaque_comm_ctx );


/**
 * Complete all pending operations on the execution context, and release
 * all associated resources. Threads and acclerators attached to this
 * context will be released.
 */
int dague_fini( dague_context_t** pcontext );

/**
 * Attach an execution handle on a context, in other words on the set of
 * resources associated to this particular context. A matching between
 * the capabilitis of the context and the support from the handle will be
 * done during this step, which will basically define if accelerators can
 * be used for the execution.
 */
int dague_enqueue( dague_context_t* context, dague_handle_t* handle);

/**
 * Start the runtime by allowing all the other threads to start executing.
 * This call should be paired with one of the completion calls, test or wait.
 *
 * @returns: 0 if the other threads in this context have been started, -1 if the
 * context was already active, -2 if there was nothing to do and no threads hav
 * been activated.
 */
int dague_context_start(dague_context_t* context);

/**
 * Check the status of a context. No progress on the context is guaranteed.
 *
 * @returns: 0 if the context is active, a negative value to indicate an error, or
 *           any positive value to indicate that the context is not active.
 */
int dague_context_test( dague_context_t* context );

/**
 * Progress the execution context until no further operations are available.
 * Upon return from this function, all resources (threads and accelerators)
 * associated with the corresponding context are put in a mode where they are
 * not active. New handles enqueued during the progress stage are automatically
 * taken into account, and the caller of this function will not return to the
 * user until all pending handles are completed and all other threads are in a
 * sleeping mode.
 */
int dague_context_wait(dague_context_t* context);

/**
 * HANDLE MANIPULATION FUNCTIONS.
 */

/**
 * The completion callback of a dague_handle. Once the handle has been
 * completed, i.e. all the local tasks associated with the handle have
 * been executed, and before the handle is marked as done, this callback
 * will be triggered. Inside the callback the handle should not be
 * modified.
 */
typedef int (*dague_completion_cb_t)(dague_handle_t* dague_handle, void*);

/* Accessors to set and get the completion callback and data */
int dague_set_complete_callback(dague_handle_t* dague_handle,
                                dague_completion_cb_t complete_cb, void* complete_data);
int dague_get_complete_callback(const dague_handle_t* dague_handle,
                                dague_completion_cb_t* complete_cb, void** complete_data);

/**< Retrieve the local object attached to a unique object id */
dague_handle_t* dague_handle_lookup(uint32_t handle_id);
/**< Reverse an unique ID for the handle. Beware that on a distributed environment the
 * connected objects must have the same ID.
 */
int dague_handle_reserve_id(dague_handle_t* handle);
/**< Register the object with the engine. The object must have a unique handle, especially
 * in a distributed environment.
 */
int dague_handle_register(dague_handle_t* handle);
/**< Unregister the object with the engine. This make the handle available for
 * future handles. Beware that in a distributed environment the connected objects
 * must have the same ID.
 */
void dague_handle_unregister(dague_handle_t* handle);
/**< globally synchronize object id's so that next register generates the same
 *  id at all ranks. */
void dague_handle_sync_ids(void);

/**
 * Compose sequentially two handles. If start is already a composed
 * object, then next will be added sequentially to the list. These
 * handles will execute one after another as if there were sequential.
 * The resulting compound dague_handle is returned.
 */
dague_handle_t* dague_compose(dague_handle_t* start, dague_handle_t* next);

/**< Free the resource allocated in the dague handle. The handle should be unregistered first. */
void dague_handle_free(dague_handle_t *handle);

/**< Update the number of tasks by adding the increment (if the increment is negative
 * the number of tasks is decreased).
 */
void dague_handle_update_nbtask( dague_handle_t* handle, int32_t nb_tasks );

/**< Print DAGuE usage message */
void dague_usage(void);

/**
 * Allow to change the default priority of an object. It returns the
 * old priority (the default priorityy of an object is 0). This function
 * can be used during the lifetime of an object, however, only tasks
 * generated after this call will be impacted.
 */
int32_t dague_set_priority( dague_handle_t* object, int32_t new_priority );

/* Dump functions */
char* dague_snprintf_execution_context( char* str, size_t size,
                                        const dague_execution_context_t* task);
struct dague_function_s;
struct assignment_s;
char* dague_snprintf_assignments( char* str, size_t size,
                                  const struct dague_function_s* function,
                                  const struct assignment_s* locals);

END_C_DECLS

#endif  /* DAGUE_H_HAS_BEEN_INCLUDED */
