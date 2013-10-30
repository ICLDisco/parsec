/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _DAGUE_profiling_h
#define _DAGUE_profiling_h

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

/**
 * Note about thread safety:
 *  Some functions are thread safe, others are not.
 *  the tracing function is not thread-safe: this is addressed here
 *  by providing a thread-specific context for the only operation
 *  that should happen in parallel.
 */

typedef struct dague_thread_profiling_s dague_thread_profiling_t;

/**
 * Initializes the profiling engine. Call this ONCE per process.
 *
 * @return 0    if success, -1 otherwise
 *
 * not thread safe
 */
int dague_profiling_init( void );

/**
 * Set the reference time to now in the profiling system.
 * Optionally called before any even is traced.
 * Not thread safe.
 */
void dague_profiling_start(void);

/**
 * Releases all resources for the tracing.
 * Thread contexts become invalid after this call.
 * Must be called after the dbp_dump if a dbp_start was called.
 *
 * @return 0    if success, -1 otherwise.
 * not thread safe
 */
int dague_profiling_fini( void );

/**
 * Removes all current logged events. Prefer this to fini / init if you want
 * to do a new profiling with the same thread contexts. This does not
 * invalidate the current thread contexts.
 *
 * @return 0 if succes, -1 otherwise
 * not thread safe
 */
int dague_profiling_reset( void );

/**
 * Add additional information about the current run, under the form key/value.
 * Used to store the value of the globals names and values in the current run
 * Not thread safe.
 */
void dague_profiling_add_information( const char *key, const char *value );

/**
 * Initializes the buffer trace with the specified length.
 * This function must be called once per thread that will use the profiling
 * functions. This creates the profiling_thread_unit_t that must be passed to
 * the tracing function call. See note about thread safety.
 *
 * @param [IN]  length: the length (in bytes) of the buffer queue to store events.
 * @param [IN]  format, ...: printf-like to associate a human-readable
 *                           definition of the calling thread
 * @return pointer to the new thread_profiling structure. NULL if an error.
 * thread safe
 */
dague_thread_profiling_t *dague_profiling_thread_init( size_t length, const char *format, ...);

/**
 * Inserts a new keyword in the dictionnary
 * The dictionnary is process-global, and operations on it are *not* thread
 * safe. All keywords should be inserted by one thread at most, and no thread
 * should use a key before it has been inserted.
 *
 * @param [IN] name: the (human readable) name of the key
 * @param [IN] attributes: attributes that can be associated to the key (e.g. color)
 * @param [IN] info_length: the number of bytes passed as additional info
 * @param [IN] convertor_code: php code to convert a info byte array into XML code.
 * @param [OUT] key_start: the key to use to denote the start of an event of this type
 * @param [OUT] key_end: the key to use to denote the end of an event of this type.
 * @return 0    if success, -1 otherwie.
 * not thread safe
 */
int dague_profiling_add_dictionary_keyword( const char* name, const char* attributes,
                                            size_t info_length,
                                            const char* convertor_code,
                                            int* key_start, int* key_end );

/**
 * Empties the global dictionnary (usefull in conjunction with reset, if
 * you want to redo an experiment.
 *
 * Emptying the dictionnary without reseting the profiling system will yield
 * undeterminate results
 *
 * @return 0 if success, -1 otherwise.
 * not thread safe
 */
int dague_profiling_dictionary_flush( void );

/**
 * Traces one event, without a reference tile
 * Not thread safe (but it takes a thread_context parameter, and threads should not share
 * the same thread_context parameter anyway).
 *
 * @param [IN] context: a thread profiling context (should be the thread profiling context of the
 *                      calling thread).
 * @param [IN] key:     the key (as returned by add_dictionary_keyword) of the event to log
 * @param [IN] event_id:a (possibly unique) event identifier. Events are coupled together: start/end.
 *                      a couple (start, end) has
 *                        - the same key
 *                        - end is the next "end" event with the same key and the same non-null event_id and
 *                          non OBJECT_ID_NULL handle_id as start in the event buffer of the thread context
 *                        - if no matching end is found, this is an error
 * @param [IN] handle_id:
 * @param [IN] info:    a pointer to an area of size info_length for this key (see
 *                        dague_profiling_add_dictionary_keyword)
 * @return 0 if success, -1 otherwise.
 * not thread safe (if two threads share a same thread_context. Safe per thread_context)
 */
#define PROFILE_OBJECT_ID_NULL ((uint32_t)-1)
int dague_profiling_trace( dague_thread_profiling_t* context,
                           int key, uint64_t event_id, uint32_t handle_id,
                           void *info );
int dague_profiling_trace_flags(dague_thread_profiling_t* context, int key,
                                uint64_t event_id, uint32_t handle_id,
                                void *info, uint16_t flags );

/**
 * Open the profile file given as a parameter to store the
 * next events.
 *
 * @param [IN] basefile: the base name of the target file to create
 *                       the file actually created will be <basefile>-%d.profile
 * @param [IN] hr_info: human readable global information associated with this
 *                      profile. Used "uniquely" identify the experiment, and
 *                      check that all separate profile files correspond to a same
 *                      experiment.
 * @return 0 if success, -1 otherwise.
 * not thread safe.
 */
int dague_profiling_dbp_start( const char *basefile, const char *hr_info );

/**
 * Dump the current profile; finishes the file opened with dbp_start.
 * Every single dbp_start should have a matching dbp_dump.
 *
 * @return 0 if success, -1 otherwise
 * not thread safe
 */
int dague_profiling_dbp_dump( void );

/**
 * Returns a char * (owned by dague_profiling library)
 * that describes the last error that happened.
 *
 * @return NULL if no error happened before
 *         the char* of the error otherwise.
 * not thread safe
 */
char *dague_profiling_strerror(void);

/**
 * Here are some helper functions, to be used
 *  (appropriately) in dague_profiling_add_dictionary_keyword
 */

typedef struct {
    struct dague_ddesc_s *desc;
    uint32_t              id;
} dague_profile_ddesc_info_t;
extern char *dague_profile_ddesc_key_to_string;

#define DAGUE_PROFILE_STREAM_STR "GPU %d-%d"
#define DAGUE_PROFILE_THREAD_STR "DAGuE Thread %d of VP %d"

extern int dague_profile_enabled;

/**
 * Enable/disable the profiling of new events.
 */
void dague_profiling_enable(void);
void dague_profiling_disable(void);

#define DAGUE_PROFILING_TRACE(context, key, event_id, object_id, info ) \
    if( dague_profile_enabled ) {                                       \
        dague_profiling_trace(context, key, event_id, object_id, info ); \
    }

#define DAGUE_PROFILING_TRACE_FLAGS(context, key, event_id, object_id, info, flags ) \
    if( dague_profile_enabled ) {                                       \
        dague_profiling_trace_flags(context, key, event_id, object_id, info, flags ); \
    }

#endif  /* _DAGUE_profiling_h */
