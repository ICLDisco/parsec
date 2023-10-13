/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _PARSEC_profiling_h
#define _PARSEC_profiling_h

#include <stdint.h>
#include <stddef.h>

/**
 * @defgroup parsec_public_profiling Tracing System
 * @ingroup parsec_public
 * @{
 *
 * @brief The PaRSEC profiling system allows to expose information
 *   about the DAG and the runtime engine for analysis.
 * @details
 *    The profiling system is designed to be used standalone when practical.
 *    See [sp-demo.c](@ref tests/standalone-profiling/sp-demo.c) as a standalone
 *    example with complete usage of the profiling API.
 *
 * @remark Note about thread safety:
 *    Some functions are thread safe, others are not.
 *    The tracing function itself  is not thread-safe: this is addressed here
 *    by providing a thread-specific context for the only operation
 *    that should happen in parallel.
 *
 * # Concepts
 * The profiling system saves information about the execution into a binary
 * profile file that can then be converted into a HDF5 Pandas dataframe
 * using the profiling python tools available in the tools/profiling/python
 * directory.
 *
 * One file per rank is created, and the profiling python tools will merge
 * these files, and make the format portable between architecture. The rationale
 * behind this two-steps approach is to do minimum processing during the
 * execution to avoid modifying it significantly. All logging happens
 * in memory until exhaustion of the buffers.
 *
 * ## Events
 * Events are timed entities that go in pairs. The main purpose of the profiling
 * system is to log information about timing of events that happen on different
 * threads. Events have a dictionary key (which defines if the event starts or
 * end a pair), a time (determined internally), a thread (on which they happen),
 * and potentially an information structure (that logs complementary information
 * related to the start or end of the event).
 *
 * ## Event keys
 * Event keys are created in a dictionary that is global and must be consistent
 * between ranks. They come with a human readable name, an RGB color
 * specification, and an information convertor and size.
 *
 * ## Information
 * Arbitrary information can be logged with each event. To do so, when registering
 * the event type, the user must specify how many bytes are passed to the information
 * of the event type, and how to convert these bytes into meaningful fields.
 * Conversion happens offline, at binary profile processing time.
 *
 * The conversion string is a series of "name{type}" separated by ';'. The type
 * must be a base type from stdint (e.g. int64_t). The name is the field name used.
 * The user must take into account padding explicitly.
 * For example, if structures like
 * ~~~~~~~~~~~~~~{.c}
 * struct { char a; char b; int c; };
 * ~~~~~~~~~~~~~~
 * is passed as information, the structure would be 8 bytes long, and a possible
 * converting string would be "a{int8_t};b{int8_t};padding{int16_t};c{int32_t}"
 *
 * ## Key/Values pairs
 * In addition to events, the profiling system can log arbitrary character
 * strings in the form of key/value pairs that relate to the execution.
 *
 * These key/value pairs can be saved globally for the process, or per profiling
 * stream.
 *
 * Note that on success, most of these functions return 0, NOT PARSEC_SUCCESS.
 * PARSEC_SUCCESS may be lower than 0 (0 >= PARSEC_SUCCESS > error codes).
 * This is done so that this file can be used without including
 * parsec/constants.h.
 */

/**
 * @brief  Flag used when an info object is attached to the event
 */
#define PARSEC_PROFILING_EVENT_HAS_INFO         (1<<0)
/**
 * @brief Flag used when the event is a reschedule of a previous event
 */
#define PARSEC_PROFILING_EVENT_RESCHEDULED      (1<<1)
/**
 * @brief Flag used when the event's info is a counter
 * @details The event's info (if present) is an integer that
 *          should be accumulated to a value starting at 0.
 *          This might be useful to represent countable entities,
 *          like amount of tasks pending, amount of memory allocated,
 *          etc. */
#define PARSEC_PROFILING_EVENT_COUNTER          (1<<2)
/**
 * @brief time at the beginning of exeuction.
 * @details Flag used to indicate to `parsec_profiling_trace_flags` that
 *          it should call `take_time` at the beginning of its execution
 *          rather than at the end. Immediately after, there is a flag
 *          (the same bit set to zero) defined for readability.
 */
#define PARSEC_PROFILING_EVENT_TIME_AT_START    (1<<3)
/**
 * @brief time at end of execution (default)
 * @details for more details, see `PARSEC_PROFILING_EVENT_TIME_AT_START`.
 */
#define PARSEC_PROFILING_EVENT_TIME_AT_END      (0<<3)
/**
 * @brief Constant to use when no handle/object ID can be associated
 *        with an event
 * @details Some functions take a handle/object ID as a parameter.
 *          When such ID is not applicable, this constant should be
 *          passed to the parameter.
 */
#define PROFILE_OBJECT_ID_NULL ((uint32_t)-1)

/* We are not using the traditional BEGIN_C_DECL in this file
 * to limit the non-system include files dependencies to a minimum
 */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque structure used to keep thread-specific information about
 * the profiling.
 */
typedef struct parsec_profiling_stream_s parsec_profiling_stream_t;
    
/**
 * @brief Initializes the profiling engine.
 *
 * @details Call this ONCE per process.
 *   @param [IN] rank: the unique identifier of the process,
 *      typically the rank of the process in an MPI application.
 * @return 0    if success, negative otherwise
 *
 * @remark not thread safe
 */
int parsec_profiling_init( int rank );

/**
 * @brief Set the reference time to now in the profiling system.
 *   In case of a parallel (multi-processes) run, the calling
 *   library should synchronize the processes before calling this,
 *   to minimize the drift.
 *
 * @details Optionally called before any even is traced.
 * @remark Not thread safe.
 */
void parsec_profiling_start(void);

#if defined(PARSEC_HAVE_OTF2)
/**
 * @brief change the default communicator when using the OTF2
 *   backend
 *
 * @details OTF2 relies on MPI to trace events types and collect
 *   information. We also need to gather the dictionaries to build
 *   a consistent list of events at the end, and profiling_otf2
 *   does this using MPI_Gatherv, which is a collective on the
 *   communicator that is passed here.
 *
 *  @param[IN] pcomm: a pointer to the communicator handle to
 *       use.
 *
 *  @remark 
 *     - this call is a collective on *pcomm. The process calls
 *       MPI_Comm_dup on *pcomm during this call
 *     - the duplicate of the communicator is MPI_Comm_free(d) when
 *       parsec_profiling_fini is called. 
 *     - If parsec_profiling_otf2_set_comm is not called before 
 *       parsec_profiling_init, MPI_COMM_WORLD is used by default.
 *     - Only local MPI calls are issued in all other functions, so
 *       any MPI threading model should be supported.
 */
void parsec_profiling_otf2_set_comm( void *pcomm );
#endif

/**
 * @brief Releases all resources for the tracing.
 *
 * @details Thread contexts become invalid after this call.
 *          Must be called after the dbp_dump if a dbp_start was called.
 *
 * @return 0    if success, negative otherwise.
 * @remark not thread safe
 */
int parsec_profiling_fini( void );

/**
 * @brief Removes all current logged events.
 *
 * @details Prefer this to fini / init if you want
 * to do a new profiling with the same thread contexts. This does not
 * invalidate the current thread contexts.
 *
 * @return 0 if succes, negative otherwise
 * not thread safe
 */
int parsec_profiling_reset( void );

/**
 * @brief Add additional information about the current run, under the form key/value.
 *
 * @details Used to store the value of the globals names and values in the current run
 * @param[in] key key part of the key/value to store
 * @param[in] value value part of the key/value to store
 * @remark Not thread safe.
 */
void parsec_profiling_add_information( const char *key, const char *value );

/**
 * @brief Add additional information about the current run, under the form key/value.
 *
 * @details This function adds key/value pairs PER STREAM, not globally.
 * @param[in] stream stream in which to store the key/value
 * @param[in] key key part of the key/value to store
 * @param[in] value value part of the key/value to store
 * @remark Not thread safe.
 */
void parsec_profiling_stream_add_information(parsec_profiling_stream_t* stream,
                                             const char *key, const char *value );

/**
 * @brief Create a profiling stream that can be used to store events.
 *
 * @details This function create a profiling stream that is not thread-safe, it
 * must be carefully protected by the caller against concurrent accesses. Moreover,
 * this stream is not associated with any runtime resources, it is free to use
 * as necessary by the caller. The stream is however tracked by the runtime, and if
 * not removed by use user before the profiling_fini it will be dumped and disposed
 * as all other profiling streams.
 *
 * param[in] length the length (in bytes) of the buffer queue to store events.
 * param[in] format the name of the stream, following the printf convention
 * @return pointer to the new stream_profiling structure. NULL if an error.
 * @remark the call to this function is thread safe, the resulting structure is not.
 */
parsec_profiling_stream_t*
parsec_profiling_stream_init( size_t length, const char *format, ...);

/**
 * @brief set the default profiling_stream to use on the calling thread
 *
 * @details When using parsec_profiling_trace_flags_ts to log an event,
 * the default profiling_stream bound to the calling thread
 * is used. By default no profiling_stream is bound to any thread. Using
 * this function, the user decided what profiling_stream should be used
 * for the calling thread. The function returns the old bound profiling_stream
 * if any.
 *
 * @param[in] new: the new profiling_stream to bind on the calling thread
 * @return         the old profiling_stream that was bound to the calling thread
 *                 (NULL initially).
 * @remark not thread safe
 */
parsec_profiling_stream_t *parsec_profiling_set_default_thread( parsec_profiling_stream_t *stream );


/**
 * @brief Inserts a new keyword in the dictionnary
 *
 * @details The dictionnary is process-global, and operations on it are *not* thread
 * safe. All keywords should be inserted by one thread at most, and no thread
 * should use a key before it has been inserted.
 *
 * @param[in] name the (human readable) name of the key
 * @param[in] attributes attributes that can be associated to the key (e.g. color)
 * @param[in] info_length the number of bytes passed as additional info
 * @param[in] convertor_code php code to convert a info byte array into XML code.
 * @param[out] key_start the key to use to denote the start of an event of this type
 * @param[out] key_end the key to use to denote the end of an event of this type.
 * @return 0 if success, negative otherwise.
 * @remark not thread safe
 */
int parsec_profiling_add_dictionary_keyword( const char* name, const char* attributes,
                                            size_t info_length,
                                            const char* convertor_code,
                                            int* key_start, int* key_end );

/**
 * @brief Empties the global dictionnary
 *
 * @details this might be usefull in conjunction with reset, if
 * you want to redo an experiment.
 *
 * @remark Emptying the dictionnary without reseting the profiling system will yield
 * undeterminate results
 *
 * @return 0 if success, negative otherwise.
 * @remark not thread safe
 */
int parsec_profiling_dictionary_flush( void );

/**
 * @brief Trace one event
 *
 * @details Event is added to the series of events related to the context passed as argument.
 *
 * @param[in] context a thread profiling context (should be the thread profiling context of the
 *                      calling thread).
 * @param[in] key     the key (as returned by add_dictionary_keyword) of the event to log
 * @param[in] event_id a (possibly unique) event identifier. Events are coupled together: start/end.
 *                      a couple (start, end) has
 *                        - the same key
 *                        - end is the next "end" event with the same key and the same non-null event_id and
 *                          non OBJECT_ID_NULL taskpool_id as start in the event buffer of the thread context
 *                        - if no matching end is found, this is an error
 * @param[in] taskpool_id unique object/handle identifier (use PROFILE_OBJECT_ID_NULL if N/A)
 * @param[in] info    a pointer to an area of size info_length for this key (see
 *                        parsec_profiling_add_dictionary_keyword)
 * @param[in] flags   flags related to the event
 * @return 0 if success, negative otherwise.
 * @remark not thread safe (if two threads share a same thread_context. Safe per thread_context)
 */
int parsec_profiling_trace_flags(parsec_profiling_stream_t* context, int key,
                                 uint64_t event_id, uint32_t taskpool_id,
                                 const void *info, uint16_t flags );

/**
 * @brief Type of user functions to write info in pre-allocated event
 * 
 * @details
 *    @param[out] dst  address into which to write the info
 *    @param[in] data  pointer passed to parsec_profiling_trace_flags_fn_info
 *    @param[in] size  number of bytes that can be written at this address
 *    @return dst
 */
typedef void *(parsec_profiling_info_fn_t)(void *dst, const void *data, size_t size);

/**
 * @brief Trace one event
 *
 * @details Event is added to the series of events related to the context passed as argument.
 *
 * @param[in] context a thread profiling context (should be the thread profiling context of the
 *                      calling thread).
 * @param[in] key     the key (as returned by add_dictionary_keyword) of the event to log
 * @param[in] event_id a (possibly unique) event identifier. Events are coupled together: start/end.
 *                      a couple (start, end) has
 *                        - the same key
 *                        - end is the next "end" event with the same key and the same non-null event_id and
 *                          non OBJECT_ID_NULL taskpool_id as start in the event buffer of the thread context
 *                        - if no matching end is found, this is an error
 * @param[in] taskpool_id unique object/handle identifier (use PROFILE_OBJECT_ID_NULL if N/A)
 * @param[in] info_fn a pointer to a function that will write the info of the event in the allocated event
 *                    that memory is of size defined during the creation of the event
 * @param[in] info_data an opaque pointer passed back to info_fn when it is called.
 * @param[in] flags   flags related to the event
 * @return 0 if success, negative otherwise.
 * @remark not thread safe (if two threads share a same thread_context. Safe per thread_context)
 */
int parsec_profiling_trace_flags_info_fn(parsec_profiling_stream_t* context, int key,
                                         uint64_t event_id, uint32_t taskpool_id,
                                         parsec_profiling_info_fn_t *info_fn, const void *info_data, uint16_t flags );

/**
 * @brief Convenience macro used to trace events without flags
 */
#define parsec_profiling_trace(CTX, KEY, EVENT_ID, TASKPOOL_ID, INFO)     \
    parsec_profiling_trace_flags( (CTX), (KEY), (EVENT_ID), (TASKPOOL_ID), (INFO), 0 )
#define parsec_profiling_trace_info_fn(CTX, KEY, EVENT_ID, TASKPOOL_ID, INFO_FN, INFO_FN_DATA) \
    parsec_profiling_trace_flags_info_fn( (CTX), (KEY), (EVENT_ID), (TASKPOOL_ID), (INFO_FN), (INFO_FN_DATA), 0)

/**
 * @brief Trace one event on the implicit thread context.
 *
 * @details Event is added to the series of events related to the context passed as argument.
 *
 * @param[in] key     the key (as returned by add_dictionary_keyword) of the event to log
 * @param[in] event_id a (possibly unique) event identifier. Events are coupled together: start/end.
 *                      a couple (start, end) has
 *                        - the same key
 *                        - end is the next "end" event with the same key and the same non-null event_id and
 *                          non OBJECT_ID_NULL taskpool_id as start in the event buffer of the thread context
 *                        - if no matching end is found, this is an error
 * @param[in] taskpool_id unique object/handle identifier (use PROFILE_OBJECT_ID_NULL if N/A)
 * @param[in] info_fn a pointer to a function that will write the info of the event in the allocated event
 *                    that memory is of size defined during the creation of the event
 * @param[in] info_data an opaque pointer passed back to info_fn when it is called.
 * @param[in] flags   flags related to the event
 * @return 0 if success, negative otherwise.
 * @remark not thread safe (if two threads share a same thread_context. Safe per thread_context)
 */
int parsec_profiling_ts_trace_flags_info_fn(int key, uint64_t event_id, uint32_t taskpool_id,
                                            parsec_profiling_info_fn_t *info_fn, const void *info_data, uint16_t flags );

/**
 * @brief Creates the profile file given as a parameter to store the
 * next events.
 *
 * @details
 * The basename is always respected, even in the case where it points to another
 * directory.
 * @param[in] basefile the base name of the target file to create
 *                      the files actually created will be <basefile>-<process_id>.prof
 *                      with one file per process that calls this function. The
 *                      process_id is the one passed to @ref parsec_proflinig_init().
 * @param[in] hr_id   human readable global identifier associated with this
 *                      profile. This string is used to uniquely identify the experiment, 
 *                      and all processes calling this function must use the same string.
 * @return 0 if success, negative otherwise.
 * @remark not thread safe.
 */
int parsec_profiling_dbp_start( const char *basefile, const char *hr_id );

/**
 * @brief Dump the current profile
 * @details Completes the file opened with dbp_start.
 * Every single dbp_start should have a matching dbp_dump.
 *
 * @return 0 if success, negative otherwise
 * @remark not thread safe
 */
int parsec_profiling_dbp_dump( void );

/**
 * @brief Returns a char * (owned by parsec_profiling library)
 * that describes the last error that happened.
 * @details
 * @return NULL if no error happened before
 *         the char* of the error otherwise.
 * @remark not thread safe
 */
char *parsec_profiling_strerror(void);

/**
 *  @brief Returns the 64 bits, OS-specific timer for the current time
 *  @details
 *  @return 64 bits, OS-specific timer for the current time
 *  @remark thread safe
 */
uint64_t parsec_profiling_get_time(void);

/**
 * This structure used to describe events infos related to
 * data placement.
 * @remark Do not change this structure without changing
 *  appropriately the info profiling generation string below
 */
typedef struct {
    struct parsec_data_collection_s *desc;      /**< The pointer to the data collection used as a key to identify the collection */
    uint32_t                         data_id;   /**< The id of each data defines a unique element in the collection */
} parsec_profile_data_collection_info_t;

/**
 * @brief Convertor for parsec_profile_data_collection_info_t
 * @details This macro is the character string to convert a parsec_profile_data_collection_info_t into
 * meaningful numbers from the binary profile format. To be used in parsec_profiling_add_dictionary_keyword.
 */
#define PARSEC_PROFILE_DATA_COLLECTION_INFO_CONVERTOR "dc_key{uint64_t};dc_dataid{uint32_t};dc_padding{uint32_t}"

/**
 * per-task profiling information.
 * @remark we don't reuse parsec_profile_data_collection_info_t to avoid
 * alignment issues, but we re-use the same keywords to avoid creating
 * too many columns in the events dataframe.
 */
typedef struct {
    struct parsec_data_collection_s *desc;
    int32_t                          priority;
    uint32_t                         data_id;
    int32_t                          task_class_id;
    int32_t                          task_return_code;
} parsec_task_prof_info_t;

#define PARSEC_TASK_PROF_INFO_CONVERTOR "dc_key{uint64_t};priority{int32_t};dc_dataid{uint32_t};tcid{uint32_t};trc{int32_t}"

/**
 * @brief String used to identify GPU streams
 */
#define PARSEC_PROFILE_STREAM_STR "GPU %d-%d"
/**
 * @brief String used to identify CPU streams
 */
#define PARSEC_PROFILE_THREAD_STR "PaRSEC Thread %d of VP %d Bound on %s"

/**
 * @brief A boolean that is not 0 only if the profile is enabled and
 * @details
 * Externally visible on/off switch for the profiling of new events. It
 * only protects the macros, a direct call to the parsec_profiling_trace
 * will always succeed. It is automatically turned on by the init call.
 */
extern int parsec_profile_enabled;

/**
 * @brief Enable the profiling of new events.
 * @details
 * @remark not thread safe
 */
void parsec_profiling_enable(void);

/**
 * @brief Disable the profiling of new events.
 * @details
 * @remark not thread safe
 */
void parsec_profiling_disable(void);

/**
 * @brief Convenience macro to trace events only if profiling is enabled
 */
#define PARSEC_PROFILING_TRACE(context, key, event_id, object_id, info ) \
    if( parsec_profile_enabled ) {                                       \
        parsec_profiling_trace((context), (key), (event_id), (object_id), (info) ); \
    }

/**
 * @brief Convenience macro to trace events with flags only if profiling is enabled
 */
#define PARSEC_PROFILING_TRACE_FLAGS(context, key, event_id, object_id, info, flags ) \
    if( parsec_profile_enabled ) {                                       \
        parsec_profiling_trace_flags((context), (key), (event_id), (object_id), (info), (flags) ); \
    }

/**
 * @brief Convenience macro to trace events only if profiling is enabled
 */
#define PARSEC_PROFILING_TRACE_INFO_FN(context, key, event_id, object_id, info_fn, info_data ) \
    if( parsec_profile_enabled ) {                                       \
        parsec_profiling_trace_info_fn((context), (key), (event_id), (object_id), (info_fn), (info_data) ); \
    }

/**
 * @brief Convenience macro to trace events with flags only if profiling is enabled
 */
#define PARSEC_PROFILING_TRACE_FLAGS_INFO_FN(context, key, event_id, object_id, info_fn, info_data, flags ) \
    if( parsec_profile_enabled ) {                                       \
        parsec_profiling_trace_flags_info_fn((context), (key), (event_id), (object_id), (info_fn), (info_data), (flags) ); \
    }

/**
 * @brief Record a key/value pair in the profile with a double value
 *
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark not thread safe
 */
void profiling_save_dinfo(const char *key, double value);

/**
 * @brief Record a key/value pair in the profile with an integer value
 *
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark not thread safe
 */
void profiling_save_iinfo(const char *key, int value);

/**
 * @brief Record a key/value pair in the profile with a long long integer value
 *
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark not thread safe
 */
void profiling_save_uint64info(const char *key, unsigned long long int value);

/**
 * @brief Record a key/value pair in the profile with a string value
 *
 * @param[in] key the key to use in the key/value pair
 * @param[in] svalue the value to use in the key/value pair
 * @remark not thread safe
 */
void profiling_save_sinfo(const char *key, char* svalue);

/**
 * @brief Record a stream-specific key/value pair in the profile with a double value
 *
 * @param[in] stream the stream context to use
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark thread safe
 */
void profiling_stream_save_dinfo(parsec_profiling_stream_t* stream,
                                 const char *key, double value);

/**
 * @brief Record a stream-specific key/value pair in the profile with an integer value
 *
 * @param[in] stream the stream context to use
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark thread safe
 */
void profiling_stream_save_iinfo(parsec_profiling_stream_t* stream,
                                 const char *key, int value);

/**
 * @brief Record a stream-specific key/value pair in the profile with a long long integer value
 *
 * @param[in] stream the stream context to use
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark thread safe
 */
void profiling_stream_save_uint64info(parsec_profiling_stream_t* stream,
                                      const char *key, unsigned long long int value);

/**
 * @brief Record a stream-specific key/value pair in the profile with a string value
 *
 * @param[in] stream the stream context to use
 * @param[in] key the key to use in the key/value pair
 * @param[in] svalue the value to use in the key/value pair
 * @remark thread safe
 */
void profiling_stream_save_sinfo(parsec_profiling_stream_t* stream,
                                 const char *key, char* svalue);

/** @cond DONT_DOCUMENT */
#if defined(PARSEC_PROF_TRACE)
#define PROFILING_SAVE_dINFO(key, double_value) profiling_save_dinfo(key, double_value)
#define PROFILING_SAVE_iINFO(key, integer_value) profiling_save_iinfo(key, integer_value)
#define PROFILING_SAVE_uint64INFO(key, integer_value) profiling_save_uint64info(key, integer_value)
#define PROFILING_SAVE_sINFO(key, str_value) profiling_save_sinfo(key, str_value)
#define PROFILING_STREAM_SAVE_dINFO(stream, key, double_value)  \
    profiling_stream_save_dinfo(stream, key, double_value)
#define PROFILING_STREAM_SAVE_iINFO(stream, key, integer_value) \
    profiling_stream_save_iinfo(stream, key, integer_value)
#define PROFILING_STREAM_SAVE_uint64INFO(stream, key, integer_value) \
    profiling_stream_save_uint64info(stream, key, integer_value)
#define PROFILING_STREAM_SAVE_sINFO(stream, key, str_value)     \
    profiling_stream_save_sinfo(stream, key, str_value)
#else
#define PROFILING_SAVE_dINFO(key, double_value) do {} while(0)
#define PROFILING_SAVE_iINFO(key, integer_value) do {} while(0)
#define PROFILING_SAVE_uint64INFO(key, integer_value) do {} while(0)
#define PROFILING_SAVE_sINFO(key, str_value) do {} while(0)
#define PROFILING_STREAM_SAVE_dINFO(stream, key, double_value) do {} while(0)
#define PROFILING_STREAM_SAVE_iINFO(stream, key, integer_value) do {} while(0)
#define PROFILING_STREAM_SAVE_uint64INFO(stream, key, integer_value) do {} while(0)
#define PROFILING_STREAM_SAVE_sINFO(stream, key, str_value) do {} while(0)
#endif
/** @endcond */

#ifdef __cplusplus
}
#endif

/** @} */

#endif  /* _PARSEC_profiling_h */
