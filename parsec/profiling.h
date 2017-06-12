/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
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
 */

/**
 * @brief  Flag used when an info object is attached to the event
 */
#define PARSEC_PROFILING_EVENT_HAS_INFO     (1<<0)
/**
 * @brief Flag used when the event is a reschedule of a previous event
 */
#define PARSEC_PROFILING_EVENT_RESCHEDULED  (1<<1)
/**
 * @brief Flag used when the event's info is a counter
 * @details The event's info (if present) is an integer that
 *          should be accumulated to a value starting at 0.
 *          This might be useful to represent countable entities,
 *          like amount of tasks pending, amount of memory allocated,
 *          etc. */
#define PARSEC_PROFILING_EVENT_COUNTER      (1<<2)
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
typedef struct parsec_thread_profiling_s parsec_thread_profiling_t;

/**
 * @brief Initializes the profiling engine.
 * @details Call this ONCE per process.
 * @return 0    if success, -1 otherwise
 *
 * @remark not thread safe
 */
int parsec_profiling_init( void );

/**
 * @brief Set the reference time to now in the profiling system.
 * @details Optionally called before any even is traced.
 * @remark Not thread safe.
 */
void parsec_profiling_start(void);

/**
 * @brief Releases all resources for the tracing.
 * @details Thread contexts become invalid after this call.
 *          Must be called after the dbp_dump if a dbp_start was called.
 *
 * @return 0    if success, -1 otherwise.
 * @remark not thread safe
 */
int parsec_profiling_fini( void );

/**
 * @brief Removes all current logged events.
 * @details Prefer this to fini / init if you want
 * to do a new profiling with the same thread contexts. This does not
 * invalidate the current thread contexts.
 *
 * @return 0 if succes, -1 otherwise
 * not thread safe
 */
int parsec_profiling_reset( void );

/**
 * @brief Add additional information about the current run, under the form key/value.
 * @details Used to store the value of the globals names and values in the current run
 * @param[in] key key part of the key/value to store
 * @param[in] value value part of the key/value to store
 * @remark Not thread safe.
 */
void parsec_profiling_add_information( const char *key, const char *value );

/**
 * @brief Add additional information about the current run, under the form key/value.
 * @details This function adds key/value pairs PER THREAD, not globally.
 * @param[in] thread thread in which to store the key/value
 * @param[in] key key part of the key/value to store
 * @param[in] value value part of the key/value to store
 * @remark Not thread safe.
 */
void parsec_profiling_thread_add_information(parsec_thread_profiling_t * thread,
                                            const char *key, const char *value );

/**
 * @brief Initializes the buffer trace with the specified length.
 * @details This function must be called once per thread that will use the profiling
 * functions. This creates the profiling_thread_unit_t that must be passed to
 * the tracing function call. See note about thread safety.
 *
 * @param[in] length the length (in bytes) of the buffer queue to store events.
 * @param[in] format printf-like to associate a human-readable
 *                           definition of the calling thread
 * @return pointer to the new thread_profiling structure. NULL if an error.
 * @remark thread safe
 */
parsec_thread_profiling_t *parsec_profiling_thread_init( size_t length, const char *format, ...);

/**
 * @brief Inserts a new keyword in the dictionnary
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
 * @return 0    if success, -1 otherwie.
 * @remark not thread safe
 */
int parsec_profiling_add_dictionary_keyword( const char* name, const char* attributes,
                                            size_t info_length,
                                            const char* convertor_code,
                                            int* key_start, int* key_end );

/**
 * @brief Empties the global dictionnary
 * @degtails this might be usefull in conjunction with reset, if
 * you want to redo an experiment.
 *
 * @remark Emptying the dictionnary without reseting the profiling system will yield
 * undeterminate results
 *
 * @return 0 if success, -1 otherwise.
 * @remark not thread safe
 */
int parsec_profiling_dictionary_flush( void );

/**
 * @brief Trace one event
 * @details Event is added to the series of events related to the context passed as argument.
 *
 * @param[in] context a thread profiling context (should be the thread profiling context of the
 *                      calling thread).
 * @param[in] key     the key (as returned by add_dictionary_keyword) of the event to log
 * @param[in] event_id a (possibly unique) event identifier. Events are coupled together: start/end.
 *                      a couple (start, end) has
 *                        - the same key
 *                        - end is the next "end" event with the same key and the same non-null event_id and
 *                          non OBJECT_ID_NULL handle_id as start in the event buffer of the thread context
 *                        - if no matching end is found, this is an error
 * @param[in] handle_id unique object/handle identifier (use PROFILE_OBJECT_ID_NULL if N/A)
 * @param[in] info    a pointer to an area of size info_length for this key (see
 *                        parsec_profiling_add_dictionary_keyword)
 * @param[in] flags   flags related to the event
 * @return 0 if success, -1 otherwise.
 * @remark not thread safe (if two threads share a same thread_context. Safe per thread_context)
 */
int parsec_profiling_trace_flags(parsec_thread_profiling_t* context, int key,
                                 uint64_t event_id, uint32_t handle_id,
                                 void *info, uint16_t flags );

/**
 * @brief Convenience macro used to trace events without flags
 */
#define parsec_profiling_trace(CTX, KEY, EVENT_ID, HANDLE_ID, INFO)     \
    parsec_profiling_trace_flags( (CTX), (KEY), (EVENT_ID), (HANDLE_ID), (INFO), 0 )

/**
 * @brief Trace one event on the implicit thread context.
 * @details This uses a TLS variable to lookup the last context created on the calling
 * thread. If the calling thread did not create a a profiling context, this
 * will generate an error. This function is significantly more costly than
 * parsec_profiling_trace_flags, as it includes the cost of TLS lookup.
 *
 * @param[in] key     the key (as returned by add_dictionary_keyword) of the event to log
 * @param[in] event_id a (possibly unique) event identifier. Events are coupled together: start/end.
 *                      a couple (start, end) has
 *                        - the same key
 *                        - end is the next "end" event with the same key and the same non-null event_id and
 *                          non OBJECT_ID_NULL handle_id as start in the event buffer of the thread context
 *                        - if no matching end is found, this is an error
 * @param[in] handle_id unique object/handle identifier (use PROFILE_OBJECT_ID_NULL if N/A)
 * @param[in] info    a pointer to an area of size info_length for this key (see
 *                        parsec_profiling_add_dictionary_keyword)
 * @param[in] flags   flags related to the event
 * @return 0 if success, -1 otherwise.
 * @remark thread safe
 */
int parsec_profiling_ts_trace_flags(int key, uint64_t event_id, uint32_t object_id,
                                    void *info, uint16_t flags );

/**
 * @brief Convenience macro when no flag needs to be passed
 */
#define parsec_profiling_ts_trace(key, event_id, object_id, info) \
    parsec_profiling_ts_trace_flags((key), (event_id), (object_id), (info), 0)

/**
 * @brief Creates the profile file given as a parameter to store the
 * next events.
 * @details
 * Globally decide on a filename for the profiling file based on the requested
 * basefile, followed by the rank and then by a 6 letter unique key (generated
 * by mkstemp). The 6 letter key is used by all participants to create profiling
 * files that can be matched together.
 *
 * The basename is always respected, even in the case where it points to another
 * directory.
 * @param[in] basefile the base name of the target file to create
 *                       the file actually created will be <basefile>-%d.profile
 * @param[in] hr_info human readable global information associated with this
 *                      profile. Used "uniquely" identify the experiment, and
 *                      check that all separate profile files correspond to a same
 *                      experiment.
 * @return 0 if success, -1 otherwise.
 * @remark not thread safe.
 */
int parsec_profiling_dbp_start( const char *basefile, const char *hr_info );

/**
 * @brief Dump the current profile
 * @details Completes the file opened with dbp_start.
 * Every single dbp_start should have a matching dbp_dump.
 *
 * @return 0 if success, -1 otherwise
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
    struct parsec_ddesc_s *desc; /**< The pointer to the ddesc is used as a key to identify the collection */
    uint32_t              id;    /**< The id of each data defines a unique element in the collection */
} parsec_profile_ddesc_info_t;

/**
 * @brief Convertor for parsec_profile_ddesc_info_t
 * @details This macro is the character string to convert a parsec_profile_ddesc_info_t into
 * meaningful numbers from the binary profile format. To be used in parsec_profiling_add_dictionary_keyword.
 */
#define PARSEC_PROFILE_DDESC_INFO_CONVERTOR "ddesc_unique_key{uint64_t};ddesc_data_id{uint32_t};ddessc_padding{uint32_t}"
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
        parsec_profiling_trace(context, key, event_id, object_id, info ); \
    }

/**
 * @brief Convenience macro to trace events with flags only if profiling is enabled
 */
#define PARSEC_PROFILING_TRACE_FLAGS(context, key, event_id, object_id, info, flags ) \
    if( parsec_profile_enabled ) {                                       \
        parsec_profiling_trace_flags(context, key, event_id, object_id, info, flags ); \
    }

/**
 * @brief Record a key/value pair in the profile with a double value
 * @detail
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark not thread safe
 */
void profiling_save_dinfo(const char *key, double value);

/**
 * @brief Record a key/value pair in the profile with an integer value
 * @detail
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark not thread safe
 */
void profiling_save_iinfo(const char *key, int value);

/**
 * @brief Record a key/value pair in the profile with a long long integer value
 * @detail
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark not thread safe
 */
void profiling_save_uint64info(const char *key, unsigned long long int value);

/**
 * @brief Record a key/value pair in the profile with a string value
 * @detail
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark not thread safe
 */
void profiling_save_sinfo(const char *key, char* svalue);

/**
 * @brief Record a thread-specific key/value pair in the profile with a double value
 * @detail
 * @param[in] thread the thread context to use
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark thread safe
 */
void profiling_thread_save_dinfo(parsec_thread_profiling_t * thread,
                                 const char *key, double value);

/**
 * @brief Record a thread-specific key/value pair in the profile with an integer value
 * @detail
 * @param[in] thread the thread context to use
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark thread safe
 */
void profiling_thread_save_iinfo(parsec_thread_profiling_t * thread,
                                 const char *key, int value);

/**
 * @brief Record a thread-specific key/value pair in the profile with a long long integer value
 * @detail
 * @param[in] thread the thread context to use
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark thread safe
 */
void profiling_thread_save_uint64info(parsec_thread_profiling_t * thread,
                                      const char *key, unsigned long long int value);

/**
 * @brief Record a thread-specific key/value pair in the profile with a string value
 * @detail
 * @param[in] thread the thread context to use
 * @param[in] key the key to use in the key/value pair
 * @param[in] value the value to use in the key/value pair
 * @remark thread safe
 */
void profiling_thread_save_sinfo(parsec_thread_profiling_t * thread,
                                 const char *key, char* svalue);

/** @cond DONT_DOCUMENT */
#if defined(PARSEC_PROF_TRACE)
#define PROFILING_SAVE_dINFO(key, double_value) profiling_save_dinfo(key, double_value)
#define PROFILING_SAVE_iINFO(key, integer_value) profiling_save_iinfo(key, integer_value)
#define PROFILING_SAVE_sINFO(key, str_value) profiling_save_sinfo(key, str_value)
#define PROFILING_THREAD_SAVE_dINFO(thread, key, double_value)  \
    profiling_thread_save_dinfo(thread, key, double_value)
#define PROFILING_THREAD_SAVE_iINFO(thread, key, integer_value) \
    profiling_thread_save_iinfo(thread, key, integer_value)
#define PROFILING_THREAD_SAVE_uint64INFO(thread, key, integer_value) \
    profiling_thread_save_uint64info(thread, key, integer_value)
#define PROFILING_THREAD_SAVE_sINFO(thread, key, str_value)     \
    profiling_thread_save_sinfo(thread, key, str_value)
#else
#define PROFILING_SAVE_dINFO(key, double_value) do {} while(0)
#define PROFILING_SAVE_iINFO(key, integer_value) do {} while(0)
#define PROFILING_SAVE_sINFO(key, str_value) do {} while(0)
#define PROFILING_THREAD_SAVE_dINFO(thread, key, double_value) do {} while(0)
#define PROFILING_THREAD_SAVE_iINFO(thread, key, integer_value) do {} while(0)
#define PROFILING_THREAD_SAVE_uint64INFO(thread, key, integer_value) do {} while(0)
#define PROFILING_THREAD_SAVE_sINFO(thread, key, str_value) do {} while(0)
#endif
/** @endcond */

#ifdef __cplusplus
}
#endif

/** @} */

#endif  /* _PARSEC_profiling_h */
