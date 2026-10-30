/*
 * Copyright (c) 2026      Stony Brook University.  All rights reserved.
 */

#ifndef PARSEC_PROFILING_INTERNAL_H_HAS_BEEN_INCLUDED
#define PARSEC_PROFILING_INTERNAL_H_HAS_BEEN_INCLUDED

/**
 * @addtogroup parsec_internal_profiling
 * @{
 *
 * The profiling system is split in a core and a set of substrates. The core
 * (profiling.c) implements the public profiling API: it owns the dictionary,
 * the streams, the information key/values and the time base, and forwards
 * the events to every active substrate.
 *
 * A substrate is either a file substrate (the PaRSEC binary format, OTF2),
 * selected with the profile_format MCA parameter and activated by
 * parsec_profiling_dbp_start, or a mirror substrate (NVTX, Score-P), that
 * replays the events into an external tool and is activated at
 * parsec_profiling_init when requested through its own MCA parameter.
 */

#include "parsec/parsec_config.h"

#include <stddef.h>
#include <stdint.h>

#include "parsec/profiling.h"
#include "parsec/class/list.h"
#include "parsec/os-spec-timing.h"

BEGIN_C_DECLS

/** A key/value information, attached to the process or to a stream */
typedef struct parsec_profiling_info_s {
    struct parsec_profiling_info_s *next;
    char                           *key;
    char                           *value;
} parsec_profiling_info_t;

/** A dictionary entry, the start and end keys of the user are derived from
 *  its index with START_KEY and END_KEY. */
typedef struct parsec_profiling_key_s {
    char    *name;
    char    *attributes;
    char    *convertor;
    int32_t  info_length;
} parsec_profiling_key_t;

#define PARSEC_PROFILING_MAX_SUBSTRATES 8

struct parsec_profiling_stream_s {
    parsec_list_item_t       super;
    char                    *hr_id;
    parsec_profiling_info_t *infos;
    void                    *substrate_data[PARSEC_PROFILING_MAX_SUBSTRATES];
};

/** One event, as given to the substrates */
typedef struct parsec_profiling_trace_event_s {
    int                         key;          /**< index of the dictionary entry */
    int                         is_start;
    int                         is_default_stream; /**< the stream is the default stream of the calling thread */
    uint16_t                    flags;
    uint32_t                    taskpool_id;
    uint64_t                    event_id;
    parsec_profiling_info_fn_t *info_fn;
    const void                 *info_data;
    parsec_time_t               start_time;   /**< taken at the beginning of the trace call,
                                               *   used when PARSEC_PROFILING_EVENT_TIME_AT_START is set */
} parsec_profiling_trace_event_t;

#define PARSEC_PROFILING_SUBSTRATE_FILE   0
#define PARSEC_PROFILING_SUBSTRATE_MIRROR 1

typedef struct parsec_profiling_substrate_s parsec_profiling_substrate_t;

/**
 * The substrate interface. All functions are optional, except trace.
 */
struct parsec_profiling_substrate_s {
    const char *name;
    int         kind;   /**< PARSEC_PROFILING_SUBSTRATE_FILE or _MIRROR */
    int         index;  /**< set by the core, index of the substrate state in the streams */

    /** Register the MCA parameters. For mirror substrates, returns non-zero
     *  if the substrate is requested. */
    int  (*register_mca)(void);
    /** Prepare the substrate, called from parsec_profiling_init for the
     *  selected file substrates and the requested mirror substrates.
     *  Mirror substrates are active once this returns 0. */
    int  (*init)(int process_id);
    /** Activate a file substrate, called from parsec_profiling_dbp_start */
    int  (*open)(const char *basefile, const char *hr_info);
    /** Called from parsec_profiling_start, before the time base is set */
    void (*start)(void);
    /** Complete and close the output of a file substrate */
    int  (*dump)(void);
    void (*fini)(void);

    /** A new dictionary entry. Called with the dictionary lock held. */
    void (*register_key)(int key, const parsec_profiling_key_t *entry);
    void (*dictionary_flush)(void);

    int  (*stream_init)(parsec_profiling_stream_t *stream, void **state);
    void (*stream_reset)(parsec_profiling_stream_t *stream, void *state);
    void (*stream_fini)(parsec_profiling_stream_t *stream, void *state);

    int  (*trace)(parsec_profiling_stream_t *stream, void *state,
                  const parsec_profiling_trace_event_t *event);
    /** Called by every thread that traced events before it terminates */
    void (*thread_fini)(void);
};

extern parsec_profiling_substrate_t parsec_profiling_substrate_binary;
#if defined(PARSEC_HAVE_OTF2)
extern parsec_profiling_substrate_t parsec_profiling_substrate_otf2;
#endif
#if defined(PARSEC_PROF_TRACE_NVTX)
extern parsec_profiling_substrate_t parsec_profiling_substrate_nvtx;
#endif
#if defined(PARSEC_PROF_TRACE_SCOREP)
extern parsec_profiling_substrate_t parsec_profiling_substrate_scorep;
#endif

/* Services of the core to the substrates */

void parsec_profiling_set_last_error(const char *format, ...);
int  parsec_profiling_is_started(void);
/** Time elapsed between the start of the profiling and @p t */
uint64_t parsec_profiling_timestamp(parsec_time_t t);
/** Number of dictionary entries, entries may be NULL */
int  parsec_profiling_dictionary_count(void);
const parsec_profiling_key_t *parsec_profiling_dictionary_entry(int key);
/** Global information list (most recent first) */
parsec_profiling_info_t *parsec_profiling_global_infos(void);
/** All the profiling streams */
parsec_list_t *parsec_profiling_streams(void);
parsec_profiling_stream_t *parsec_profiling_get_default_stream(void);

static inline void *
parsec_profiling_stream_state(const parsec_profiling_stream_t *stream,
                              const parsec_profiling_substrate_t *substrate)
{
    return stream->substrate_data[substrate->index];
}

/* Services of the core to the runtime */

/** Returns non-zero if at least one substrate is active */
int  parsec_profiling_is_active(void);
/** Must be called by every thread that traced events before it terminates */
void parsec_profiling_thread_fini(void);

END_C_DECLS

/** @} */

#endif /* PARSEC_PROFILING_INTERNAL_H_HAS_BEEN_INCLUDED */
