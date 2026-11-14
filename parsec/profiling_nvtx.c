/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"

#if defined(PARSEC_PROF_TRACE_NVTX)

#include "parsec/profiling_internal.h"

#include <ctype.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <nvtx3/nvToolsExt.h>

#include "parsec/sys/atomic.h"
#include "parsec/utils/mca_param.h"

typedef struct parsec_profiling_nvtx_range_s parsec_profiling_nvtx_range_t;

struct parsec_profiling_nvtx_range_s {
    struct parsec_profiling_nvtx_range_s *next;
    struct parsec_profiling_nvtx_range_s *global_next;
    struct parsec_profiling_nvtx_range_s **owner_active;
    int                                   key;
    uint64_t                              event_id;
    uint32_t                              taskpool_id;
    nvtxRangeId_t                         range_id;
};

typedef struct parsec_profiling_nvtx_key_s {
    char    *name;
    uint32_t color;
} parsec_profiling_nvtx_key_t;

/* Keys indexed by the dictionary key. The table is two-level and chunks are
 * never reallocated, so that tracing threads can read it while new keys are
 * registered. */
#define PARSEC_PROFILING_NVTX_KEYS_CHUNK      256
#define PARSEC_PROFILING_NVTX_KEYS_MAX_CHUNKS 128  /* the dictionary has at most 1<<15 keys */

static nvtxDomainHandle_t parsec_profiling_nvtx_domain = NULL;
static parsec_profiling_nvtx_key_t *parsec_profiling_nvtx_keys[PARSEC_PROFILING_NVTX_KEYS_MAX_CHUNKS];
static int parsec_profiling_nvtx_enabled = 0;
static int parsec_profiling_nvtx_mca_registered = 0;
static parsec_profiling_nvtx_range_t *parsec_profiling_nvtx_active_ranges = NULL;
static pthread_mutex_t parsec_profiling_nvtx_ranges_lock = PTHREAD_MUTEX_INITIALIZER;

static uint32_t parsec_profiling_nvtx_fallback_color(int key)
{
    static const uint32_t colors[] = {
        0xff4e79a7, 0xfff28e2b, 0xffe15759, 0xff76b7b2,
        0xff59a14f, 0xffedc949, 0xffaf7aa1, 0xffff9da7,
        0xff9c755f, 0xffbab0ab
    };
    return colors[(unsigned int)key % (sizeof(colors) / sizeof(colors[0]))];
}

static int parsec_profiling_nvtx_hex(char c)
{
    if( c >= '0' && c <= '9' ) return c - '0';
    c = (char)tolower((unsigned char)c);
    if( c >= 'a' && c <= 'f' ) return c - 'a' + 10;
    return -1;
}

static uint32_t parsec_profiling_nvtx_parse_color(const char *attributes, int key)
{
    const char *c;
    uint32_t color = 0;

    if( NULL == attributes ) {
        return parsec_profiling_nvtx_fallback_color(key);
    }

    c = strchr(attributes, '#');
    if( NULL == c ) {
        return parsec_profiling_nvtx_fallback_color(key);
    }
    c++;
    for(int i = 0; i < 6; i++) {  /* Reading a hex RGB */
        if( '\0'== c[i] ) {
            return parsec_profiling_nvtx_fallback_color(key);
        }
        int v = parsec_profiling_nvtx_hex(c[i]);
        if( v < 0 ) {
            return parsec_profiling_nvtx_fallback_color(key);
        }
        color = (color << 4) | (uint32_t)v;
    }
    return 0xff000000 | color;
}

static int parsec_profiling_nvtx_register_mca(void)
{
    if( !parsec_profiling_nvtx_mca_registered ) {
        parsec_mca_param_reg_int_name("profile", "nvtx",
                                      "Enable the NVTX profiling substrate and mirror PaRSEC profiling events as ranges for NVIDIA Nsight Systems",
                                      false, false,
                                      parsec_profiling_nvtx_enabled,
                                      &parsec_profiling_nvtx_enabled);
        parsec_profiling_nvtx_mca_registered = 1;
    }
    return parsec_profiling_nvtx_enabled;
}

static int parsec_profiling_nvtx_init(int process_id)
{
    (void)process_id;

    parsec_profiling_nvtx_register_mca();
    if( !parsec_profiling_nvtx_enabled ) {
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    parsec_profiling_nvtx_domain = nvtxDomainCreateA("PaRSEC");
    return (NULL != parsec_profiling_nvtx_domain) ? 0 : PARSEC_ERROR;
}

/* The key, or NULL if no key was registered at this position */
static parsec_profiling_nvtx_key_t *parsec_profiling_nvtx_get_key(int key)
{
    parsec_profiling_nvtx_key_t *chunk;
    int c = key / PARSEC_PROFILING_NVTX_KEYS_CHUNK;

    if( key < 0 || c >= PARSEC_PROFILING_NVTX_KEYS_MAX_CHUNKS ) {
        return NULL;
    }
    chunk = parsec_profiling_nvtx_keys[c];
    if( NULL == chunk || NULL == chunk[key % PARSEC_PROFILING_NVTX_KEYS_CHUNK].name ) {
        return NULL;
    }
    return &chunk[key % PARSEC_PROFILING_NVTX_KEYS_CHUNK];
}

static void parsec_profiling_nvtx_dictionary_flush(void)
{
    for(int c = 0; c < PARSEC_PROFILING_NVTX_KEYS_MAX_CHUNKS; c++) {
        if( NULL == parsec_profiling_nvtx_keys[c] ) {
            continue;
        }
        for(int i = 0; i < PARSEC_PROFILING_NVTX_KEYS_CHUNK; i++) {
            free(parsec_profiling_nvtx_keys[c][i].name);
            parsec_profiling_nvtx_keys[c][i].name = NULL;
        }
    }
}

static void parsec_profiling_nvtx_fini(void)
{
    parsec_profiling_nvtx_dictionary_flush();
    for(int c = 0; c < PARSEC_PROFILING_NVTX_KEYS_MAX_CHUNKS; c++) {
        free(parsec_profiling_nvtx_keys[c]);
        parsec_profiling_nvtx_keys[c] = NULL;
    }
    if( NULL != parsec_profiling_nvtx_domain ) {
        nvtxDomainDestroy(parsec_profiling_nvtx_domain);
        parsec_profiling_nvtx_domain = NULL;
    }
}

static void parsec_profiling_nvtx_register_key(int key, const char *name,
                                               const char *attributes)
{
    parsec_profiling_nvtx_key_t *entry;
    char *new_name, *old_name;
    int c = key / PARSEC_PROFILING_NVTX_KEYS_CHUNK;

    if( !parsec_profiling_nvtx_enabled ||
        NULL == parsec_profiling_nvtx_domain ||
        key < 0 || c >= PARSEC_PROFILING_NVTX_KEYS_MAX_CHUNKS ) {
        return;
    }

    /* Keys are registered with the dictionary lock held */
    if( NULL == parsec_profiling_nvtx_keys[c] ) {
        parsec_profiling_nvtx_keys[c] = (parsec_profiling_nvtx_key_t*)
            calloc(PARSEC_PROFILING_NVTX_KEYS_CHUNK, sizeof(parsec_profiling_nvtx_key_t));
        if( NULL == parsec_profiling_nvtx_keys[c] ) {
            return;
        }
    }
    entry = &parsec_profiling_nvtx_keys[c][key % PARSEC_PROFILING_NVTX_KEYS_CHUNK];

    new_name = strdup((NULL == name) ? "PaRSEC event" : name);
    if( NULL == new_name ) {
        return;
    }
    old_name = entry->name;
    entry->color = parsec_profiling_nvtx_parse_color(attributes, key);
    /* The entry is complete before it becomes visible through its name */
    parsec_mfence();
    entry->name = new_name;
    free(old_name);
    nvtxDomainNameCategoryA(parsec_profiling_nvtx_domain,
                            (uint32_t)key,
                            entry->name);
}

static parsec_profiling_nvtx_range_t*
parsec_profiling_nvtx_range_alloc(parsec_profiling_nvtx_range_t **freelist)
{
    parsec_profiling_nvtx_range_t *range = *freelist;
    if( NULL != range ) {
        *freelist = range->next;
        return range;
    }
    return (parsec_profiling_nvtx_range_t*)malloc(sizeof(*range));
}

static void
parsec_profiling_nvtx_range_free(parsec_profiling_nvtx_range_t **freelist,
                                 parsec_profiling_nvtx_range_t *range)
{
    range->global_next = NULL;
    range->owner_active = NULL;
    range->next = *freelist;
    *freelist = range;
}

static void
parsec_profiling_nvtx_unlink_global_locked(parsec_profiling_nvtx_range_t *range)
{
    parsec_profiling_nvtx_range_t *cur = parsec_profiling_nvtx_active_ranges;
    parsec_profiling_nvtx_range_t *prev = NULL;

    while( NULL != cur ) {
        if( cur == range ) {
            if( NULL == prev ) {
                parsec_profiling_nvtx_active_ranges = cur->global_next;
            } else {
                prev->global_next = cur->global_next;
            }
            range->global_next = NULL;
            return;
        }
        prev = cur;
        cur = cur->global_next;
    }
}

static void
parsec_profiling_nvtx_unlink_owner_locked(parsec_profiling_nvtx_range_t *range)
{
    parsec_profiling_nvtx_range_t **cur;

    if( NULL == range->owner_active ) {
        return;
    }

    cur = range->owner_active;
    while( NULL != *cur ) {
        if( *cur == range ) {
            *cur = range->next;
            break;
        }
        cur = &((*cur)->next);
    }
    range->next = NULL;
    range->owner_active = NULL;
}

static void parsec_profiling_nvtx_release_stream(parsec_profiling_nvtx_range_t **active,
                                                 parsec_profiling_nvtx_range_t **freelist)
{
    parsec_profiling_nvtx_range_t *range, *to_end = NULL;

    pthread_mutex_lock(&parsec_profiling_nvtx_ranges_lock);
    while( NULL != *active ) {
        range = *active;
        *active = range->next;
        parsec_profiling_nvtx_unlink_global_locked(range);
        range->next = to_end;
        range->owner_active = NULL;
        to_end = range;
    }
    pthread_mutex_unlock(&parsec_profiling_nvtx_ranges_lock);

    while( NULL != to_end ) {
        range = to_end;
        to_end = range->next;
        if( NULL != parsec_profiling_nvtx_domain ) {
            nvtxDomainRangeEnd(parsec_profiling_nvtx_domain, range->range_id);
        }
        free(range);
    }
    while( NULL != *freelist ) {
        range = *freelist;
        *freelist = range->next;
        free(range);
    }
}

static int parsec_profiling_nvtx_same_range(parsec_profiling_nvtx_range_t *range,
                                            int key,
                                            uint64_t event_id,
                                            uint32_t taskpool_id)
{
    if( range->key != key || range->event_id != event_id ) {
        return 0;
    }
    return (range->taskpool_id == taskpool_id) ||
           (range->taskpool_id == PROFILE_OBJECT_ID_NULL) ||
           (taskpool_id == PROFILE_OBJECT_ID_NULL);
}

static parsec_profiling_nvtx_range_t*
parsec_profiling_nvtx_find_active_locked(int key,
                                         uint64_t event_id,
                                         uint32_t taskpool_id)
{
    parsec_profiling_nvtx_range_t *range = parsec_profiling_nvtx_active_ranges;

    while( NULL != range ) {
        if( parsec_profiling_nvtx_same_range(range, key, event_id, taskpool_id) ) {
            parsec_profiling_nvtx_unlink_global_locked(range);
            parsec_profiling_nvtx_unlink_owner_locked(range);
            return range;
        }
        range = range->global_next;
    }
    return NULL;
}

static void parsec_profiling_nvtx_trace(parsec_profiling_nvtx_range_t **active,
                                        parsec_profiling_nvtx_range_t **freelist,
                                        int key, int is_start,
                                        uint64_t event_id, uint32_t taskpool_id)
{
    parsec_profiling_nvtx_range_t *range;
    parsec_profiling_nvtx_key_t *entry;
    nvtxEventAttributes_t event_attrib;

    if( !parsec_profiling_nvtx_enabled ||
        NULL == parsec_profiling_nvtx_domain ||
        NULL == (entry = parsec_profiling_nvtx_get_key(key)) ) {
        return;
    }

    if( is_start ) {
        memset(&event_attrib, 0, sizeof(event_attrib));
        event_attrib.version = NVTX_VERSION;
        event_attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        event_attrib.category = (uint32_t)key;
        event_attrib.colorType = NVTX_COLOR_ARGB;
        event_attrib.color = entry->color;
        event_attrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        event_attrib.message.ascii = entry->name;
        event_attrib.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        event_attrib.payload.ullValue = event_id;

        range = parsec_profiling_nvtx_range_alloc(freelist);
        if( NULL == range ) {
            return;
        }
        range->key = key;
        range->event_id = event_id;
        range->taskpool_id = taskpool_id;
        range->range_id = nvtxDomainRangeStartEx(parsec_profiling_nvtx_domain,
                                                 &event_attrib);
        pthread_mutex_lock(&parsec_profiling_nvtx_ranges_lock);
        range->owner_active = active;
        range->next = *active;
        *active = range;
        range->global_next = parsec_profiling_nvtx_active_ranges;
        parsec_profiling_nvtx_active_ranges = range;
        pthread_mutex_unlock(&parsec_profiling_nvtx_ranges_lock);
        return;
    }

    pthread_mutex_lock(&parsec_profiling_nvtx_ranges_lock);
    range = parsec_profiling_nvtx_find_active_locked(key, event_id, taskpool_id);
    pthread_mutex_unlock(&parsec_profiling_nvtx_ranges_lock);
    if( NULL != range ) {
        nvtxDomainRangeEnd(parsec_profiling_nvtx_domain, range->range_id);
        parsec_profiling_nvtx_range_free(freelist, range);
    }
}

/* Substrate interface */

typedef struct parsec_profiling_nvtx_stream_s {
    parsec_profiling_nvtx_range_t *ranges;
    parsec_profiling_nvtx_range_t *freelist;
} parsec_profiling_nvtx_stream_t;

static void nvtx_register_key(int key, const parsec_profiling_key_t *entry)
{
    parsec_profiling_nvtx_register_key(key, entry->name, entry->attributes);
}

static int nvtx_stream_init(parsec_profiling_stream_t *stream, void **state)
{
    (void)stream;
    *state = calloc(1, sizeof(parsec_profiling_nvtx_stream_t));
    if( NULL == *state ) {
        parsec_profiling_set_last_error("Profiling system: NVTX: unable to allocate %zu bytes",
                                        sizeof(parsec_profiling_nvtx_stream_t));
        return PARSEC_ERR_OUT_OF_RESOURCE;
    }
    return 0;
}

static void nvtx_stream_reset(parsec_profiling_stream_t *stream, void *state)
{
    parsec_profiling_nvtx_stream_t *s = (parsec_profiling_nvtx_stream_t*)state;
    (void)stream;
    parsec_profiling_nvtx_release_stream(&s->ranges, &s->freelist);
}

static void nvtx_stream_fini(parsec_profiling_stream_t *stream, void *state)
{
    nvtx_stream_reset(stream, state);
    free(state);
}

static int nvtx_trace(parsec_profiling_stream_t *stream, void *state,
                      const parsec_profiling_trace_event_t *event)
{
    parsec_profiling_nvtx_stream_t *s = (parsec_profiling_nvtx_stream_t*)state;
    (void)stream;
    /* Counter events carry values to accumulate, they are not ranges */
    if( NULL != s && !(event->flags & PARSEC_PROFILING_EVENT_COUNTER) ) {
        parsec_profiling_nvtx_trace(&s->ranges, &s->freelist,
                                    event->key, event->is_start,
                                    event->event_id, event->taskpool_id);
    }
    return 0;
}

parsec_profiling_substrate_t parsec_profiling_substrate_nvtx = {
    .name             = "nvtx",
    .kind             = PARSEC_PROFILING_SUBSTRATE_MIRROR,
    .register_mca     = parsec_profiling_nvtx_register_mca,
    .init             = parsec_profiling_nvtx_init,
    .fini             = parsec_profiling_nvtx_fini,
    .register_key     = nvtx_register_key,
    .dictionary_flush = parsec_profiling_nvtx_dictionary_flush,
    .stream_init      = nvtx_stream_init,
    .stream_reset     = nvtx_stream_reset,
    .stream_fini      = nvtx_stream_fini,
    .trace            = nvtx_trace,
};

#endif /* defined(PARSEC_PROF_TRACE_NVTX) */
