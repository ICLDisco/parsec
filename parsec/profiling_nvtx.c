/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"

#if defined(PARSEC_PROF_TRACE_NVTX)

#include "parsec/profiling_nvtx.h"

#include <ctype.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <nvtx3/nvToolsExt.h>

#include "parsec/utils/mca_param.h"

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

static nvtxDomainHandle_t parsec_profiling_nvtx_domain = NULL;
static parsec_profiling_nvtx_key_t *parsec_profiling_nvtx_keys = NULL;
static int parsec_profiling_nvtx_keys_count = 0;
static int parsec_profiling_nvtx_keys_size = 0;
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

int parsec_profiling_nvtx_register_mca(void)
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

int parsec_profiling_nvtx_is_enabled(void)
{
    return parsec_profiling_nvtx_enabled;
}

void parsec_profiling_nvtx_init(int process_id)
{
    (void)process_id;

    parsec_profiling_nvtx_register_mca();
    if( !parsec_profiling_nvtx_enabled ) {
        return;
    }

    parsec_profiling_nvtx_domain = nvtxDomainCreateA("PaRSEC");
}

void parsec_profiling_nvtx_dictionary_flush(void)
{
    for(int i = 0; i < parsec_profiling_nvtx_keys_count; i++) {
        free(parsec_profiling_nvtx_keys[i].name);
    }
    free(parsec_profiling_nvtx_keys);
    parsec_profiling_nvtx_keys = NULL;
    parsec_profiling_nvtx_keys_count = 0;
    parsec_profiling_nvtx_keys_size = 0;
}

void parsec_profiling_nvtx_fini(void)
{
    parsec_profiling_nvtx_dictionary_flush();
    if( NULL != parsec_profiling_nvtx_domain ) {
        nvtxDomainDestroy(parsec_profiling_nvtx_domain);
        parsec_profiling_nvtx_domain = NULL;
    }
}

void parsec_profiling_nvtx_register_key(int key, const char *name,
                                        const char *attributes)
{
    char *new_name;

    if( !parsec_profiling_nvtx_enabled ||
        NULL == parsec_profiling_nvtx_domain ||
        key < 0 ) {
        return;
    }

    if( key >= parsec_profiling_nvtx_keys_size ) {
        int old_size = parsec_profiling_nvtx_keys_size;
        int new_size = (0 == old_size) ? 128 : old_size;
        parsec_profiling_nvtx_key_t *new_keys;
        while( key >= new_size ) {
            new_size *= 2;
        }
        new_keys = realloc(parsec_profiling_nvtx_keys,
                           (size_t)new_size * sizeof(parsec_profiling_nvtx_key_t));
        if( NULL == new_keys ) {
            return;
        }
        parsec_profiling_nvtx_keys = new_keys;
        memset(&parsec_profiling_nvtx_keys[old_size], 0,
               (size_t)(new_size - old_size) * sizeof(parsec_profiling_nvtx_key_t));
        parsec_profiling_nvtx_keys_size = new_size;
    }
    if( key >= parsec_profiling_nvtx_keys_count ) {
        parsec_profiling_nvtx_keys_count = key + 1;
    }

    new_name = strdup((NULL == name) ? "PaRSEC event" : name);
    if( NULL == new_name ) {
        return;
    }
    free(parsec_profiling_nvtx_keys[key].name);
    parsec_profiling_nvtx_keys[key].name = new_name;
    parsec_profiling_nvtx_keys[key].color =
        parsec_profiling_nvtx_parse_color(attributes, key);
    nvtxDomainNameCategoryA(parsec_profiling_nvtx_domain,
                            (uint32_t)key,
                            parsec_profiling_nvtx_keys[key].name);
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

void parsec_profiling_nvtx_release_stream(parsec_profiling_nvtx_range_t **active,
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

void parsec_profiling_nvtx_trace(parsec_profiling_nvtx_range_t **active,
                                 parsec_profiling_nvtx_range_t **freelist,
                                 int key, int is_start,
                                 uint64_t event_id, uint32_t taskpool_id)
{
    parsec_profiling_nvtx_range_t *range;
    nvtxEventAttributes_t event_attrib;

    if( !parsec_profiling_nvtx_enabled ||
        NULL == parsec_profiling_nvtx_domain ||
        key < 0 ||
        key >= parsec_profiling_nvtx_keys_count ||
        NULL == parsec_profiling_nvtx_keys[key].name ) {
        return;
    }

    if( is_start ) {
        memset(&event_attrib, 0, sizeof(event_attrib));
        event_attrib.version = NVTX_VERSION;
        event_attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        event_attrib.category = (uint32_t)key;
        event_attrib.colorType = NVTX_COLOR_ARGB;
        event_attrib.color = parsec_profiling_nvtx_keys[key].color;
        event_attrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        event_attrib.message.ascii = parsec_profiling_nvtx_keys[key].name;
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

#endif /* defined(PARSEC_PROF_TRACE_NVTX) */
