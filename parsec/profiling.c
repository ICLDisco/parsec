/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      Stony Brook University.  All rights reserved.
 */

/*
 * Core of the profiling system: implements the public profiling API and
 * forwards the events to the active substrates (see profiling_internal.h).
 */

#include "parsec/parsec_config.h"

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>

#include "parsec/profiling.h"
#include "parsec/profiling_internal.h"
#include "parsec/parsec_binary_profile.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/sys/atomic.h"
#include "parsec/sys/tls.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/mca_param.h"

#ifndef HOST_NAME_MAX
#if defined(__APPLE__)
#define HOST_NAME_MAX _SC_HOST_NAME_MAX
#else
#define HOST_NAME_MAX 1024
#endif  /* defined(__APPLE__) */
#endif /* defined(HOST_NAME_MAX) */

#if defined(PARSEC_HAVE_OTF2)
#define PARSEC_PROFILING_DEFAULT_FORMAT "otf2"
#else
#define PARSEC_PROFILING_DEFAULT_FORMAT "binary"
#endif

int parsec_profile_enabled = 0;
static int __profile_initialized = 0;  /* not initialized */
static int parsec_prof_warning_issued = 0; /* no warning was issued */

static parsec_time_t parsec_start_time;
static int           start_called = 0;

/* All the substrates built in, in the order in which they receive the events */
static parsec_profiling_substrate_t *parsec_profiling_substrates[] = {
    &parsec_profiling_substrate_binary,
#if defined(PARSEC_HAVE_OTF2)
    &parsec_profiling_substrate_otf2,
#endif
#if defined(PARSEC_PROF_TRACE_NVTX)
    &parsec_profiling_substrate_nvtx,
#endif
#if defined(PARSEC_PROF_TRACE_SCOREP)
    &parsec_profiling_substrate_scorep,
#endif
};
#define NB_SUBSTRATES ((int)(sizeof(parsec_profiling_substrates) / sizeof(parsec_profiling_substrates[0])))

#define SUBSTRATE_UNUSED   0  /* not selected, or failed to initialize */
#define SUBSTRATE_PREPARED 1  /* initialized file substrate, waiting for parsec_profiling_dbp_start */
#define SUBSTRATE_ACTIVE   2  /* receives the events */
#define SUBSTRATE_CLOSED   3  /* file substrate dumped, does not receive events anymore */
static int substrate_status[NB_SUBSTRATES];

/* The active substrates, read on the critical path of the trace */
static parsec_profiling_substrate_t *active_substrates[NB_SUBSTRATES];
static int nb_active_substrates = 0;

/* Process-global dictionary. Chunks are never reallocated, so that tracing
 * threads can read the entries while new keys are registered. */
#define PARSEC_PROFILING_KEYS_CHUNK 128
/* The binary format stores the start/end keys on 16 bits */
#define PARSEC_PROFILING_KEYS_MAX   (1 << 15)
static parsec_profiling_key_t *parsec_prof_keys[PARSEC_PROFILING_KEYS_MAX / PARSEC_PROFILING_KEYS_CHUNK];
static int parsec_prof_keys_count = 0;
static int parsec_prof_keys_number = PARSEC_PROFILING_KEYS_MAX;
static pthread_mutex_t profiling_keyword_lock = PTHREAD_MUTEX_INITIALIZER;

/* Process-global profiling streams and information */
static parsec_list_t streams;
static parsec_profiling_info_t *parsec_profiling_infos = NULL;
static PARSEC_TLS_DECLARE(tls_profiling);

#define MAX_PROFILING_ERROR_STRING_LEN 1024
static char  parsec_profiling_last_error[MAX_PROFILING_ERROR_STRING_LEN+1] = { '\0', };
static int   parsec_profiling_raise_error = 0;

void parsec_profiling_set_last_error(const char *format, ...)
{
    va_list ap;
    int rc;
    va_start(ap, format);
    rc = vsnprintf(parsec_profiling_last_error, MAX_PROFILING_ERROR_STRING_LEN, format, ap);
    va_end(ap);
    fprintf(stderr, "-- %s", parsec_profiling_last_error);
    parsec_profiling_raise_error = 1;
    (void)rc;
}

static void reset_last_error(void)
{
    snprintf(parsec_profiling_last_error, MAX_PROFILING_ERROR_STRING_LEN, "Profiling system: success");
    parsec_profiling_raise_error = 0;
}

char *parsec_profiling_strerror(void)
{
    return parsec_profiling_last_error;
}

int parsec_profiling_is_started(void)
{
    return start_called;
}

int parsec_profiling_is_active(void)
{
    return nb_active_substrates > 0;
}

uint64_t parsec_profiling_timestamp(parsec_time_t t)
{
    return diff_time(parsec_start_time, t);
}

uint64_t parsec_profiling_get_time(void)
{
    return diff_time(parsec_start_time, take_time());
}

parsec_list_t *parsec_profiling_streams(void)
{
    return &streams;
}

parsec_profiling_stream_t *parsec_profiling_get_default_stream(void)
{
    return PARSEC_TLS_GET_SPECIFIC(tls_profiling);
}

parsec_profiling_info_t *parsec_profiling_global_infos(void)
{
    return parsec_profiling_infos;
}

int parsec_profiling_dictionary_count(void)
{
    return parsec_prof_keys_count;
}

static parsec_profiling_key_t *dictionary_slot(int key)
{
    parsec_profiling_key_t *chunk = parsec_prof_keys[key / PARSEC_PROFILING_KEYS_CHUNK];
    if( NULL == chunk ) {
        return NULL;
    }
    return &chunk[key % PARSEC_PROFILING_KEYS_CHUNK];
}

const parsec_profiling_key_t *parsec_profiling_dictionary_entry(int key)
{
    parsec_profiling_key_t *entry;
    if( key < 0 || key >= parsec_prof_keys_count ) {
        return NULL;
    }
    entry = dictionary_slot(key);
    if( NULL == entry || NULL == entry->name ) {
        return NULL;
    }
    return entry;
}

/* Make a substrate receive the events, after giving it the keys and the
 * streams created before its activation. */
static void activate_substrate(int s)
{
    parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
    parsec_list_item_t *it;

    if( NULL != sub->register_key ) {
        pthread_mutex_lock(&profiling_keyword_lock);
        for( int k = 0; k < parsec_prof_keys_count; k++ ) {
            const parsec_profiling_key_t *entry = parsec_profiling_dictionary_entry(k);
            if( NULL != entry ) {
                sub->register_key(k, entry);
            }
        }
        pthread_mutex_unlock(&profiling_keyword_lock);
    }
    if( NULL != sub->stream_init ) {
        parsec_list_lock(&streams);
        for( it = PARSEC_LIST_ITERATOR_FIRST(&streams);
             it != PARSEC_LIST_ITERATOR_END(&streams);
             it = PARSEC_LIST_ITERATOR_NEXT(it) ) {
            parsec_profiling_stream_t *stream = (parsec_profiling_stream_t*)it;
            if( 0 != sub->stream_init(stream, &stream->substrate_data[s]) ) {
                parsec_warning("Profiling system: substrate %s could not attach to stream %s, its events will be ignored.",
                               sub->name, stream->hr_id);
                stream->substrate_data[s] = NULL;
            }
        }
        parsec_list_unlock(&streams);
    }
    substrate_status[s] = SUBSTRATE_ACTIVE;
    active_substrates[nb_active_substrates] = sub;
    parsec_mfence();
    nb_active_substrates++;
}

static void deactivate_substrate(int s)
{
    int i, j;
    for( i = j = 0; i < nb_active_substrates; i++ ) {
        if( active_substrates[i] != parsec_profiling_substrates[s] ) {
            active_substrates[j++] = active_substrates[i];
        }
    }
    nb_active_substrates = j;
    substrate_status[s] = SUBSTRATE_CLOSED;
}

static int format_is_selected(const char *formats, const char *name)
{
    size_t len = strlen(name);
    const char *c = formats;

    while( NULL != c && '\0' != *c ) {
        while( ',' == *c || ' ' == *c ) c++;
        if( 0 == strncasecmp(c, name, len) && ('\0' == c[len] || ',' == c[len] || ' ' == c[len]) ) {
            return 1;
        }
        c = strchr(c, ',');
    }
    return 0;
}

static void check_formats(const char *formats)
{
    char *copy = strdup(formats), *saveptr = NULL;
    for( char *f = strtok_r(copy, ", ", &saveptr); NULL != f; f = strtok_r(NULL, ", ", &saveptr) ) {
        int s;
        for( s = 0; s < NB_SUBSTRATES; s++ ) {
            if( PARSEC_PROFILING_SUBSTRATE_FILE == parsec_profiling_substrates[s]->kind &&
                0 == strcasecmp(f, parsec_profiling_substrates[s]->name) ) {
                break;
            }
        }
        if( s == NB_SUBSTRATES ) {
            parsec_warning("Profiling system: profile_format '%s' is not available in this build and is ignored.", f);
        }
    }
    free(copy);
}

void parsec_profiling_add_information( const char *key, const char *value )
{
    parsec_profiling_info_t *n;
    n = (parsec_profiling_info_t *)calloc(1, sizeof(parsec_profiling_info_t));
    n->key = strdup(key);
    n->value = strdup(value);
    n->next = parsec_profiling_infos;
    parsec_profiling_infos = n;
}

void parsec_profiling_stream_add_information(parsec_profiling_stream_t* stream,
                                             const char *key, const char *value )
{
    parsec_profiling_info_t *n;
    n = (parsec_profiling_info_t *)calloc(1, sizeof(parsec_profiling_info_t));
    n->key = strdup(key);
    n->value = strdup(value);
    n->next = stream->infos;
    stream->infos = n;
}

static void free_infos(parsec_profiling_info_t *infos)
{
    parsec_profiling_info_t *n;
    while( NULL != (n = infos) ) {
        infos = n->next;
        free(n->key);
        free(n->value);
        free(n);
    }
}

static void add_runtime_informations(void)
{
    /* add the hostname, for the sake of explicit profiling */
    char buf[HOST_NAME_MAX];
    if (0 == gethostname(buf, HOST_NAME_MAX))
        parsec_profiling_add_information("hostname", buf);
    else
        parsec_profiling_add_information("hostname", "");

    /* the current working directory may also be helpful */
    char * newcwd = NULL;
    int bufsize = HOST_NAME_MAX;
    errno = 0;
    char * cwd = getcwd(buf, bufsize);
    while (cwd == NULL && errno == ERANGE) {
        bufsize *= 2;
        cwd = realloc(cwd, bufsize);
        if (cwd == NULL)            /* failed  - just give up */
            break;
        errno = 0;
        newcwd = getcwd(cwd, bufsize);
        if (newcwd == NULL) {
            free(cwd);
            cwd = NULL;
        }
    }
    if (cwd != NULL) {
        parsec_profiling_add_information("cwd", cwd);
        if (cwd != buf)
            free(cwd);
    } else
        parsec_profiling_add_information("cwd", "");
}

/* Reserve keys 0/1 to catch traces that use an uninitialized key. */
static void reserve_null_key(void)
{
    int na_s, na_e;
    parsec_profiling_add_dictionary_keyword( "N/A", "fill:#000000", 0, "", &na_s, &na_e);
    assert(na_s == 0 && na_e == 1);
    (void)na_s; (void)na_e;
}

int parsec_profiling_init( int process_id )
{
    char *formats = NULL;

    if( __profile_initialized ) return PARSEC_ERR_NOT_SUPPORTED;

    PARSEC_TLS_KEY_CREATE(tls_profiling);
    PARSEC_OBJ_CONSTRUCT( &streams, parsec_list_t );

    parsec_prof_keys_count = 0;
    parsec_mca_param_reg_int_name("profile", "keys", "Maximum number of profiling keys",
                                  false, false, parsec_prof_keys_number, &parsec_prof_keys_number);
    if( parsec_prof_keys_number <= 0 || parsec_prof_keys_number > PARSEC_PROFILING_KEYS_MAX ) {
        parsec_prof_keys_number = PARSEC_PROFILING_KEYS_MAX;
    }
    parsec_mca_param_reg_string_name("profile", "format",
                                     "Comma separated list of the file formats written when a profile_filename is provided"
                                     " (binary"
#if defined(PARSEC_HAVE_OTF2)
                                     ", otf2"
#endif
                                     ")",
                                     false, false, PARSEC_PROFILING_DEFAULT_FORMAT, &formats);
    if( NULL != formats ) {
        check_formats(formats);
    }

    reset_last_error();
    for( int s = 0; s < NB_SUBSTRATES; s++ ) {
        parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
        int requested;

        sub->index = s;
        substrate_status[s] = SUBSTRATE_UNUSED;
        requested = (NULL != sub->register_mca) ? sub->register_mca() : 0;
        if( PARSEC_PROFILING_SUBSTRATE_FILE == sub->kind ) {
            requested = (NULL != formats) && format_is_selected(formats, sub->name);
        }
        if( !requested ) {
            continue;
        }
        if( (NULL != sub->init) && (0 != sub->init(process_id)) ) {
            continue;
        }
        if( PARSEC_PROFILING_SUBSTRATE_FILE == sub->kind ) {
            substrate_status[s] = SUBSTRATE_PREPARED;
        } else {
            activate_substrate(s);
        }
    }
    free(formats);

    /* As we called the _start function automatically, the timing will be
     * based on this moment. */
    parsec_profile_enabled = 1;  /* turn on the profiling */

    add_runtime_informations();

    __profile_initialized = 1; //* confirmed */
    reserve_null_key();
    return 0;
}

int parsec_profiling_dbp_start( const char *basefile, const char *hr_info )
{
    char *xmlbuffer;
    int  buflen;
    int  rc = PARSEC_ERR_NOT_SUPPORTED, nb_opened = 0;

    if( !__profile_initialized ) return PARSEC_ERR_NOT_SUPPORTED;

    reset_last_error();
    for( int s = 0; s < NB_SUBSTRATES; s++ ) {
        parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
        if( SUBSTRATE_PREPARED != substrate_status[s] ) {
            continue;
        }
        rc = sub->open(basefile, hr_info);
        if( 0 != rc ) {
            parsec_warning("Profiling system: the %s substrate could not be started (%s).",
                           sub->name, parsec_profiling_strerror());
            continue;
        }
        activate_substrate(s);
        nb_opened++;
    }
    if( 0 == nb_opened ) {
        if( PARSEC_ERR_NOT_SUPPORTED == rc ) {
            parsec_profiling_set_last_error("Profiling system: error: no file format selected (profile_format)\n");
        }
        return (0 != rc) ? rc : PARSEC_ERROR;
    }
    /* The file substrates reported their errors through warnings, the
     * profiling is started on the other ones. */
    reset_last_error();

    /* It's fine to re-reset the event date: we're back with a zero-length event set */
    start_called = 0;

    if( parsec_hwloc_export_topology(&buflen, &xmlbuffer) != -1 &&
        buflen > 0 ) {
        parsec_profiling_add_information("HWLOC-XML", xmlbuffer);
        parsec_hwloc_free_xml_buffer(xmlbuffer);
    }

    return 0;
}

void parsec_profiling_start(void)
{
    if(start_called)
        return;
    if( !__profile_initialized || !parsec_profiling_is_active() )
        return;

    for( int i = 0; i < nb_active_substrates; i++ ) {
        if( NULL != active_substrates[i]->start ) {
            active_substrates[i]->start();
        }
    }
    start_called = 1;
    parsec_start_time = take_time();
}

parsec_profiling_stream_t* parsec_profiling_stream_init( size_t length, const char *format, ...)
{
    parsec_profiling_stream_t *stream;
    va_list ap;
    int rc, s;

    (void)length;
    if( !__profile_initialized ) return NULL;
    if( !parsec_profiling_is_active() ) {
        parsec_profiling_set_last_error("Profiling system: parsec_profiling_stream_init: call before a profiling substrate was started");
        return NULL;
    }

    stream = (parsec_profiling_stream_t*)calloc(1, sizeof(parsec_profiling_stream_t));
    if( NULL == stream ) {
        parsec_profiling_set_last_error("Profiling system: parsec_profiling_stream_init: unable to allocate %zu bytes",
                                        sizeof(parsec_profiling_stream_t));
        return NULL;
    }
    PARSEC_OBJ_CONSTRUCT(stream, parsec_list_item_t);
    va_start(ap, format);
    rc = vasprintf(&stream->hr_id, format, ap); assert(rc!=-1); (void)rc;
    va_end(ap);

    for( s = 0; s < NB_SUBSTRATES; s++ ) {
        parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
        if( SUBSTRATE_ACTIVE != substrate_status[s] || NULL == sub->stream_init ) {
            continue;
        }
        if( 0 != sub->stream_init(stream, &stream->substrate_data[s]) ) {
            goto cleanup_and_fail;
        }
    }

    parsec_list_push_back( &streams, (parsec_list_item_t*)stream );
    return stream;

  cleanup_and_fail:
    while( --s >= 0 ) {
        parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
        if( NULL != stream->substrate_data[s] && NULL != sub->stream_fini ) {
            sub->stream_fini(stream, stream->substrate_data[s]);
        }
    }
    PARSEC_OBJ_DESTRUCT(stream);
    free(stream->hr_id);
    free(stream);
    return NULL;
}

parsec_profiling_stream_t *parsec_profiling_set_default_thread( parsec_profiling_stream_t *stream )
{
    parsec_profiling_stream_t *old;
    old = PARSEC_TLS_GET_SPECIFIC(tls_profiling);
    PARSEC_TLS_SET_SPECIFIC(tls_profiling, stream);
    return old;
}

void parsec_profiling_thread_fini(void)
{
    for( int s = 0; s < NB_SUBSTRATES; s++ ) {
        parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
        if( SUBSTRATE_UNUSED != substrate_status[s] && NULL != sub->thread_fini ) {
            sub->thread_fini();
        }
    }
}

int parsec_profiling_dbp_dump( void )
{
    int rc, nb_dumped = 0;

    if( !__profile_initialized ) return 0;

    for( int s = 0; s < NB_SUBSTRATES; s++ ) {
        parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
        if( PARSEC_PROFILING_SUBSTRATE_FILE != sub->kind ||
            SUBSTRATE_ACTIVE != substrate_status[s] ) {
            continue;
        }
        deactivate_substrate(s);
        rc = sub->dump();
        if( 0 != rc ) {
            parsec_profiling_raise_error = 1;
        }
        nb_dumped++;
    }
    if( 0 == nb_dumped ) {
        parsec_profiling_set_last_error("Profiling system: User Error: parsec_profiling_dbp_dump before parsec_profiling_dbp_start()");
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    if( parsec_profiling_raise_error )
        return PARSEC_ERROR;

    return 0;
}

static void dictionary_clear(void)
{
    for( int i = 0; i < parsec_prof_keys_count; i++ ) {
        parsec_profiling_key_t *entry = dictionary_slot(i);
        if( NULL != entry && NULL != entry->name ) {
            free(entry->name);
            free(entry->attributes);
            free(entry->convertor);
            memset(entry, 0, sizeof(parsec_profiling_key_t));
        }
    }
    parsec_prof_keys_count = 0;
    for( int s = 0; s < NB_SUBSTRATES; s++ ) {
        parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
        if( SUBSTRATE_UNUSED != substrate_status[s] && NULL != sub->dictionary_flush ) {
            sub->dictionary_flush();
        }
    }
}

int parsec_profiling_fini( void )
{
    parsec_profiling_stream_t *t;

    if( !__profile_initialized ) return PARSEC_ERR_NOT_SUPPORTED;

    for( int s = 0; s < NB_SUBSTRATES; s++ ) {
        if( PARSEC_PROFILING_SUBSTRATE_FILE == parsec_profiling_substrates[s]->kind &&
            SUBSTRATE_ACTIVE == substrate_status[s] ) {
            if( 0 != parsec_profiling_dbp_dump() ) {
                return PARSEC_ERROR;
            }
            break;
        }
    }

    parsec_profiling_thread_fini();
    nb_active_substrates = 0;

    while( (t = (parsec_profiling_stream_t*)parsec_list_nolock_pop_front(&streams)) ) {
        for( int s = 0; s < NB_SUBSTRATES; s++ ) {
            parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
            if( NULL != t->substrate_data[s] && NULL != sub->stream_fini ) {
                sub->stream_fini(t, t->substrate_data[s]);
            }
        }
        free_infos(t->infos);
        PARSEC_OBJ_DESTRUCT(t);
        free(t->hr_id);
        free(t);
    }
    PARSEC_OBJ_DESTRUCT(&streams);

    dictionary_clear();
    for( int i = 0; i < PARSEC_PROFILING_KEYS_MAX / PARSEC_PROFILING_KEYS_CHUNK; i++ ) {
        free(parsec_prof_keys[i]);
        parsec_prof_keys[i] = NULL;
    }

    for( int s = 0; s < NB_SUBSTRATES; s++ ) {
        parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
        if( SUBSTRATE_UNUSED != substrate_status[s] && NULL != sub->fini ) {
            sub->fini();
        }
        substrate_status[s] = SUBSTRATE_UNUSED;
    }

    free_infos(parsec_profiling_infos);
    parsec_profiling_infos = NULL;

    start_called = 0;            /* Allow the profiling to be reinitialized */
    parsec_profile_enabled = 0;  /* turn off the profiling */
    __profile_initialized = 0;   /* not initialized */

    return 0;
}

int parsec_profiling_reset( void )
{
    parsec_list_item_t *it;

    parsec_list_lock(&streams);
    for( it = PARSEC_LIST_ITERATOR_FIRST(&streams);
         it != PARSEC_LIST_ITERATOR_END(&streams);
         it = PARSEC_LIST_ITERATOR_NEXT(it) ) {
        parsec_profiling_stream_t *t = (parsec_profiling_stream_t*)it;
        for( int s = 0; s < NB_SUBSTRATES; s++ ) {
            parsec_profiling_substrate_t *sub = parsec_profiling_substrates[s];
            if( NULL != t->substrate_data[s] && NULL != sub->stream_reset ) {
                sub->stream_reset(t, t->substrate_data[s]);
            }
        }
    }
    parsec_list_unlock(&streams);

    return 0;
}

int parsec_profiling_add_dictionary_keyword( const char* key_name, const char* attributes,
                                            size_t info_length,
                                            const char* convertor_code,
                                            int* key_start, int* key_end )
{
    parsec_profiling_key_t *entry;
    int ret = 0;
    int i;
    int pos = -1;

    if( !__profile_initialized ) return 0;
    pthread_mutex_lock(&profiling_keyword_lock);
    for( i = 0; i < parsec_prof_keys_count; i++ ) {
        entry = dictionary_slot(i);
        if( NULL == entry->name ) {
            if( -1 == pos ) {
                pos = i;
            }
            continue;
        }
        if( 0 == strcmp(entry->name, key_name) ) {
            *key_start = START_KEY(i);
            *key_end = END_KEY(i);
            goto profiling_keyword_out;
        }
    }
    if( -1 == pos ) {
        if( parsec_prof_keys_count == parsec_prof_keys_number ) {
            parsec_profiling_set_last_error("Profiling system: error: parsec_profiling_add_dictionary_keyword: Number of keyword limits reached");
            ret = PARSEC_ERR_OUT_OF_RESOURCE;
            goto profiling_keyword_out;
        }
        pos = parsec_prof_keys_count;
        if( NULL == parsec_prof_keys[pos / PARSEC_PROFILING_KEYS_CHUNK] ) {
            parsec_prof_keys[pos / PARSEC_PROFILING_KEYS_CHUNK] =
                (parsec_profiling_key_t*)calloc(PARSEC_PROFILING_KEYS_CHUNK, sizeof(parsec_profiling_key_t));
            if( NULL == parsec_prof_keys[pos / PARSEC_PROFILING_KEYS_CHUNK] ) {
                ret = PARSEC_ERR_OUT_OF_RESOURCE;
                goto profiling_keyword_out;
            }
        }
    }

    entry = dictionary_slot(pos);
    entry->attributes = strdup(attributes);
    entry->info_length = info_length;
    if( NULL != convertor_code )
        entry->convertor = strdup(convertor_code);
    else
        entry->convertor = NULL;
    /* The entry is complete before it becomes visible to the tracing
     * threads, through its name and the number of entries */
    parsec_mfence();
    entry->name = strdup(key_name);
    parsec_mfence();
    if( pos == parsec_prof_keys_count ) {
        parsec_prof_keys_count++;
    }

    *key_start = START_KEY(pos);
    *key_end = END_KEY(pos);

    for( i = 0; i < nb_active_substrates; i++ ) {
        if( NULL != active_substrates[i]->register_key ) {
            active_substrates[i]->register_key(pos, entry);
        }
    }
profiling_keyword_out:
    pthread_mutex_unlock(&profiling_keyword_lock);
    return ret;
}

int parsec_profiling_dictionary_flush( void )
{
    pthread_mutex_lock(&profiling_keyword_lock);
    dictionary_clear();
    pthread_mutex_unlock(&profiling_keyword_lock);
    if( __profile_initialized ) {
        reserve_null_key();
    }
    return 0;
}

int parsec_profiling_ts_trace_flags_info_fn(int key, uint64_t event_id, uint32_t taskpool_id,
                                            parsec_profiling_info_fn_t *info_fn, const void *info_data, uint16_t flags )
{
    parsec_profiling_stream_t* ctx;

    if( !__profile_initialized ||
        !start_called ||
        !parsec_profiling_is_active() ) {
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    ctx = PARSEC_TLS_GET_SPECIFIC(tls_profiling);
    if( NULL != ctx )
        return parsec_profiling_trace_flags_info_fn(ctx, key, event_id, taskpool_id, info_fn, info_data, flags);

    parsec_profiling_set_last_error("Profiling system: error: called parsec_profiling_ts_trace_flags_info_fn"
                                    " from a thread that did not call parsec_profiling_stream_init\n");
    return PARSEC_ERR_NOT_SUPPORTED;
}

int
parsec_profiling_trace_flags(parsec_profiling_stream_t* context, int key,
                            uint64_t event_id, uint32_t taskpool_id,
                            const void *info, uint16_t flags)
{
    return parsec_profiling_trace_flags_info_fn(context, key, event_id, taskpool_id, memcpy, info, flags);
}

int
parsec_profiling_trace_flags_info_fn(parsec_profiling_stream_t* context, int key,
                                     uint64_t event_id, uint32_t taskpool_id,
                                     parsec_profiling_info_fn_t *info_fn, const void *info_data, uint16_t flags)
{
    parsec_profiling_trace_event_t event;
    int rc, ret = 0;

    if( !__profile_initialized ||
        !start_called ||
        !parsec_profiling_is_active() ) {
        return PARSEC_ERR_NOT_SUPPORTED;
    }

    if(flags & PARSEC_PROFILING_EVENT_TIME_AT_START) {
        event.start_time = take_time();
    }

    if( -1 == key ) {
        /* -1 marks a disabled key, silently ignore it */
        return PARSEC_ERR_BAD_PARAM;
    }
    if( key < 2 || key >= 2*parsec_prof_keys_count ) {
        if( !parsec_prof_warning_issued ) {
            fprintf(stderr,
                    "*** PaRSEC Profiling Warning *** Logging an event with key %d which is either reserved (0 and 1) or larger than the highest registered key (%d). Profiling might be corrupted.\n",
                    key, 2*parsec_prof_keys_count);
            parsec_prof_warning_issued = 1;
        }
        assert(0); /* In DEBUG mode, we provide a catch point here to find what task issued this profiling */
        return PARSEC_ERR_BAD_PARAM;
    }

    event.key = BASE_KEY(key);
    event.is_start = KEY_IS_START(key);
    event.is_default_stream = (context == PARSEC_TLS_GET_SPECIFIC(tls_profiling));
    event.flags = flags;
    event.taskpool_id = taskpool_id;
    event.event_id = event_id;
    event.info_fn = info_fn;
    event.info_data = info_data;

    for( int i = 0; i < nb_active_substrates; i++ ) {
        parsec_profiling_substrate_t *sub = active_substrates[i];
        rc = sub->trace(context, context->substrate_data[sub->index], &event);
        if( 0 != rc ) {
            ret = rc;
        }
    }
    return ret;
}

void parsec_profiling_enable(void)
{
    parsec_profile_enabled = 1;
}
void parsec_profiling_disable(void)
{
    parsec_profile_enabled = 0;
}

void profiling_save_dinfo(const char *key, double value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%g", value);
    (void)rv;
    parsec_profiling_add_information(key, svalue);
    free(svalue);
}

void profiling_save_iinfo(const char *key, int value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%d", value);
    (void)rv;
    parsec_profiling_add_information(key, svalue);
    free(svalue);
}

void profiling_save_uint64info(const char *key, unsigned long long int value)
{
    char *svalue;
    int rv=asprintf(&svalue, "%llu", value);
    (void)rv;
    parsec_profiling_add_information(key, svalue);
    free(svalue);
}

void profiling_save_sinfo(const char *key, char* svalue)
{
    parsec_profiling_add_information(key, svalue);
}

void profiling_stream_save_dinfo(parsec_profiling_stream_t* stream,
                                 const char *key, double value)
{
    char *svalue;
    int rv = asprintf(&svalue, "%g", value);
    (void)rv;
    parsec_profiling_stream_add_information(stream, key, svalue);
    free(svalue);
}

void profiling_stream_save_iinfo(parsec_profiling_stream_t* stream,
                                 const char *key, int value)
{
    char *svalue;
    int rv = asprintf(&svalue, "%d", value);
    (void)rv;
    parsec_profiling_stream_add_information(stream, key, svalue);
    free(svalue);
}

void profiling_stream_save_uint64info(parsec_profiling_stream_t* stream,
                                      const char *key, unsigned long long int value)
{
    char *svalue;
    int rv = asprintf(&svalue, "%llu", value);
    (void)rv;
    parsec_profiling_stream_add_information(stream, key, svalue);
    free(svalue);
}

void profiling_stream_save_sinfo(parsec_profiling_stream_t* stream,
                                 const char *key, char* svalue)
{
    parsec_profiling_stream_add_information(stream, key, svalue);
}
