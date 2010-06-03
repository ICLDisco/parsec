/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "profiling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <inttypes.h>
#include "profiling.h"

#include "atomic.h"
#define min(a, b) ((a)<(b)?(a):(b))

#include "os-spec-timing.h"
#include "dequeue.h"

typedef struct dague_profiling_output_t {
    int key;
    unsigned long id;
    dague_time_t timestamp;
#if defined(HAVE_PAPI)
    long long counter_value;
#endif /* defined(HAVE_PAPI) */
} dague_profiling_output_t;

typedef struct dague_profiling_info {
    const char *key;
    int value;
    struct dague_profiling_info *next;
} dague_profiling_info_t;

struct dague_thread_profiling_t {
    dague_list_item_t list;
    unsigned int events_count;
    unsigned int events_limit;
    char *hr_id;
    dague_profiling_info_t  *infos;
    dague_profiling_output_t events[1];
};

typedef struct dague_profiling_key_t {
    char* name;
    char* attributes;
} dague_profiling_key_t;

#define START_KEY(key)  ( (key) * 2 )
#define END_KEY(key)    ( (key) * 2 + 1 )

/* Process-global dictionnary */
static int dague_prof_keys_count, dague_prof_keys_number;
static dague_profiling_key_t* dague_prof_keys;

/* Process-global profiling list */
static dague_dequeue_t threads;
static char *hr_id = NULL;
static dague_profiling_info_t *dague_profiling_infos = NULL;

static char *dague_profiling_last_error = NULL;
static void ERROR(const char *format, ...)
{
    va_list ap;

    if( dague_profiling_last_error != NULL ) {
        free(dague_profiling_last_error);
    }
    va_start(ap, format);
    vasprintf(&dague_profiling_last_error, format, ap);
    va_end(ap);
}

char *dague_profiling_strerror(void)
{
    return dague_profiling_last_error;
}

void dague_profiling_add_information( const char *key, int value )
{
    dague_profiling_info_t *n;
    n = (dague_profiling_info_t *)calloc(1, sizeof(dague_profiling_info_t));
    n->key = strdup(key);
    n->value = value;
    n->next = dague_profiling_infos;
    dague_profiling_infos = n;
}

int dague_profiling_change_profile_attribute( const char *format, ... )
{
    va_list ap;

    if( hr_id != NULL ) {
        free(hr_id);
    }

    va_start(ap, format);
    vasprintf(&hr_id, format, ap);
    va_end(ap);

    return 0;
}

int dague_profiling_init( const char *format, ... )
{
    va_list ap;

    if( hr_id != NULL ) {
        ERROR("dague_profiling_init: profiling already initialized");
        return -1;
    }

    va_start(ap, format);
    vasprintf(&hr_id, format, ap);
    va_end(ap);

    dague_dequeue_construct( &threads );

    dague_prof_keys = (dague_profiling_key_t*)calloc(128, sizeof(dague_profiling_key_t));
    dague_prof_keys_count = 0;
    dague_prof_keys_number = 128;

    return 0;
}

dague_thread_profiling_t *dague_profiling_thread_init( unsigned int length, const char *format, ...)
{
    va_list ap;
    dague_thread_profiling_t *res;

    /** Remark: maybe calloc would be less perturbing for the measurements,
     *  if we consider that we don't care about the _init phase, but only
     *  about the measurement phase that happens later.
     */
    res = (dague_thread_profiling_t*)malloc( sizeof(dague_thread_profiling_t) + (length-1) * sizeof(dague_profiling_output_t) );
    if( NULL == res ) {
        ERROR("dague_profiling_thread_init: unable to allocate %u output elements", length);
        return NULL;
    }

    va_start(ap, format);
    vasprintf(&res->hr_id, format, ap);
    va_end(ap);

    res->events_limit = length;
    res->events_count = 0;

    dplamsa_dequeue_item_construct( (dague_list_item_t*)res );
    dague_dequeue_push_back( &threads, (dague_list_item_t*)res );

    return res;
}

int dague_profiling_fini( void )
{
    dague_thread_profiling_t *t;
    
    while( !dague_dequeue_is_empty( &threads ) ) {
        t = (dague_thread_profiling_t*)dague_dequeue_pop_front( &threads );
        if( NULL == t ) 
            continue;
        free(t->hr_id);
        free(t);
    }

    free(hr_id);

    dague_profiling_dictionary_flush();
    free(dague_prof_keys);
    dague_prof_keys_number = 0;

    return 0;
}

int dague_profiling_reset( void )
{
    dague_thread_profiling_t *t;
    dague_list_item_t *it;
    
    dague_atomic_lock( &threads.atomic_lock );
    for( it = (dague_list_item_t*)threads.ghost_element.list_next; 
         it != &threads.ghost_element; 
         it = (dague_list_item_t*)it->list_next ) {
        t = (dague_thread_profiling_t*)it;
        t->events_count = 0;
    }
    dague_atomic_unlock( &threads.atomic_lock );

    return 0;
}

int dague_profiling_add_dictionary_keyword( const char* key_name, const char* attributes,
                                              int* key_start, int* key_end )
{
    int i, pos = -1;

    for( i = 0; i < dague_prof_keys_count; i++ ) {
        if( NULL == dague_prof_keys[i].name ) {
            if( -1 == pos ) {
                pos = i;
            }
            continue;
        }
        if( 0 == strcmp(dague_prof_keys[i].name, key_name) ) {
            *key_start = START_KEY(i);
            *key_end = END_KEY(i);
            return 0;
        }
    }
    if( -1 == pos ) {
        if( dague_prof_keys_count == dague_prof_keys_number ) {
            ERROR("dague_profiling_add_dictionary_keyword: Number of keyword limits reached");
            return -1;
        }
        pos = dague_prof_keys_count;
    }

    dague_prof_keys[pos].name = strdup(key_name);
    dague_prof_keys[pos].attributes = strdup(attributes);

    *key_start = START_KEY(pos);
    *key_end = END_KEY(pos);
    dague_prof_keys_count++;
    return 0;
}

int dague_profiling_dictionary_flush( void )
{
    int i;

    for( i = 0; i < dague_prof_keys_count; i++ ) {
        if( NULL != dague_prof_keys[i].name ) {
            free(dague_prof_keys[i].name);
            free(dague_prof_keys[i].attributes);
        }
    }
    dague_prof_keys_count = 0;

    return 0;
}

int dague_profiling_trace( dague_thread_profiling_t* context, int key, unsigned long id )
{
    int my_event = context->events_count++;

    if( my_event >= context->events_limit ) {
        return -1;
    }
    context->events[my_event].key = key;
    context->events[my_event].id  = id;
    context->events[my_event].timestamp = take_time();
    
    return 0;
}

static int dague_profiling_dump_one_xml( const dague_thread_profiling_t *profile, 
                                           FILE *out,
                                           dague_time_t relative )
{
    int key, start_idx, end_idx, displayed_key;
    uint64_t start, end;
    static int displayed_error_message = 0;

    for( key = 0; key < dague_prof_keys_count; key++ ) {
        displayed_key = 0;
        for( start_idx = 0; start_idx < min(profile->events_count, profile->events_limit); start_idx++ ) {
            /* if not my current start_idx key, ignore */
            if( profile->events[start_idx].key != START_KEY(key) )
                continue;
            
            /* find the end_idx event */
            for( end_idx = start_idx+1; end_idx < min(profile->events_count, profile->events_limit); end_idx++) {
                if( (profile->events[end_idx].key == END_KEY(key)) &&
                    (profile->events[end_idx].id == profile->events[start_idx].id) )
                    break;
            }
            if( end_idx == min(profile->events_count, profile->events_limit) ) {
                if( !displayed_error_message ) {
                    fprintf(stderr, "Profiling error: end event of key %d id %lu was not found -- some histories are truncated\n", key, profile->events[start_idx].id);
                    displayed_error_message = 1;
                }
                continue;
            }

            start = diff_time( relative, profile->events[start_idx].timestamp );
            end = diff_time( relative, profile->events[end_idx].timestamp );

            if( displayed_key == 0 ) {
                fprintf(out, "    <KEY ID=\"%d\">\n", key);
                displayed_key = 1;
            }
            
            fprintf(out, "     <EVENT>\n");

            fprintf(out, "       <ID>%lu</ID>\n"
                         "       <START>%"PRIu64"</START>\n"
                         "       <END>%"PRIu64"</END>\n",
                    profile->events[start_idx].id,
                    start, end);
#ifdef HAVE_PAPI
            fprintf(out, "       <PAPI_START>%ld</PAPI_START>\n"
                         "       <PAPI_END>%ld</PAPI_END>\n",
                    profile->events[start_idx].counter_value,
                    profile->events[end_idx].counter_value);
#endif
            fprintf(out, "     </EVENT>\n");
        }
        if( displayed_key ) {
            fprintf(out, "    </KEY>\n");
        }
    }
    return 0;
}

int dague_profiling_dump_xml( const char* filename )
{
    int i, last_timestamp, foundone;
    dague_time_t relative = ZERO_TIME, latest = ZERO_TIME;
    dague_list_item_t *it;
    dague_thread_profiling_t* profile;
    FILE* tracefile;
    dague_profiling_info_t *info;
 
    tracefile = fopen(filename, "w");
    if( NULL == tracefile ) {
        return -1;
    }

    fprintf(tracefile,
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<PROFILING>\n"
            " <IDENTIFIER><![CDATA[%s]]></IDENTIFIER>\n"
            "  <INFOS>\n", hr_id);
    for(info = dague_profiling_infos; info != NULL; info = info->next ) {
        fprintf(tracefile, "    <INFO NAME=\"%s\">%d</INFO>\n", info->key, info->value);
    }
    fprintf(tracefile,
            "  </INFOS>\n"
            "  <DICTIONARY>\n");

    for(i = 0; i < dague_prof_keys_count; i++) {
        fprintf(tracefile,
                "   <KEY ID=\"%d\">\n"
                "    <NAME>%s</NAME>\n"
                "    <ATTRIBUTES><![CDATA[%s]]></ATTRIBUTES>\n"
                "   </KEY>\n",
                i, dague_prof_keys[i].name, dague_prof_keys[i].attributes);
    }
    fprintf(tracefile, " </DICTIONARY>\n");

    foundone = 0;
   
    dague_atomic_lock( &threads.atomic_lock );
    for( it = (dague_list_item_t*)threads.ghost_element.list_next; 
         it != &threads.ghost_element; 
         it = (dague_list_item_t*)it->list_next ) {
        profile = (dague_thread_profiling_t*)it;

        if( profile->events_count == 0 ) {
            continue;
        }

        if( !foundone ) {
            relative = profile->events[0].timestamp;
            last_timestamp = min(profile->events_count, profile->events_limit) - 1;
            latest   = profile->events[last_timestamp].timestamp;
            foundone = 1;
        } else {
            if( time_less(profile->events[0].timestamp, relative) ) {
                relative = profile->events[0].timestamp;
            }
            last_timestamp = min(profile->events_count, profile->events_limit) - 1;
            if( time_less( latest, profile->events[last_timestamp].timestamp) ) {
                latest = profile->events[last_timestamp].timestamp;
            }
        }

    }

    fprintf(tracefile, " <PROFILES TOTAL_DURATION=\"%"PRIu64"\" TIME_UNIT=\""TIMER_UNIT"\">\n",
            diff_time(relative, latest));

    for( it = (dague_list_item_t*)threads.ghost_element.list_next; 
         it != &threads.ghost_element; 
         it = (dague_list_item_t*)it->list_next ) {
        profile = (dague_thread_profiling_t*)it;

        fprintf(tracefile, 
                "   <THREAD>\n"
                "    <IDENTIFIER><![CDATA[%s]]></IDENTIFIER>\n", profile->hr_id);
        dague_profiling_dump_one_xml(profile, tracefile, relative);
        fprintf(tracefile, 
                "   </THREAD>\n");
    }
    dague_atomic_unlock( &threads.atomic_lock );

    fprintf(tracefile, 
            " </PROFILES>\n"
            "</PROFILING>\n");
    fclose(tracefile);
    
    return 0;
}

