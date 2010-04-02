/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "profiling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <inttypes.h>

#include "atomic.h"
#define min(a, b) ((a)<(b)?(a):(b))

#include "os-spec-timing.h"
#include "dequeue.h"

typedef struct dplasma_profiling_output_t {
    int key;
    unsigned long id;
    dplasma_time_t timestamp;
#if defined(HAVE_PAPI)
    long long counter_value;
#endif /* defined(HAVE_PAPI) */
} dplasma_profiling_output_t;

struct dplasma_thread_profiling_t {
    dplasma_list_item_t list;
    unsigned int events_count;
    unsigned int events_limit;
    char *hr_id;
    dplasma_profiling_output_t events[1];
};

typedef struct dplasma_profiling_key_t {
    char* name;
    char* attributes;
} dplasma_profiling_key_t;

#define START_KEY(key)  ( (key) * 2 )
#define END_KEY(key)    ( (key) * 2 + 1 )

/* Process-global dictionnary */
static int dplasma_prof_keys_count, dplasma_prof_keys_number;
static dplasma_profiling_key_t* dplasma_prof_keys;

/* Process-global profiling list */
static dplasma_dequeue_t threads;
static char *hr_id = NULL;

static char *dplasma_profiling_last_error = NULL;
static void ERROR(const char *format, ...)
{
    va_list ap;

    if( dplasma_profiling_last_error != NULL ) {
        free(dplasma_profiling_last_error);
    }
    va_start(ap, format);
    vasprintf(&dplasma_profiling_last_error, format, ap);
    va_end(ap);
}

char *dplasma_profiling_strerror(void)
{
    return dplasma_profiling_last_error;
}

int dplasma_profiling_change_profile_attribute( const char *format, ... )
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

int dplasma_profiling_init( const char *format, ... )
{
    va_list ap;

    if( hr_id != NULL ) {
        ERROR("dplasma_profiling_init: profiling already initialized");
        return -1;
    }

    va_start(ap, format);
    vasprintf(&hr_id, format, ap);
    va_end(ap);

    dplasma_dequeue_construct( &threads );

    dplasma_prof_keys = (dplasma_profiling_key_t*)calloc(128, sizeof(dplasma_profiling_key_t));
    dplasma_prof_keys_count = 0;
    dplasma_prof_keys_number = 128;

    return 0;
}

dplasma_thread_profiling_t *dplasma_profiling_thread_init( unsigned int length, const char *format, ...)
{
    va_list ap;
    dplasma_thread_profiling_t *res;

    /** Remark: maybe calloc would be less perturbing for the measurements,
     *  if we consider that we don't care about the _init phase, but only
     *  about the measurement phase that happens later.
     */
    res = (dplasma_thread_profiling_t*)malloc( sizeof(dplasma_thread_profiling_t) + (length-1) * sizeof(dplasma_profiling_output_t) );
    if( NULL == res ) {
        ERROR("dplasma_profiling_thread_init: unable to allocate %u output elements", length);
        return NULL;
    }

    va_start(ap, format);
    vasprintf(&res->hr_id, format, ap);
    va_end(ap);

    res->events_limit = length;
    res->events_count = 0;

    dplamsa_dequeue_item_construct( (dplasma_list_item_t*)res );
    dplasma_dequeue_push_back( &threads, (dplasma_list_item_t*)res );

    return res;
}

int dplasma_profiling_fini( void )
{
    dplasma_thread_profiling_t *t;
    
    while( !dplasma_dequeue_is_empty( &threads ) ) {
        t = (dplasma_thread_profiling_t*)dplasma_dequeue_pop_front( &threads );
        if( NULL == t ) 
            continue;
        free(t->hr_id);
        free(t);
    }

    free(hr_id);

    dplasma_profiling_dictionary_flush();
    free(dplasma_prof_keys);
    dplasma_prof_keys_number = 0;

    return 0;
}

int dplasma_profiling_reset( void )
{
    dplasma_thread_profiling_t *t;
    dplasma_list_item_t *it;
    
    dplasma_atomic_lock( &threads.atomic_lock );
    for( it = (dplasma_list_item_t*)threads.ghost_element.list_next; 
         it != &threads.ghost_element; 
         it = (dplasma_list_item_t*)it->list_next ) {
        t = (dplasma_thread_profiling_t*)it;
        t->events_count = 0;
    }
    dplasma_atomic_unlock( &threads.atomic_lock );

    return 0;
}

int dplasma_profiling_add_dictionary_keyword( const char* key_name, const char* attributes,
                                              int* key_start, int* key_end )
{
    int i, pos = -1;

    for( i = 0; i < dplasma_prof_keys_count; i++ ) {
        if( NULL == dplasma_prof_keys[i].name ) {
            if( -1 == pos ) {
                pos = i;
            }
            continue;
        }
        if( 0 == strcmp(dplasma_prof_keys[i].name, key_name) ) {
            *key_start = START_KEY(i);
            *key_end = END_KEY(i);
            return 0;
        }
    }
    if( -1 == pos ) {
        if( dplasma_prof_keys_count == dplasma_prof_keys_number ) {
            ERROR("dplasma_profiling_add_dictionary_keyword: Number of keyword limits reached");
            return -1;
        }
        pos = dplasma_prof_keys_count;
    }

    dplasma_prof_keys[pos].name = strdup(key_name);
    dplasma_prof_keys[pos].attributes = strdup(attributes);

    *key_start = START_KEY(pos);
    *key_end = END_KEY(pos);
    dplasma_prof_keys_count++;
    return 0;
}

int dplasma_profiling_dictionary_flush( void )
{
    int i;

    for( i = 0; i < dplasma_prof_keys_count; i++ ) {
        if( NULL != dplasma_prof_keys[i].name ) {
            free(dplasma_prof_keys[i].name);
            free(dplasma_prof_keys[i].attributes);
        }
    }
    dplasma_prof_keys_count = 0;

    return 0;
}

int dplasma_profiling_trace( dplasma_thread_profiling_t* context, int key, unsigned long id )
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

static int dplasma_profiling_dump_one_xml( const dplasma_thread_profiling_t *profile, 
                                           FILE *out,
                                           dplasma_time_t relative )
{
    int key, start_idx, end_idx, displayed_key;
    uint64_t start, end;
    static int displayed_error_message = 0;

    for( key = 0; key < dplasma_prof_keys_count; key++ ) {
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

int dplasma_profiling_dump_xml( const char* filename )
{
    int i, last_timestamp, foundone;
    dplasma_time_t relative = ZERO_TIME, latest = ZERO_TIME;
    dplasma_list_item_t *it;
    dplasma_thread_profiling_t* profile;
    FILE* tracefile;
 
    tracefile = fopen(filename, "w");
    if( NULL == tracefile ) {
        return -1;
    }

    fprintf(tracefile,
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<PROFILING>\n"
            " <IDENTIFIER><![CDATA[%s]]></IDENTIFIER>\n"
            " <DICTIONARY>\n",
            hr_id);

    for(i = 0; i < dplasma_prof_keys_count; i++) {
        fprintf(tracefile,
                "   <KEY ID=\"%d\">\n"
                "    <NAME>%s</NAME>\n"
                "    <ATTRIBUTES><![CDATA[%s]]></ATTRIBUTES>\n"
                "   </KEY>\n",
                i, dplasma_prof_keys[i].name, dplasma_prof_keys[i].attributes);
    }
    fprintf(tracefile, " </DICTIONARY>\n");

    foundone = 0;
   
    dplasma_atomic_lock( &threads.atomic_lock );
    for( it = (dplasma_list_item_t*)threads.ghost_element.list_next; 
         it != &threads.ghost_element; 
         it = (dplasma_list_item_t*)it->list_next ) {
        profile = (dplasma_thread_profiling_t*)it;

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

    for( it = (dplasma_list_item_t*)threads.ghost_element.list_next; 
         it != &threads.ghost_element; 
         it = (dplasma_list_item_t*)it->list_next ) {
        profile = (dplasma_thread_profiling_t*)it;

        fprintf(tracefile, 
                "   <THREAD>\n"
                "    <IDENTIFIER><![CDATA[%s]]></IDENTIFIER>\n", profile->hr_id);
        dplasma_profiling_dump_one_xml(profile, tracefile, relative);
        fprintf(tracefile, 
                "   </THREAD>\n");
    }
    dplasma_atomic_unlock( &threads.atomic_lock );

    fprintf(tracefile, 
            " </PROFILES>\n"
            "</PROFILING>\n");
    fclose(tracefile);
    
    return 0;
}

