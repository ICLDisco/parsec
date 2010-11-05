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
#include "data_distribution.h"

#include "atomic.h"
#define min(a, b) ((a)<(b)?(a):(b))

#include "os-spec-timing.h"
#include "dequeue.h"

typedef struct dague_profiling_output_t {
    unsigned int   key;
    unsigned long  id;
    dague_time_t   timestamp;
    unsigned char  has_info;
    char           info[1];
} dague_profiling_output_t;

typedef struct dague_profiling_info {
    const char *key;
    const char *value;
    struct dague_profiling_info *next;
} dague_profiling_info_t;

struct dague_thread_profiling_t {
    dague_list_item_t list;
    char *next_event;
    char *last_event;
    char *events_top;
    char *hr_id;
    dague_profiling_info_t  *infos;
    char  events[1];
};

typedef struct dague_profiling_key_t {
    char*  name;
    char*  attributes;
    size_t info_length;
    dague_profiling_info_convert_fct_t cnvt;
} dague_profiling_key_t;

/** here key is the key given to the USER */
#define BASE_KEY(key)     ( (key) / 2 )
#define EVENT_LENGTH(key) ( sizeof(dague_profiling_output_t) - 1 + dague_prof_keys[BASE_KEY(key)].info_length )

/** here keys are the internal key */
#define START_KEY(key)    ( (key) * 2 )
#define END_KEY(key)      ( (key) * 2 + 1 )

#define FORALL_EVENTS(iterator, start)                                  \
    for(iterator = (dague_profiling_output_t *)(start);                 \
        (char *)iterator != profile->next_event;                        \
        iterator = (dague_profiling_output_t *)(((char*)iterator) + EVENT_LENGTH( iterator->key ) ) )

/* Process-global dictionnary */
static unsigned int dague_prof_keys_count, dague_prof_keys_number;
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

void dague_profiling_add_information( const char *key, const char *value )
{
    dague_profiling_info_t *n;
    n = (dague_profiling_info_t *)calloc(1, sizeof(dague_profiling_info_t));
    n->key = strdup(key);
    n->value = strdup(value);
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

dague_thread_profiling_t *dague_profiling_thread_init( size_t length, const char *format, ...)
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

    res->next_event = res->events;
    res->last_event = NULL;
    res->events_top = res->events + length;

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
        t->last_event = t->next_event;
        t->next_event = t->events;
    }
    dague_atomic_unlock( &threads.atomic_lock );

    return 0;
}

int dague_profiling_add_dictionary_keyword( const char* key_name, const char* attributes,
                                            size_t info_length, dague_profiling_info_convert_fct_t cnvt,
                                            int* key_start, int* key_end )
{
    unsigned int i;
    int pos = -1;

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
    dague_prof_keys[pos].info_length = info_length;
    dague_prof_keys[pos].cnvt = cnvt;

    *key_start = START_KEY(pos);
    *key_end = END_KEY(pos);
    dague_prof_keys_count++;
    return 0;
}

int dague_profiling_dictionary_flush( void )
{
    unsigned int i;

    for( i = 0; i < dague_prof_keys_count; i++ ) {
        if( NULL != dague_prof_keys[i].name ) {
            free(dague_prof_keys[i].name);
            free(dague_prof_keys[i].attributes);
        }
    }
    dague_prof_keys_count = 0;

    return 0;
}

int dague_profiling_trace( dague_thread_profiling_t* context, int key, unsigned long id, void *info )
{
    size_t my_event_length, info_length;
    dague_profiling_output_t *my_event;

    info_length = dague_prof_keys[ BASE_KEY(key) ].info_length;
    my_event_length = EVENT_LENGTH( key );
    my_event = (dague_profiling_output_t *)context->next_event;

    if( context->next_event == context->events_top ) {
        return -1;
    }
    context->last_event = context->next_event;
    context->next_event += my_event_length;

    my_event->key = key;
    my_event->id  = id;

    if( NULL != info ) {
        memcpy(my_event->info, info, info_length);
        my_event->has_info = 1;
    } else {
        my_event->has_info = 0;
    }
    my_event->timestamp = take_time();    
    
    return 0;
}

static dague_profiling_output_t *find_matching_event_in_profile(const dague_thread_profiling_t *profile,
                                                                const dague_profiling_output_t *start,
                                                                unsigned int pos)
{
    dague_profiling_output_t *e;
    FORALL_EVENTS(e, start) {
        if( ( time_less(start->timestamp, e->timestamp) || 
             (diff_time(start->timestamp, e->timestamp) == 0) ) &&
            e->key == END_KEY(pos) &&
            e->id == start->id) {
            return e;
        }
    }
    return NULL;
}

static int dague_profiling_dump_one_xml( const dague_thread_profiling_t *profile, 
                                         FILE *out,
                                         dague_time_t relative )
{
    unsigned int pos, displayed_key;
    uint64_t start, end;
    static int displayed_error_message = 0;
    char *infostr = malloc(4);
    int infostrsize = 4, infostrresize;
    int event_not_found;
    dague_list_item_t *it;
    dague_thread_profiling_t *op;
    const dague_profiling_output_t *start_event, *end_event;

    for( pos = 0; pos < dague_prof_keys_count; pos++ ) {
        displayed_key = 0;
        FORALL_EVENTS( start_event, profile->events ) {
            /* if not my current start_idx key, ignore */
            if( start_event->key != START_KEY(pos) )
                continue;
            
            if( (end_event = find_matching_event_in_profile(profile, start_event, pos)) == NULL ) {
                /* Argh, couldn't find the end in this profile */

                event_not_found = 1;
                if( start_event->id != 0 ) {
                    /* It has an id, let's look somewhere in another profile, maybe it's end has been
                     * logged by another thread
                     */
                    for( it = (dague_list_item_t*)threads.ghost_element.list_next; 
                         it != &threads.ghost_element; 
                         it = (dague_list_item_t*)it->list_next ) {

                        op = (dague_thread_profiling_t*)it;   
                        if( op == profile )
                            continue;

                        if( (end_event = find_matching_event_in_profile(op, (dague_profiling_output_t*)op->events, pos)) != NULL ) {
                            event_not_found = 0;
                            break;
                        }
                    }
                }

                /* Couldn't find the end, or no id. Bad. */
                if( event_not_found ) {
                    if( !displayed_error_message ) {
                        if( profile->next_event == profile->events_top ) {
                            fprintf(stderr, "Profiling error: end event of key %u (%s) id %lu was not found for ID %s\n"
                                    "\t-- some histories are truncated\n",
                                    pos, dague_prof_keys[pos].name, start_event->id, profile->hr_id);
                        } else {
                            fprintf(stderr, "Profiling error: end event of key %u (%s) id %lu was not found for ID %s\n",
                                    pos, dague_prof_keys[pos].name, start_event->id, profile->hr_id);
                        }
                        displayed_error_message = 1;
                    }
                    continue;
                }
            }

            start = diff_time( relative, start_event->timestamp );
            end = diff_time( relative, end_event->timestamp );

            if( displayed_key == 0 ) {
                fprintf(out, "    <KEY ID=\"%u\">\n", pos);
                displayed_key = 1;
            }
            
            fprintf(out, 
                    "     <EVENT>\n"
                    "       <ID>%lu</ID>\n"
                    "       <START>%"PRIu64"</START>\n"
                    "       <END>%"PRIu64"</END>\n",
                    start_event->id,
                    start, end);

            if( start_event->has_info && (NULL != dague_prof_keys[pos].cnvt) ) {
                do {
                    infostrresize = dague_prof_keys[pos].cnvt( (void*)start_event->info, infostr, infostrsize );
                    
                    if( infostrresize >= infostrsize ) {
                        infostr = (char*)realloc(infostr, infostrresize+1);
                        infostrsize = infostrresize+1;
                    } else {
                        break;
                    }
                } while(1);
                if( infostrresize >= 0 )
                    fprintf(out, "       <INFO>%s</INFO>\n", infostr);
            } 
            if( end_event->has_info && (NULL != dague_prof_keys[pos].cnvt) ) {
                do {
                    infostrresize = dague_prof_keys[pos].cnvt( (void*)end_event->info, infostr, infostrsize );
                    
                    if( infostrresize >= infostrsize ) {
                        infostr = (char*)realloc(infostr, infostrresize+1);
                        infostrsize = infostrresize+1;
                    } else {
                        break;
                    }
                } while(1);
                if( infostrresize >= 0 )
                    fprintf(out, "       <INFO ATEND=\"true\">%s</INFO>\n", infostr);
            } 
            fprintf(out, "     </EVENT>\n");
        }
        if( displayed_key ) {
            fprintf(out, "    </KEY>\n");
        }
    }

    free(infostr);
    return 0;
}

int dague_profiling_dump_xml( const char* filename )
{
    unsigned int i;
    int foundone;
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
        fprintf(tracefile, "    <INFO NAME=\"%s\">%s</INFO>\n", info->key, info->value);
    }

    fprintf(tracefile,
            "  </INFOS>\n"
            "  <DICTIONARY>\n");

    for(i = 0; i < dague_prof_keys_count; i++) {
        fprintf(tracefile,
                "   <KEY ID=\"%u\">\n"
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

        if( profile->last_event == NULL ) {
            continue;
        }

        if( !foundone ) {
            relative = ((dague_profiling_output_t *)(profile->events))->timestamp;
            latest   = ((dague_profiling_output_t*)(profile->last_event))->timestamp;
            foundone = 1;
        } else {
            if( time_less(((dague_profiling_output_t *)(profile->events))->timestamp, relative) ) {
                relative = ((dague_profiling_output_t *)(profile->events))->timestamp;
            }
            if( time_less( latest, ((dague_profiling_output_t*)(profile->last_event))->timestamp) ) {
                latest = ((dague_profiling_output_t*)(profile->last_event))->timestamp;
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

int dague_profile_ddesc_key_to_string(void *info, char *text, size_t size)
{
    int res;
    dague_profile_ddesc_info_t nfo = *(dague_profile_ddesc_info_t*)info;
    if( nfo.desc != NULL ) {
        res = snprintf(text, size, "%s", nfo.desc->key);
        if( res >= size ) {
            res += nfo.desc->key_to_string( nfo.desc, nfo.id, text, (uint32_t) size );
        } else {
            res += nfo.desc->key_to_string( nfo.desc, nfo.id, text + res, (uint32_t)(size-res) );
        }
        return res;
    }
    return -1;
}
