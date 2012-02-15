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
#include "debug.h"
#include "os-spec-timing.h"
#include "fifo.h"

#define min(a, b) ((a)<(b)?(a):(b))

#define DAGUE_PROFILING_EVENT_HAS_INFO     0x0001

typedef struct dague_profiling_output_base_event_t {
    uint16_t        key;
    uint16_t        flags;
    unsigned long   id;
    dague_time_t    timestamp;
} dague_profiling_output_base_event_t;

typedef struct dague_profiling_output_t {
    dague_profiling_output_base_event_t event;
    uint64_t                            info[1];
} dague_profiling_output_t;

typedef struct dague_profiling_info {
    const char *key;
    const char *value;
    struct dague_profiling_info *next;
} dague_profiling_info_t;

struct dague_thread_profiling_t {
    dague_list_item_t        list;
    char                    *next_event;
    char                    *last_event;
    char                    *events_top;
    char                    *hr_id;
    uint64_t                 nb_events;
    dague_profiling_info_t  *infos;
    char                     events[1];
};

typedef struct dague_profiling_key_t {
    char*  name;
    char*  attributes;
    size_t info_length;
    dague_profiling_info_convert_fct_t cnvt;
} dague_profiling_key_t;

/** here key is the key given to the USER */
#define BASE_KEY(key)     ((key) >> 1)
#define EVENT_LENGTH(key, has_info) (sizeof(dague_profiling_output_base_event_t) + \
                                     ((has_info) ? dague_prof_keys[BASE_KEY(key)].info_length : 0))
#define EVENT_HAS_INFO(EV)  ((EV)->event.flags & DAGUE_PROFILING_EVENT_HAS_INFO)

/** here keys are the internal key */
#define START_KEY(key)    (((key) << 1) + 0)
#define END_KEY(key)      (((key) << 1) + 1)

#define FORALL_EVENTS(iterator, start, profile)                         \
    for(iterator = (dague_profiling_output_t *)(start);                 \
        (char *)iterator < (profile)->next_event;                       \
        iterator = (dague_profiling_output_t *)(((char*)iterator) +     \
                                                EVENT_LENGTH(iterator->event.key, EVENT_HAS_INFO(iterator))) )

/* Process-global dictionnary */
static unsigned int dague_prof_keys_count, dague_prof_keys_number;
static dague_profiling_key_t* dague_prof_keys;

/* Process-global profiling list */
static dague_list_t threads;
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

    dague_list_construct( &threads );

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
    res = (dague_thread_profiling_t*)malloc( sizeof(dague_thread_profiling_t) + length );
    if( NULL == res ) {
        ERROR("dague_profiling_thread_init: unable to allocate %u bytes", length);
        return NULL;
    }

    va_start(ap, format);
    vasprintf(&res->hr_id, format, ap);
    va_end(ap);

    res->next_event = res->events;
    res->last_event = NULL;
    res->events_top = res->events + length;
    res->nb_events = 0;

    DAGUE_LIST_ITEM_CONSTRUCT( res );
    dague_list_fifo_push( &threads, (dague_list_item_t*)res );

    return res;
}

int dague_profiling_fini( void )
{
    dague_thread_profiling_t *t;
    
    while( t = (dague_thread_profiling_t*)dague_ulist_fifo_pop(&threads) ) {
        free(t->hr_id);
        free(t);
    }
    free(hr_id);
    dague_list_destruct(&threads);

    dague_profiling_dictionary_flush();
    free(dague_prof_keys);
    dague_prof_keys_number = 0;

    return 0;
}

int dague_profiling_reset( void )
{
    dague_thread_profiling_t *t;
    
    DAGUE_LIST_ITERATOR(&threads, it, {
        t = (dague_thread_profiling_t*)it;
        t->last_event = t->next_event;
        t->next_event = t->events;
    });

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
    dague_profiling_output_t *this_event;
    size_t this_event_length;

    this_event_length = EVENT_LENGTH( key, (NULL != info) );
    if( (context->next_event + this_event_length) > context->events_top ) {
        if( context->next_event <= context->events_top ) {
            WARNING(("Profiling: trace for ID %s will be truncated after %lu events\n",
                    context->hr_id, (unsigned long)context->nb_events));
            context->next_event = context->events_top + 1;
        }
        return -1;
    }
    this_event = (dague_profiling_output_t *)context->next_event;

    context->last_event = context->next_event;
    context->next_event += this_event_length;
    context->nb_events++;

    this_event->event.key   = (uint16_t)key;
    this_event->event.id    = id;
    this_event->event.flags = 0;

    if( NULL != info ) {
        memcpy(this_event->info, info, dague_prof_keys[ BASE_KEY(key) ].info_length);
        this_event->event.flags = DAGUE_PROFILING_EVENT_HAS_INFO;
    }
    this_event->event.timestamp = take_time();    
    
    return 0;
}

static dague_profiling_output_t *find_matching_event_in_profile(const dague_thread_profiling_t *profile,
                                                                const dague_profiling_output_t *start,
                                                                const dague_profiling_output_t *ref)
{
    dague_profiling_output_t *e;
    FORALL_EVENTS(e, start, profile) {
        if( (e->event.id == ref->event.id) &&
            (e->event.key == END_KEY(BASE_KEY(ref->event.key))) &&
            (time_less(ref->event.timestamp, e->event.timestamp) ||
             (diff_time(ref->event.timestamp, e->event.timestamp) == 0)) ) {
            return e;
        }
    }
    return NULL;
}

#if defined(DAGUE_DEBUG_VERBOSE1)
static void dump_whole_trace(void)
{
    const dague_profiling_output_t *event;
    const dague_thread_profiling_t *profile;
    dague_time_t zero = ZERO_TIME;
    int i;

    for( i = 0; i < dague_prof_keys_count; i++ ) {
        if( NULL == dague_prof_keys[i].name ) {
            break;
        }
        DEBUG(("TRACE event [%d:%d] name <%s> attributes <%s> info_length %d\n",
               START_KEY(i), END_KEY(i), dague_prof_keys[i].name, dague_prof_keys[i].attributes, dague_prof_keys[i].info_length ));
    }

    DAGUE_ULIST_ITERATOR(&threads, it, {
        profile = (dague_thread_profiling_t*)it;
        FORALL_EVENTS( event, profile->events, profile ) {
            DEBUG(("TRACE %d/%lu on %p (timestamp %llu)\n", event->event.key, event->event.id, profile,
                   diff_time(zero, event->event.timestamp)));
        }
    });
}
#endif

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
    dague_thread_profiling_t *op;
    const dague_profiling_output_t *start_event, *end_event;

    for( pos = 0; pos < dague_prof_keys_count; pos++ ) {
        displayed_key = 0;
        FORALL_EVENTS( start_event, profile->events, profile ) {
            /* if not my current start_idx key, ignore */
            if( start_event->event.key != START_KEY(pos) )
                continue;
            
            if( (end_event = find_matching_event_in_profile(profile, start_event, start_event)) == NULL ) {
                /* Argh, couldn't find the end in this profile */

                event_not_found = 1;
                /* It has an id, let's look somewhere in another profile, maybe it's end has been
                 * logged by another thread
                 */
                DAGUE_ULIST_ITERATOR(&threads, it, {
                    op = (dague_thread_profiling_t*)it;   
                    if( op == profile )
                        continue;

                    if( (end_event = find_matching_event_in_profile(op, (dague_profiling_output_t*)op->events, start_event)) != NULL ) {
                        event_not_found = 0;
                        break;
                    }
                });

                /* Couldn't find the end, or no id. Bad. */
                if( event_not_found ) {

#if defined(DAGUE_DEBUG_VERBOSE1)
                    dump_whole_trace();
#endif

                    if( !displayed_error_message ) {
                        if( profile->next_event >= profile->events_top ) {
                            WARNING(("Profiling: end event of key %u (%s) id %lu was not found for ID %s\n"
                                    "\t-- some histories are truncated\n",
                                    END_KEY(pos), dague_prof_keys[pos].name, start_event->event.id, profile->hr_id));
                        } else {
                            WARNING(("Profiling: end event of key %u (%s) id %lu was not found for ID %s\n",
                                    END_KEY(pos), dague_prof_keys[pos].name, start_event->event.id, profile->hr_id));
                        }
                        displayed_error_message = 1;
                    }
                    continue;
                }
            }

            start = diff_time( relative, start_event->event.timestamp );
            end = diff_time( relative, end_event->event.timestamp );

            if( displayed_key == 0 ) {
                fprintf(out, "    <KEY ID=\"%u\">\n", pos);
                displayed_key = 1;
            }
            
            fprintf(out, 
                    "     <EVENT>\n"
                    "       <ID>%lu</ID>\n"
                    "       <START>%"PRIu64"</START>\n"
                    "       <END>%"PRIu64"</END>\n",
                    start_event->event.id,
                    start, end);

            if( EVENT_HAS_INFO(start_event) && (NULL != dague_prof_keys[pos].cnvt) ) {
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
            if( EVENT_HAS_INFO(end_event) && (NULL != dague_prof_keys[pos].cnvt) ) {
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
   
    DAGUE_LIST_ITERATOR(&threads, it, {
        profile = (dague_thread_profiling_t*)it;

        if( profile->last_event == NULL ) {
            continue;
        }

        if( !foundone ) {
            relative = ((dague_profiling_output_t *)(profile->events))->event.timestamp;
            latest   = ((dague_profiling_output_t*)(profile->last_event))->event.timestamp;
            foundone = 1;
        } else {
            if( time_less(((dague_profiling_output_t *)(profile->events))->event.timestamp, relative) ) {
                relative = ((dague_profiling_output_t *)(profile->events))->event.timestamp;
            }
            if( time_less( latest, ((dague_profiling_output_t*)(profile->last_event))->event.timestamp) ) {
                latest = ((dague_profiling_output_t*)(profile->last_event))->event.timestamp;
            }
        }
    });

    fprintf(tracefile, " <PROFILES TOTAL_DURATION=\"%"PRIu64"\" TIME_UNIT=\""TIMER_UNIT"\">\n",
            diff_time(relative, latest));

    DAGUE_LIST_ITERATOR(&threads, it, {
        profile = (dague_thread_profiling_t*)it;

        fprintf(tracefile, 
                "   <THREAD>\n"
                "    <IDENTIFIER><![CDATA[%s]]></IDENTIFIER>\n", profile->hr_id);
        dague_profiling_dump_one_xml(profile, tracefile, relative);
        fprintf(tracefile, 
                "   </THREAD>\n");
    });

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
        if( res >= (int)size ) {
            res += nfo.desc->key_to_string( nfo.desc, nfo.id, text, (uint32_t) size );
        } else {
            res += nfo.desc->key_to_string( nfo.desc, nfo.id, text + res, (uint32_t)(size-res) );
        }
        return res;
    }
    return -1;
}
