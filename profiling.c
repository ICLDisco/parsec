/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifdef __gnu_linux__
/* We need it for the high resolution timers on Linux */
#define _GNU_SOURCE
#endif  /* __gnu_linux__ */

#include "profiling.h"
#include <stdlib.h>
#include <string.h>

#include "atomic.h"
#include "tooltip.h"

#if defined(__gnu_linux__) && !defined INTEL 
#include <unistd.h>
#include <time.h>
typedef struct timespec dplasma_time_t;
static inline dplasma_time_t take_time(void)
{
    dplasma_time_t ret;
    clock_gettime(CLOCK_REALTIME, &ret);
    return ret;
}

static inline uint64_t diff_time( dplasma_time_t start, dplasma_time_t end )
{
    uint64_t diff;
    diff = (end.tv_sec - start.tv_sec) * 1000000 +
           (end.tv_nsec - start.tv_nsec);
    return diff;
}

static int time_less( dplasma_time_t start, dplasma_time_t end )
{
    return start.tv_sec < end.tv_sec ||
        (start.tv_sec == end.tv_sec &&
         start.tv_usec < end.tv_usec);
}
#elif defined(__IA64)
typedef uint64_t dplasma_time_t;
static inline dplasma_time_t take_time(void)
{
    dplasma_time_t ret;
    __asm__ __volatile__ ("mov %0=ar.itc" : "=r"(ret));
    return ret;
}
static inline uint64_t diff_time( dplasma_time_t start, dplasma_time_t end )
{
    return (end - start);
}
static int time_less( dplasma_time_t start, dplasma_time_t end )
{
    return start < end;
}
#elif defined(__X86)
typedef uint64_t dplasma_time_t;
static inline dplasma_time_t take_time(void)
{
    dplasma_time_t ret;
    __asm__ __volatile__("rdtsc" : "=A"(ret));
    return ret;
}
static inline uint64_t diff_time( dplasma_time_t start, dplasma_time_t end )
{
    return (end - start);
}
static int time_less( dplasma_time_t start, dplasma_time_t end )
{
    return start < end;
}
#else
#include <sys/time.h>
typedef struct timeval dplasma_time_t;
static inline dplasma_time_t take_time(void)
{
    struct timeval tv;

    gettimeofday( &tv, NULL );
    return tv;
}
static inline uint64_t diff_time( dplasma_time_t start, dplasma_time_t end )
{
    uint64_t diff;
    diff = (end.tv_sec - start.tv_sec) * 1000000 +
           (end.tv_usec - start.tv_usec);
    return diff;
}
static int time_less( dplasma_time_t start, dplasma_time_t end )
{
    return start.tv_sec < end.tv_sec ||
        (start.tv_sec == end.tv_sec &&
         start.tv_usec < end.tv_usec);
}
#endif

typedef struct dplasma_profiling_key_t {
    char* name;
    char* attributes;
} dplasma_profiling_key_t;

typedef struct dplasma_profiling_output_t {
    int key;
    dplasma_time_t timestamp;
} dplasma_profiling_output_t;

int dplasma_prof_events_number;
int dplasma_prof_keys_count, dplasma_prof_keys_number;
dplasma_profiling_key_t* dplasma_prof_keys;

typedef struct dplasma_eu_profiling_t {
    int events_count;
    dplasma_profiling_output_t events[1];
} dplasma_eu_profiling_t;

int dplasma_profiling_init( dplasma_context_t* context, size_t length )
{
    dplasma_eu_profiling_t* prof;
    int i;

    dplasma_prof_events_number = length;

    for( i = 0; i < context->nb_cores; i++ ) {
        prof = (dplasma_eu_profiling_t*)malloc(sizeof(dplasma_eu_profiling_t) +
                                               sizeof(dplasma_profiling_output_t) * length);
        prof->events_count = 0;
        context->execution_units[i].eu_profile = prof;
    }

    dplasma_prof_keys = (dplasma_profiling_key_t*)malloc(128 * sizeof(dplasma_profiling_key_t));
    dplasma_prof_keys_count = 0;
    dplasma_prof_keys_number = 128;
    for( i = 0; i < dplasma_prof_keys_number; i++ ) {
        dplasma_prof_keys[i].name = NULL;
        dplasma_prof_keys[i].attributes = NULL;
    }
    return 0;
}

/**
 * Release all resources for the tracing. If threads are enabled only
 * the resources related to this thread are released.
 */
int dplasma_profiling_fini( dplasma_context_t* context )
{
    int i;

    dplasma_prof_events_number = 0;

    for( i = 0; i < dplasma_prof_keys_number; i++ ) {
        if( NULL != dplasma_prof_keys[i].name ) {
            free(dplasma_prof_keys[i].name);
            free(dplasma_prof_keys[i].attributes);
        }
    }
    free(dplasma_prof_keys);
    dplasma_prof_keys = NULL;
    dplasma_prof_keys_count = 0;
    dplasma_prof_keys_number = 0;

    for( i = 0; i < context->nb_cores; i++ ) {
        free(context->execution_units[i].eu_profile);
        context->execution_units[i].eu_profile = NULL;
    }

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
            *key_start = 2 * i;
            *key_end = 2 * i + 1;
            return 0;
        }
    }
    if( -1 == pos ) {
        pos = dplasma_prof_keys_count;
    }

    dplasma_prof_keys[pos].name = strdup(key_name);
    dplasma_prof_keys[pos].attributes = strdup(attributes);

    *key_start = 2 * pos;
    *key_end = 2 * pos + 1;
    dplasma_prof_keys_count++;
    return 0;
}

int dplasma_profiling_del_dictionary_keyword( int key )
{
    return 0;
}

int dplasma_profiling_trace( dplasma_execution_unit_t* context, int key )
{
    int my_event = context->eu_profile->events_count++;

    if( my_event >= dplasma_prof_events_number ) {
        return -1;
    }
    context->eu_profile->events[my_event].key = key;
    context->eu_profile->events[my_event].timestamp = take_time();
    
    return 0;
}

int dplasma_profiling_dump_svg( dplasma_context_t* context, const char* filename )
{
    FILE* tracefile;
    uint64_t start, end;
    dplasma_time_t relative;
    double scale = 0.01, gaps, gaps_last, last, total_time;
    dplasma_eu_profiling_t* profile;
    int i, thread_id, tag;

    tracefile = fopen(filename, "w");
    if( NULL == tracefile ) {
        return -1;
    }

    fprintf(tracefile,
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<!DOCTYPE svg PUBLIC \"-_W3C_DTD SVG 1.0_EN\"\n" 
            "\"http://www.w3.org/TR/SVG/DTD/svg10.dtd\">\n"
            "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'\n"
            "  onload='Init(evt)'\n"
            "  onmousemove='GetTrueCoords(evt); ShowTooltip(evt, true)'\n"
            "  onmouseout='ShowTooltip(evt, false)'\n"
            ">\n"
            "  <script type=\"text/ecmascript\">\n"
            "    <![CDATA[%s]]>\n"
            "  </script>\n",
            tooltip_script);

    relative = context->execution_units[0].eu_profile->events[0].timestamp;
    for( thread_id = 1; thread_id < context->nb_cores; thread_id++ ) {
        if( time_less(context->execution_units[0].eu_profile->events[0].timestamp, relative) ) {
            relative = profile->events[0].timestamp;
        }
    }

    for( thread_id = 0; thread_id < context->nb_cores; thread_id++ ) {
        profile = context->execution_units[thread_id].eu_profile;
        gaps = 0.0;
        gaps_last = 0.0;
        total_time = diff_time(relative, profile->events[profile->events_count-1].timestamp);
        last = diff_time( relative, profile->events[1].timestamp );
        for( i = 0; i < profile->events_count; i+=2 ) {
            start = diff_time( relative, profile->events[i].timestamp );
            end = diff_time( relative, profile->events[i+1].timestamp );
            tag = profile->events[i].key / 2;
            
            gaps += start - gaps_last;
            gaps_last = end;
            
            if( last < end ) last = end;
            
            fprintf(tracefile,
                    "    <rect x=\"%.2lf\" y=\"%.0lf\" width=\"%.2lf\" height=\"%.0lf\" style=\"%s\">\n"
                    "       <title>%s</title>\n"
                    "       <desc>%.0lf time units (%.2lf%% of time)</desc>\n"
                    "    </rect>\n",                
                    start * scale,
                    thread_id * 100.0 + 1.0,
                    (end - start) * scale,
                    98.0,
                    dplasma_prof_keys[tag].attributes,
                    dplasma_prof_keys[tag].name,
                    (double)end-(double)start,
                    100.0 * ( (double)end-(double)start) / (double)total_time);
            
            printf("Found %.4lf ticks gaps out of %.4lf (%.2lf%%)\n", gaps,
                   last, (gaps * 100.0) / last);
        }
    }
    fprintf(tracefile, 
            "  <g id='ToolTip' opacity='0.8' display='none' pointer-events='none'>\n"
            "    <rect id='tipbox' x='0' y='5' width='88' height='20' rx='2' ry='2' fill='white' stroke='black'/>\n"
            "    <text id='tipText' x='5' y='20' font-family='Arial' font-size='12'>\n"
            "      <tspan id='tipTitle' x='5' font-weight='bold'><![CDATA[]]></tspan>\n"
            "      <tspan id='tipDesc' x='5' dy='1.2em' fill='blue'><![CDATA[]]></tspan>\n"
            "    </text>\n"
            "  </g>\n"
            "</svg>\n");
    fclose(tracefile);
    
    return 0;
}

