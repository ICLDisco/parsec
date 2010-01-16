/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "profiling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "atomic.h"
#include "tooltip.h"

#define WIDTH 800.0
#define CORE_STRIDE 25.0

#define min(a, b) ((a)<(b)?(a):(b))
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
    diff = (end.tv_sec - start.tv_sec) * 1000000000 +
           (end.tv_nsec - start.tv_nsec);
    return diff;
}

static int time_less( dplasma_time_t start, dplasma_time_t end )
{
    return start.tv_sec < end.tv_sec ||
        (start.tv_sec == end.tv_sec &&
         start.tv_nsec < end.tv_nsec);
}
#define ZERO_TIME {0,0}
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
#define ZERO_TIME 0
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
#define ZERO_TIME 0
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
#define ZERO_TIME {0,0}
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
    int i;

    free(dplasma_prof_keys[key].name);
    dplasma_prof_keys[key].name = NULL;
    free(dplasma_prof_keys[key].attributes);
    dplasma_prof_keys[key].attributes = NULL;

    /* Update the number of active/registered keys */
    for( i = key; i < dplasma_prof_keys_count; i++ )
        if( NULL != dplasma_prof_keys[i].name ) {
            return 0;
        }
    dplasma_prof_keys_count = key;

    return 0;
}

int dplasma_profiling_reset( dplasma_execution_unit_t* context )
{
    context->eu_profile->events_count = 0;
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
    int i, thread_id, tag, last_timestamp, key, keyplotted, nplot, foundone;
    uint64_t start, end, total_time, last, gaps, gaps_last;
    dplasma_time_t relative = ZERO_TIME, latest = ZERO_TIME;
    dplasma_eu_profiling_t* profile;
    FILE* tracefile;
    double scale;

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
            "  width='%lf'\n"
            "  height='%lf'\n"
            ">\n"
            "  <script type=\"text/ecmascript\">\n"
            "    <![CDATA[%s]]>\n"
            "  </script>\n"
            "    <rect x='0' y='0' width='100%%' height='100%%' fill='white'>\n"
            "      <title><![CDATA[]]></title>\n"
            "      <desc><![CDATA[]]></desc>\n"
            "    </rect>\n\n",
            WIDTH,
            context->nb_cores * CORE_STRIDE +  dplasma_prof_keys_count*20,
            tooltip_script);

    foundone = 0;
    for( thread_id = 0; thread_id < context->nb_cores; thread_id++ ) {
        profile = context->execution_units[thread_id].eu_profile;

        if( profile->events_count == 0 ) {
            continue;
        }

        if( !foundone ) {
            relative = profile->events[0].timestamp;
            last_timestamp = min(profile->events_count, dplasma_prof_events_number) - 1;
            latest   = profile->events[last_timestamp].timestamp;
            foundone = 1;
        } else {
            if( time_less(profile->events[0].timestamp, relative) ) {
                relative = profile->events[0].timestamp;
            }
            last_timestamp = min(profile->events_count, dplasma_prof_events_number) - 1;
            if( time_less( latest, profile->events[last_timestamp].timestamp) ) {
                latest = profile->events[last_timestamp].timestamp;
            }
        }
    }

    scale = WIDTH / (double)diff_time(relative, latest);

    for( thread_id = 0; thread_id < context->nb_cores; thread_id++ ) {
        profile = context->execution_units[thread_id].eu_profile;
        gaps = 0.0;
        gaps_last = 0.0;
        total_time = diff_time(relative, profile->events[min(profile->events_count-1, dplasma_prof_events_number-1)].timestamp);
        last = diff_time( relative, profile->events[1].timestamp );
        for( i = 0; i < min(profile->events_count, dplasma_prof_events_number); i+=2 ) {
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
                    thread_id * CORE_STRIDE + 2.0,
                    (end - start) * scale,
                    CORE_STRIDE,
                    dplasma_prof_keys[tag].attributes,
                    dplasma_prof_keys[tag].name,
                    (double)end-(double)start,
                    100.0 * ( (double)end-(double)start) / (double)total_time);
        }
        printf("Found %lu ticks gaps out of %lu (%.2lf%%)\n", (unsigned long)gaps,
               (unsigned long)last, (gaps * 100.0) / (double)last);
    }
    fprintf(tracefile, 
            "  <g id='ToolTip' opacity='0.8' display='none' pointer-events='none'>\n"
            "    <rect id='tipbox' x='0' y='5' width='88' height='20' rx='2' ry='2' fill='white' stroke='black'/>\n"
            "    <text id='tipText' x='5' y='20' font-family='Arial' font-size='12'>\n"
            "      <tspan id='tipTitle' x='5' font-weight='bold'><![CDATA[]]></tspan>\n"
            "      <tspan id='tipDesc' x='5' dy='1.2em' fill='blue'><![CDATA[]]></tspan>\n"
            "    </text>\n"
            "  </g>\n");

    nplot = 0;
    for( key = 0; key < dplasma_prof_keys_count; key++ ) {
        int key_start = 2*key, key_end = 2*key + 1;

        keyplotted = 0;

        for(  thread_id = 0; thread_id < context->nb_cores; thread_id++ ) {
            uint64_t time, sum = 0, sqsum = 0;
            double avg, var;
            int nb = 0;

            profile = context->execution_units[thread_id].eu_profile;
            for( i = 0; i < min(profile->events_count, dplasma_prof_events_number); i+=2 ) {
                if( profile->events[i].key == key_start ) {
                    assert( profile->events[i+1].key == key_end);

                    time = diff_time( profile->events[i].timestamp, profile->events[i+1].timestamp );
                    sum += time;
                    sqsum += time*time;
                    nb++;
                } 
            }
            
            avg = (double) sum / (double) nb;
            var = (double) sqsum - (double) nb * avg * avg;

            if( !keyplotted && (sum > 0)) {
                int ptid;
                fprintf(tracefile,
                        "  <rect x='%d' y='%d' width='20' height='10' style='%s' />\n"
                        "  <text x='%d' y='%d'>%s",
                        (nplot /  dplasma_prof_keys_count) * 400 + 10,
                        (nplot %  dplasma_prof_keys_count) * 15 + 10 + context->nb_cores * (int)CORE_STRIDE,
                        dplasma_prof_keys[key].attributes,
                        (nplot /  dplasma_prof_keys_count) * 400 + 35,
                        (nplot %  dplasma_prof_keys_count) * 15 + 20 + context->nb_cores * (int)CORE_STRIDE,
                        dplasma_prof_keys[key].name);
                for( ptid = 0; ptid < thread_id; ptid++) {
                    fprintf(tracefile, "nan(nan)%c", (ptid + 1 == context->nb_cores) ? ' ' : '/');
                }
                fprintf(tracefile,
                        "  %.2lf(%.2lf)%c",
                        avg,
                        sqrt(var),
                        (thread_id + 1 == context->nb_cores) ? ' ' : '/');
                keyplotted = 1;
                nplot++;
            } else if( keyplotted ) {
                fprintf(tracefile,
                        "  %.2lf(%.2lf)%c",
                        avg,
                        sqrt(var),
                        (thread_id + 1 == context->nb_cores) ? ' ' : '/');
            }
        }
        if( keyplotted ) {
            fprintf(tracefile, "</text>\n");
        }
    }

    fprintf(tracefile,
            "</svg>\n");
    fclose(tracefile);
    
    return 0;
}

