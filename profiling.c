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
#endif

typedef struct dplasma_profiling_key_t {
    char* name;
    char* attributes;
} dplasma_profiling_key_t;

typedef struct dplasma_profiling_output_t {
    int key;
    dplasma_time_t timestamp;
} dplasma_profiling_output_t;

int dplasma_prof_events_count, dplasma_prof_events_number;
dplasma_profiling_output_t* dplasma_prof_events;

int dplasma_prof_keys_count, dplasma_prof_keys_number;
dplasma_profiling_key_t* dplasma_prof_keys;

int dplasma_profiling_init( size_t length )
{
    int i;

    dplasma_prof_events_number = length;
    dplasma_prof_events_count = 0;
    dplasma_prof_events = (dplasma_profiling_output_t*)malloc(sizeof(dplasma_profiling_output_t) * length);

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
int dplasma_profiling_fini( void )
{
    int i;

    dplasma_prof_events_number = 0;
    dplasma_prof_events_count = 0;
    free(dplasma_prof_events);
    dplasma_prof_events = NULL;

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

int dplasma_profiling_trace( int key )
{
    int my_event = dplasma_atomic_inc_32b(&dplasma_prof_events_count);

    if( my_event >= dplasma_prof_events_number ) {
        return -1;
    }
    dplasma_prof_events[my_event].key = key;
    dplasma_prof_events[my_event].timestamp = take_time();
    return 0;
}

int dplasma_profiling_dump_svg( const char* filename )
{
    FILE* tracefile;
    uint64_t start, end;
    dplasma_time_t relative;
    double scale = 0.01, gaps = 0.0, gaps_last = 0.0, last;
    int i, tag, core;

    tracefile = fopen(filename, "w");
    if( NULL == tracefile ) {
        return -1;
    }

    core = 1;

    fprintf(tracefile,
            "<?xml version=\"1.0\" standalone=\"no\"?>\n"
            "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n" 
            "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
            "<svg width=\"50000\" height=\"%d\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            400 * (core+1));

    relative = dplasma_prof_events[1].timestamp;
    printf("relative = %llu\n", relative);
    last = diff_time( relative, dplasma_prof_events[0].timestamp );
    for( i = 1; i < dplasma_prof_events_count; i+=2 ) {
        start = diff_time( relative, dplasma_prof_events[i].timestamp );
        end = diff_time( relative, dplasma_prof_events[i+1].timestamp );
        tag = dplasma_prof_events[i].key / 2;

        gaps += start - gaps_last;
        gaps_last = end;

        if( last < end ) last = end;

        fprintf(tracefile,
                "    "
                "<rect x=\"%.2lf\" y=\"%.0lf\" width=\"%.2lf\" height=\"%.0lf\" "
                "style=\"%s;stroke-width:1\"/>\n",
                start * scale,
                (core - 1) * 100.0,
                (end - start) * scale,
                200.0,
                dplasma_prof_keys[tag].attributes);

        /*        fprintf(tracefile,
                "    "
                "<text x=\"%.2lf\" y=\"%.0lf\" font-size=\"20\" fill=\"black\">"
                "%d"
                "</text>\n",
                start * scale + 10,
                core * 100.0 + 20,
                (int)tag);*/
    }
    fprintf(tracefile, "  </svg>\n");

    fclose(tracefile);

    printf("Found %.4lf ticks gaps out of %.4lf (%.2lf%%)\n", gaps,
           last, (gaps * 100.0) / last);
    return 0;
}

