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

#define TIMER_UNIT "nanosecond"
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
#define TIMER_UNIT "cycles"
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
#define TIMER_UNIT "cycles"
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
#define TIMER_UNIT "microseconds"
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
    unsigned long id;
    dplasma_time_t timestamp;
#if defined(USE_PAPI)
    long long counter_value;
#endif /* defined(USE_PAPI) */
} dplasma_profiling_output_t;

#define START_KEY(key)  ( (key) * 2 )
#define END_KEY(key)    ( (key) * 2 + 1 )

int dplasma_prof_events_number;
int dplasma_prof_keys_count, dplasma_prof_keys_number;
dplasma_profiling_key_t* dplasma_prof_keys;

typedef struct dplasma_eu_profiling_t {
    int events_count;
    dplasma_profiling_output_t events[1];
} dplasma_eu_profiling_t;


#ifdef USE_MPI
#define EXTRA_CTX 1
#else 
#define EXTRA_CTX 0
#endif

int dplasma_profiling_init( dplasma_context_t* context, size_t length )
{
    dplasma_eu_profiling_t* prof;
    int i;

    dplasma_prof_events_number = length;

    for( i = 0; i < (context->nb_cores + EXTRA_CTX); i++ ) {
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

    for( i = 0; i < (context->nb_cores + EXTRA_CTX); i++ ) {
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
            *key_start = START_KEY(i);
            *key_end = END_KEY(i);
            return 0;
        }
    }
    if( -1 == pos ) {
        pos = dplasma_prof_keys_count;
    }

    dplasma_prof_keys[pos].name = strdup(key_name);
    dplasma_prof_keys[pos].attributes = strdup(attributes);

    *key_start = START_KEY(pos);
    *key_end = END_KEY(pos);
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
    return 0;
}

int dplasma_profiling_trace( dplasma_execution_unit_t* context, int key, unsigned long id )
{
    int my_event = context->eu_profile->events_count++;

    if( my_event >= dplasma_prof_events_number ) {
        return -1;
    }
    context->eu_profile->events[my_event].key = key;
    context->eu_profile->events[my_event].id  = id;
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
            "<!DOCTYPE svg PUBLIC \"-_W3C_DTD SVG 1.0_EN\" \"http://www.w3.org/TR/SVG/DTD/svg10.dtd\">\n"
            "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' \n"
            " onload=\"startup(evt)\"\n"
            " width='%g'\n"
            " height='%g'>\n"
            "  <script><![CDATA[\n"
            "var svgDoc;\n"
            "var Root;\n"
            "var scale=1;\n"
            "var translate=1;\n"
            "var xmlns=\"http://www.w3.org/2000/svg\"\n"
            "var cursel=undefined;\n"
            "var oldSelStyle=\"\";\n"
            "function startup(evt){\n"
            "  O=evt.target\n"
            "  svgDoc=O.ownerDocument;\n"
            "  Root=svgDoc.documentElement;\n"
            "  O.setAttribute(\"onmousemove\",\"adjust(evt)\")\n"
            "  O.setAttribute(\"onmousedown\",\"recolor(evt)\")\n"
            "  top.svgzoom = svgzoom\n"
            "  top.svgtranslate = svgtranslate\n"
            "  top.svg_outside_select = outsideSelect\n"
            "  top.ready()\n"
            "}\n"
            "function outsideSelect(x){\n"
            "  if( cursel != undefined ) {\n"
            "      cursel.setAttribute(\"style\", oldSelStyle);\n"
            "      cursel = undefined;\n"
            "      oldSelStyle = \"\";\n"
            "  }\n"
            "  cursel = svgDoc.getElementById(x);\n"
            "  if( !cursel ) {\n"
            "    opera.postError(\"dposv.svg warning: unable to find the element named \" + x);\n"
            "  } else {\n"
            "    oldSelStyle = cursel.getAttribute(\"style\");\n"
            "    cursel.setAttribute(\"style\", \"fill:#FFCC00\");\n"
            "  }\n"
            "}\n"
            "function recolor(evt){\n"
            "  if( cursel != undefined ) {\n"
            "      cursel.setAttribute(\"style\", oldSelStyle);\n"
            "      cursel = undefined;\n"
            "      oldSelStyle = \"\";\n"
            "  }\n"
            "\n"
            "  if( evt.target.getElementsByTagName('FID').item(0) &&\n"
            "      evt.target.getElementsByTagName('FID').item(0).firstChild &&\n"
            "      evt.target.getElementsByTagName('FID').item(0).firstChild.nodeValue != \"\" ) {\n"
            "      \n"
            "      cursel = evt.target;\n"
            "      oldSelStyle = cursel.getAttribute(\"style\");\n"
            "      cursel.setAttribute(\"style\", \"fill:#FFCC00\");\n"
            "      top.select_function(evt.target.getElementsByTagName('FName').item(0).firstChild.nodeValue +\n"
            "                          evt.target.getElementsByTagName('FID').item(0).firstChild.nodeValue);\n"
            "  } else {\n"
            "      top.select_function("");\n"
            "  }\n"
            "}\n"
            "function adjust(evt){\n"
            "  if( evt.target.getElementsByTagName('FName').item(0) &&\n"
            "      evt.target.getElementsByTagName('FName').item(0).firstChild ) {\n"
            "    targetFName = evt.target.getElementsByTagName('FName').item(0).firstChild.nodeValue;\n"
            "  } else {\n"
            "    targetFName = \"\";\n"
            "  }\n"
            "  if( evt.target.getElementsByTagName('FDesc').item(0) &&\n"
            "      evt.target.getElementsByTagName('FDesc').item(0).firstChild ) {\n"
            "    targetFDesc = evt.target.getElementsByTagName('FDesc').item(0).firstChild.nodeValue;\n"
            "  } else {\n"
            "    targetFDesc = \"\";\n"
            "  }\n"
            "  top.mouseMove(targetFName, targetFDesc)\n"
            "}\n"
            "function svgzoom( x ) {\n"
            "  scale=x;\n"
            "  svgDoc.getElementById('gantt').setAttribute(\"transform\", \"scale(\" + scale + \", 1) translate(\" + translate + \", 0)\");\n"
            "}\n"
            "function svgtranslate( x ) {\n"
            "  translate=-x;\n"
            "  svgDoc.getElementById('gantt').setAttribute(\"transform\", \"scale(\" + scale + \", 1) translate(\" + translate + \", 0)\");\n"
            "}\n"
            "//]]>\n"
            "  </script>\n"
            "    <rect x='0' y='0' width='100%%' height='100%%' fill='white'>\n"
            "      <FName><![CDATA[]]></FName>\n"
            "      <FDesc><![CDATA[]]></FDesc>\n"
            "    </rect>\n"
            "    <g id=\"gantt\" transform=\"scale(1) translate(1, 0)\">\n",
            WIDTH,
            (context->nb_cores + EXTRA_CTX) * CORE_STRIDE +  dplasma_prof_keys_count*20);

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

    fprintf(tracefile, "    <rect x=\"0.00\" y=\"18.0\" width=\"%.2lf\" height=\"1.0\" />\n", WIDTH);
    for( i = 0; i <= WIDTH; i += WIDTH/20 ) {
        fprintf(tracefile, "    <rect x=\"%.2lf\" y=\"14.0\" width=\"1.0\" height=\"8.0\" />\n", (double)i);
                
        fprintf(tracefile,
                "    <text x=\"%.2lf\" y=\"12.0\">%.2lf</text>\n",
                (double)i, i*scale);
    }

    for( thread_id = 0; thread_id < context->nb_cores + EXTRA_CTX; thread_id++ ) {
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
                    "    <rect x=\"%.2lf\" y=\"%.0lf\" width=\"%.2lf\" height=\"%.0lf\" style=\"%s\" id='%s%lu'>\n"
                    "       <FName>%s</FName>\n"
                    "       <FDesc>%.0lf time units (%.2lf%% of time)</FDesc>\n"
                    "       <FID>%lu</FID>\n"
                    "    </rect>\n",                
                    start * scale,
                    thread_id * CORE_STRIDE + 25.0,
                    (end - start) * scale,
                    CORE_STRIDE,
                    dplasma_prof_keys[tag].attributes,
                    dplasma_prof_keys[tag].name,
                    profile->events[i].id,
                    dplasma_prof_keys[tag].name,
                    (double)end-(double)start,
                    100.0 * ( (double)end-(double)start) / (double)total_time,
                    profile->events[i].id);
        }
        if( 0 != i ) {
            printf("Found %lu ticks gaps out of %lu (%.2lf%%)\n", (unsigned long)gaps,
                   (unsigned long)last, (gaps * 100.0) / (double)last);
        }
    }

    fprintf(tracefile,
            "  </g>\n");

    nplot = 0;
    for( key = 0; key < dplasma_prof_keys_count; key++ ) {
        int key_start = 2*key, key_end = 2*key + 1;

        keyplotted = 0;

        for(  thread_id = 0; thread_id < context->nb_cores + EXTRA_CTX; thread_id++ ) {
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
            var = ((double) sqsum - (double)sum * avg) / ((double)nb - 1.0);

            if( !keyplotted && (sum > 0)) {
                int ptid;
                fprintf(tracefile,
                        "  <rect x='%d' y='%d' width='20' height='10' style='%s' />\n"
                        "  <text x='%d' y='%d'>%s",
                        (nplot /  dplasma_prof_keys_count) * 400 + 10,
                        (nplot %  dplasma_prof_keys_count) * 15 + 33 + (context->nb_cores + EXTRA_CTX) * (int)CORE_STRIDE,
                        dplasma_prof_keys[key].attributes,
                        (nplot /  dplasma_prof_keys_count) * 400 + 35,
                        (nplot %  dplasma_prof_keys_count) * 15 + 43 + (context->nb_cores + EXTRA_CTX) * (int)CORE_STRIDE,
                        dplasma_prof_keys[key].name);
                for( ptid = 0; ptid < thread_id; ptid++) {
                    fprintf(tracefile, "nan(nan)%c", (ptid + 1 == (context->nb_cores + EXTRA_CTX)) ? ' ' : '/');
                }
                fprintf(tracefile,
                        "  %.2lf(%.2lf)%c",
                        avg,
                        sqrt(var),
                        (thread_id + 1 == (context->nb_cores + EXTRA_CTX)) ? ' ' : '/');
                keyplotted = 1;
                nplot++;
            } else if( keyplotted ) {
                fprintf(tracefile,
                        "  %.2lf(%.2lf)%c",
                        avg,
                        sqrt(var),
                        (thread_id + 1 == (context->nb_cores + EXTRA_CTX)) ? ' ' : '/');
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

static int dplasma_profiling_dump_one_xml( const dplasma_eu_profiling_t *profile, 
                                           FILE *out,
                                           dplasma_time_t relative )
{
    int key, start_idx, end_idx, displayed_key;
    uint64_t start, end;
    
    for( key = 0; key < dplasma_prof_keys_count; key++ ) {
        displayed_key = 0;
        for( start_idx = 0; start_idx < min(profile->events_count, dplasma_prof_events_number); start_idx++ ) {
            /* if not my current start_idx key, ignore */
            if( profile->events[start_idx].key != START_KEY(key) )
                continue;
            
            /* find the end_idx event */
            for( end_idx = start_idx+1; end_idx < min(profile->events_count, dplasma_prof_events_number); end_idx++) {
                if( (profile->events[end_idx].key == END_KEY(key)) &&
                    (profile->events[end_idx].id == profile->events[start_idx].id) )
                    break;
            }
            if( end_idx == min(profile->events_count, dplasma_prof_events_number) ) {
                for(end_idx = 0; end_idx < start_idx; end_idx++) {
                    if( (profile->events[end_idx].key == END_KEY(key)) &&
                        (profile->events[end_idx].id == profile->events[start_idx].id) ) {
                        fprintf(stderr, "Profiling warning: end_idx event of key %d id %lu was found before the corresponding start event\n",
                                key, profile->events[end_idx].id);
                        break;
                    }
                }
                if( end_idx == start_idx ) {
                    fprintf(stderr, "Profiling error: end event of key %d id %lu was not found\n", key, profile->events[end_idx].id);
                    return -1;
                }
            }

            start = diff_time( relative, profile->events[start_idx].timestamp );
            end = diff_time( relative, profile->events[end_idx].timestamp );

            if( displayed_key == 0 ) {
                fprintf(out, "    <KEY ID=\"%d\">\n", key);
                displayed_key = 1;
            }
            
            fprintf(out, "     <EVENT>\n");

            fprintf(out, "       <ID>%lu</ID>\n"
                         "       <START>%llu</START>\n"
                         "       <END>%llu</END>\n",
                    profile->events[start_idx].id,
                    start, end);
#ifdef USE_PAPI
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

int dplasma_profiling_dump_xml( dplasma_context_t* context, const char* filename )
{
    int i, thread_id, last_timestamp, foundone;
    dplasma_time_t relative = ZERO_TIME, latest = ZERO_TIME;
    dplasma_eu_profiling_t* profile;
    FILE* tracefile;

    tracefile = fopen(filename, "w");
    if( NULL == tracefile ) {
        return -1;
    }

    fprintf(tracefile,
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<PROFILING>\n"
            " <DICTIONARY>\n");

    for(i = 0; i < dplasma_prof_keys_count; i++) {
        fprintf(tracefile,
                "   <KEY ID=\"%d\">\n"
                "    <NAME>%s</NAME>\n"
                "    <ATTRIBUTES><![CDATA[%s]]></ATTRIBUTES>\n"
                "   </KEY>\n",
                i, dplasma_prof_keys[i].name, dplasma_prof_keys[i].attributes);
    }
    fprintf(tracefile, " </DICTIONNARY>\n");

    foundone = 0;
    for( thread_id = 0; thread_id < context->nb_cores + EXTRA_CTX; thread_id++ ) {
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

    fprintf(tracefile, " <PROFILES TOTAL_DURATION=\"%llu\" TIME_UNIT=\""TIMER_UNIT"\">\n",
            diff_time(relative, latest));

    for( thread_id = 0; thread_id < context->nb_cores + EXTRA_CTX; thread_id++) {
        fprintf(tracefile, "   <THREAD ID=\"%d\">\n", thread_id);
        dplasma_profiling_dump_one_xml(context->execution_units[thread_id].eu_profile, tracefile, relative);
        fprintf(tracefile, "   </THREAD>\n");
    }

    fprintf(tracefile, 
            " </PROFILES>\n"
            "</PROFILING>\n");
    fclose(tracefile);
    
    return 0;
}

