/*
 * Copyright (c) 2010-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_prof_grapher.h"
#if defined(DAGUE_PROF_TRACE)
#include "dague/dague_binary_profile.h"
#endif

#if defined(DAGUE_PROF_GRAPHER)

#include <stdio.h>
#include <math.h>
#include <errno.h>

FILE *grapher_file = NULL;
static int nbfuncs = -1;
static char **colors = NULL;

/**
 * A simple solution to generate different color tables for each rank. For a
 * more detailed and visualy appealing solution take a look at
 * http://phrogz.net/css/distinct-colors.html
 * and http://en.wikipedia.org/wiki/HSV_color_space
 */
static void HSVtoRGB( double *r, double *g, double *b, double h, double s, double v )
{
    int i;
    double c, x, m;

    c = v * s;
    h /= 60.0;
    i = (int)floor( h );
    x = c * (1 - abs(i % 2 - 1));
    m = v - c;

    switch( i ) {
    case 0:
        *r = c;
        *g = x;
        *b = 0;
        break;
    case 1:
        *r = x;
        *g = c;
        *b = 0;
        break;
    case 2:
        *r = 0;
        *g = c;
        *b = x;
        break;
    case 3:
        *r = 0;
        *g = x;
        *b = c;
        break;
    case 4:
        *r = x;
        *g = 0;
        *b = c;
        break;
    default:		// case 5:
        *r = c;
        *g = 0;
        *b = x;
        break;
    }
    *r += m;
    *g += m;
    *b += m;
}

static inline double get_rand_in_range(int m, int M)
{
    return (double)m + (double)rand() / ((double)RAND_MAX / (M - m + 1) + 1);
}

static char *unique_color(int index, int colorspace)
{
    char color[8];
    double r, g, b;

    double hue = get_rand_in_range(0, 360);  //  0.0 to 360.0
    double saturation = get_rand_in_range(180, 360) / 360.0;  //  0.5 to 1.0, away from white
    double brightness = get_rand_in_range(180, 360) / 360.0;  //  0.5 to 1.0, away from black
    HSVtoRGB(&r, &g, &b, hue, saturation, brightness);
    (void)index; (void)colorspace;
    snprintf(color, 8, "#%02x%02x%02x", (int)floor(255.0*r), (int)floor(255.0*g), (int)floor(255.0*b));
    return strdup(color);
}

void dague_prof_grapher_init(const char *base_filename, int nbthreads)
{
    char *filename;
    int t, size = 1, rank = 0;

#if defined(DISTRIBUTED) && defined(HAVE_MPI)
    char *format;
    int l10 = 0, cs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cs = size;
    while(cs > 0) {
      l10++;
      cs = cs/10;
    }
    asprintf(&format, "%%s-%%0%dd.dot", l10);
    asprintf(&filename, format, base_filename, rank);
    free(format);
#else
    asprintf(&filename, "%s.dot", base_filename);
#endif

    grapher_file = fopen(filename, "w");
    if( NULL == grapher_file ) {
        WARNING(("Grapher:\tunable to create %s (%s) -- DOT graphing disabled\n", filename, strerror(errno)));
        free(filename);
        return;
    } else {
        free(filename);
    }
    fprintf(grapher_file, "digraph G {\n");
    fflush(grapher_file);

    srandom(size*(rank+1));  /* for consistent color generation */
    (void)nbthreads;
    nbfuncs = 128;
    colors = (char**)malloc(nbfuncs * sizeof(char*));
    for(t = 0; t < nbfuncs; t++)
        colors[t] = unique_color(rank * nbfuncs + t, size * nbfuncs);
}

char *dague_prof_grapher_taskid(const dague_execution_context_t *exec_context, char *tmp, int length)
{
    const dague_function_t* function = exec_context->function;
    unsigned int i, index = 0;

    assert( NULL!= exec_context->dague_handle );
    index += snprintf( tmp + index, length - index, "%s_%u", function->name, exec_context->dague_handle->handle_id );
    for( i = 0; i < function->nb_parameters; i++ ) {
        index += snprintf( tmp + index, length - index, "_%d",
                           exec_context->locals[function->params[i]->context_index].value );
    }

    return tmp;
}

void dague_prof_grapher_task(const dague_execution_context_t *context,
                             int thread_id, int vp_id, int task_hash)
{
    if( NULL != grapher_file ) {
        char tmp[MAX_TASK_STRLEN], nmp[MAX_TASK_STRLEN];
        dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context);
        dague_prof_grapher_taskid(context, nmp, MAX_TASK_STRLEN);
#if defined(DAGUE_SIM)
#  if defined(DAGUE_PROF_TRACE)
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\","
                "fontcolor=\"black\",label=\"<%d/%d> %s [%d]\","
                "tooltip=\"hid=%u:did=%d:tname=%s:tid=%d\"];\n",
                nmp, colors[context->function->function_id % nbfuncs],
                thread_id, vp_id, tmp, context->sim_exec_date,
                context->dague_handle->handle_id,
                context->dague_handle->profiling_array != NULL 
                    ? BASE_KEY(context->dague_handle->profiling_array[2*context->function->function_id])
                    : -1,
                context->function->name,
                task_hash);
#  else
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\","
                "fontcolor=\"black\",label=\"<%d/%d> %s [%d]\","
                "tooltip=\"hid=%u:tname=%s:tid=%d\"];\n",
                nmp, colors[context->function->function_id % nbfuncs],
                thread_id, vp_id, tmp, context->sim_exec_date,
                context->dague_handle->handle_id,
                context->function->name,
                task_hash);
#  endif
#else
#  if defined(DAGUE_PROF_TRACE)
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\","
                "fontcolor=\"black\",label=\"<%d/%d> %s\","
                "tooltip=\"hid=%u:did=%d:tname=%s:tid=%d\"];\n",
                nmp, colors[context->function->function_id % nbfuncs],
                thread_id, vp_id, tmp,
                context->dague_handle->handle_id,
                context->dague_handle->profiling_array != NULL 
                    ? BASE_KEY(context->dague_handle->profiling_array[2*context->function->function_id])
                    : -1,
                context->function->name,
                task_hash);
#  else
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\","
                "fontcolor=\"black\",label=\"<%d/%d> %s\","
                "tooltip=\"hid=%u:tname=%s:tid=%d\"];\n",
                nmp, colors[context->function->function_id % nbfuncs],
                thread_id, vp_id, tmp,
                context->dague_handle->handle_id,
                context->function->name,
                task_hash);
#  endif
#endif
        fflush(grapher_file);
    }
}

void dague_prof_grapher_dep(const dague_execution_context_t* from, const dague_execution_context_t* to,
                            int dependency_activates_task,
                            const dague_flow_t* origin_flow, const dague_flow_t* dest_flow)
{
    if( NULL != grapher_file ) {
        char tmp[128];
        int index = 0;

        dague_prof_grapher_taskid( from, tmp, 128 );
        index = strlen(tmp);
        index += snprintf( tmp + index, 128 - index, " -> " );
        dague_prof_grapher_taskid( to, tmp + index, 128 - index - 4 );
        fprintf(grapher_file,
                "%s [label=\"%s=>%s\",color=\"#%s\",style=\"%s\"]\n",
                tmp, origin_flow->name, dest_flow->name,
                dependency_activates_task ? "00FF00" : "FF0000",
                ((dest_flow->flow_flags == FLOW_ACCESS_NONE) ? "dotted":
                 (dest_flow->flow_flags == FLOW_ACCESS_RW) ? "solid" : "dashed"));
        fflush(grapher_file);
    }
}

void dague_prof_grapher_fini(void)
{
    int t;

    if( NULL == grapher_file ) return;

    fprintf(grapher_file, "}\n");
    fclose(grapher_file);
    for(t = 0; t < nbfuncs; t++)
        free(colors[t]);
    free(colors);
    colors = NULL;
    grapher_file = NULL;
}

#endif /* DAGUE_PROF_GRAPHER */
