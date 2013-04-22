/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_prof_grapher.h"

#if defined(DAGUE_PROF_GRAPHER)

#include <stdio.h>
#include <math.h>

static FILE *grapher_file = NULL;
static int nbfuncs = -1;
static char **colors = NULL;

static void HSVtoRGB( double *r, double *g, double *b, double h, double s, double v )
{
	int i;
	double f, p, q, t;
	if( s == 0 ) {
		// achromatic (grey)
		*r = *g = *b = v;
		return;
	}
	h /= 60.0;			// sector 0 to 5
	i = (int)floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );
	switch( i ) {
		case 0:
			*r = v;
			*g = t;
			*b = p;
			break;
		case 1:
			*r = q;
			*g = v;
			*b = p;
			break;
		case 2:
			*r = p;
			*g = v;
			*b = t;
			break;
		case 3:
			*r = p;
			*g = q;
			*b = v;
			break;
		case 4:
			*r = t;
			*g = p;
			*b = v;
			break;
		default:		// case 5:
			*r = v;
			*g = p;
			*b = q;
			break;
	}
}

static char *unique_color(int index, int colorspace)
{
    char color[8];
    double r,g,b;
    double h;

    h = 360.0 * (double)index / (double)colorspace;
    HSVtoRGB(&r, &g, &b, h, 0.2, 0.7);

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
    MPI_Comm_rank(MPI_COMM_WORLD, &size);
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

    nbfuncs = nbthreads;
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

void dague_prof_grapher_task(const dague_execution_context_t *context, int thread_id, int vp_id, int task_hash)
{
    char tmp[MAX_TASK_STRLEN], nmp[MAX_TASK_STRLEN];
    if( NULL != grapher_file ) {
        dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, context);
        dague_prof_grapher_taskid(context, nmp, MAX_TASK_STRLEN);
#if defined(DAGUE_SIM)
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\",fontcolor=\"black\",label=\"<%d/%d> %s [%d]\",tooltip=\"%s%d\"];\n",
                nmp, colors[context->function->function_id % nbfuncs], thread_id, vp_id, tmp, context->sim_exec_date, context->function->name, task_hash);
#else
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\",fontcolor=\"black\",label=\"<%d/%d> %s\",tooltip=\"%s%d\"];\n",
                nmp, colors[context->function->function_id % nbfuncs], thread_id, vp_id, tmp, context->function->name, task_hash);
#endif
        fflush(grapher_file);
    }
}

void dague_prof_grapher_dep(const dague_execution_context_t* from, const dague_execution_context_t* to,
                            int dependency_activates_task,
                            const dague_flow_t* origin_flow, const dague_flow_t* dest_flow)
{    
    char tmp[128];
    int index = 0;

    if( NULL != grapher_file ) {
        dague_prof_grapher_taskid( from, tmp, 128 );
        index = strlen(tmp);
        index += snprintf( tmp + index, 128 - index, " -> " );
        dague_prof_grapher_taskid( to, tmp + index, 128 - index - 4 );
        fprintf(grapher_file, 
                "%s [label=\"%s=>%s\" color=\"#%s\" style=\"solid\"]\n", 
                tmp, origin_flow->name, dest_flow->name,
                dependency_activates_task ? "00FF00" : "FF0000");
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
