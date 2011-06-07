#include "dague_config.h"
#include "dague_prof_grapher.h"

#if defined(DAGUE_PROF_GRAPHER)

#include <stdio.h>
#include <math.h>

static FILE *grapher_file = NULL;
static int nbthreads = -1;
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

void dague_prof_grapher_init(const char *base_filename, int rank, int size, int nb)
{
    char *filename;
    int t;

#if defined(DISTRIBUTED) && defined(HAVE_MPI)
    filename = malloc(strlen(base_filename) + 16);
    snprintf(filename, strlen(base_filename) + 16, "%s-%d.dot", base_filename, rank);
#else
    filename = malloc(strlen(base_filename) + 16);
    snprintf(filename, strlen(base_filename) + 16, "%s.dot", base_filename);
#endif

    grapher_file = fopen(filename, "w");
    if( NULL == grapher_file ) {
        fprintf(stderr, "Warning: unable to create %s -- DOT graphing disabled\n", filename);
        free(filename);
        return;
    }

    fprintf(grapher_file, "digraph G {\n");
    fflush(grapher_file);

    nbthreads = nb;
    colors = (char**)malloc(nbthreads * sizeof(char*));
    for(t = 0; t < nbthreads; t++)
        colors[t] = unique_color(rank * nbthreads + t, size * nbthreads);
}

static char *service_to_taskid(const dague_execution_context_t *exec_context, char *tmp, int length)
{
    const dague_t* function = exec_context->function;
    unsigned int i, index = 0;

    index += snprintf( tmp + index, length - index, "%s", function->name );
    for( i = 0; i < function->nb_locals; i++ ) {
        index += snprintf( tmp + index, length - index, "_%d",
                           exec_context->locals[i].value );
    }

    return tmp;
}

void dague_prof_grapher_task(const dague_execution_context_t *context, int thread_id, int task_hash)
{
    char tmp[128];
    char nmp[128];
    if( NULL != grapher_file ) {
        dague_service_to_string(context, tmp, 128);
        service_to_taskid(context, nmp, 128);
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\",fontcolor=\"black\",label=\"%s\",tooltip=\"%s%d\"];\n",
                nmp, colors[thread_id % nbthreads], tmp, context->function->name, task_hash);
        fflush(grapher_file);
    }
}

void dague_prof_grapher_dep(const dague_execution_context_t* from, const dague_execution_context_t* to,
                            int dependency_activates_task,
                            const param_t* origin_param, const param_t* dest_param)
{    
    char tmp[128];
    int index = 0;

    if( NULL != grapher_file ) {
        service_to_taskid( from, tmp, 128 );
        index = strlen(tmp);
        index += snprintf( tmp + index, 128 - index, " -> " );
        service_to_taskid( to, tmp + index, 128 - index - 4 );
        fprintf(grapher_file, 
                "%s [label=\"%s=>%s\" color=\"#%s\" style=\"solid\"]\n", 
                tmp, origin_param->name, dest_param->name,
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
    for(t = 0; t < nbthreads; t++)
        free(colors[t]);
    free(colors);
    colors = NULL;
    grapher_file = NULL;
}

#endif /* DAGUE_PROF_GRAPHER */
