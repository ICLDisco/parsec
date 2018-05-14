/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "data.h"
#include "parsec_prof_grapher.h"
#include "parsec_internal.h"
#if defined(PARSEC_PROF_TRACE)
#include "parsec/parsec_binary_profile.h"
#endif
#include "parsec/utils/colors.h"
#include "parsec/parsec_internal.h"
#include "parsec/parsec_description_structures.h"
#include "parsec/utils/debug.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

#include <errno.h>
#include <stdio.h>
#include <string.h>

#if defined(PARSEC_PROF_GRAPHER)

FILE *grapher_file = NULL;
static int nbfuncs = -1;
static char **colors = NULL;

void parsec_prof_grapher_init(const char *base_filename, int nbthreads)
{
    char *filename;
    int t, size = 1, rank = 0;

#if defined(DISTRIBUTED) && defined(PARSEC_HAVE_MPI)
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
        parsec_warning("Grapher:\tunable to create %s (%s) -- DOT graphing disabled", filename, strerror(errno));
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

char *parsec_prof_grapher_taskid(const parsec_task_t *task, char *tmp, int length)
{
    const parsec_task_class_t* tc = task->task_class;
    unsigned int i, index = 0;

    assert( NULL!= task->taskpool );
    index += snprintf( tmp + index, length - index, "%s_%u", tc->name, task->taskpool->taskpool_id );
    for( i = 0; i < tc->nb_parameters; i++ ) {
        index += snprintf( tmp + index, length - index, "_%d",
                           task->locals[tc->params[i]->context_index].value );
    }

    return tmp;
}

void parsec_prof_grapher_task(const parsec_task_t *context,
                             int thread_id, int vp_id, uint64_t task_hash)
{
    if( NULL != grapher_file ) {
        char tmp[MAX_TASK_STRLEN], nmp[MAX_TASK_STRLEN];
        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, context);
        parsec_prof_grapher_taskid(context, nmp, MAX_TASK_STRLEN);
#if defined(PARSEC_SIM)
#  if defined(PARSEC_PROF_TRACE)
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\","
                "fontcolor=\"black\",label=\"<%d/%d> %s [%d]\","
                "tooltip=\"hid=%u:did=%d:tname=%s:tid=%lu\"];\n",
                nmp, colors[context->task_class->task_class_id % nbfuncs],
                thread_id, vp_id, tmp, context->sim_exec_date,
                context->taskpool->taskpool_id,
                context->taskpool->profiling_array != NULL 
                    ? BASE_KEY(context->taskpool->profiling_array[2*context->task_class->task_class_id])
                    : -1,
                context->task_class->name,
                task_hash);
#  else
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\","
                "fontcolor=\"black\",label=\"<%d/%d> %s [%d]\","
                "tooltip=\"hid=%u:tname=%s:tid=%lu\"];\n",
                nmp, colors[context->task_class->task_class_id % nbfuncs],
                thread_id, vp_id, tmp, context->sim_exec_date,
                context->taskpool->taskpool_id,
                context->task_class->name,
                task_hash);
#  endif
#else
#  if defined(PARSEC_PROF_TRACE)
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\","
                "fontcolor=\"black\",label=\"<%d/%d> %s\","
                "tooltip=\"hid=%u:did=%d:tname=%s:tid=%lu\"];\n",
                nmp, colors[context->task_class->task_class_id % nbfuncs],
                thread_id, vp_id, tmp,
                context->taskpool->taskpool_id,
                context->taskpool->profiling_array != NULL 
                    ? BASE_KEY(context->taskpool->profiling_array[2*context->task_class->task_class_id])
                    : -1,
                context->task_class->name,
                task_hash);
#  else
        fprintf(grapher_file,
                "%s [shape=\"polygon\",style=filled,fillcolor=\"%s\","
                "fontcolor=\"black\",label=\"<%d/%d> %s\","
                "tooltip=\"hid=%u:tname=%s:tid=%lu\"];\n",
                nmp, colors[context->task_class->task_class_id % nbfuncs],
                thread_id, vp_id, tmp,
                context->taskpool->taskpool_id,
                context->task_class->name,
                task_hash);
#  endif
#endif
        fflush(grapher_file);
    }
}

void parsec_prof_grapher_dep(const parsec_task_t* from, const parsec_task_t* to,
                            int dependency_activates_task,
                            const parsec_flow_t* origin_flow, const parsec_flow_t* dest_flow)
{
    if( NULL != grapher_file ) {
        char tmp[128];
        int index = 0;

        parsec_prof_grapher_taskid( from, tmp, 128 );
        index = strlen(tmp);
        index += snprintf( tmp + index, 128 - index, " -> " );
        parsec_prof_grapher_taskid( to, tmp + index, 128 - index - 4 );
        fprintf(grapher_file,
                "%s [label=\"%s=>%s\",color=\"#%s\",style=\"%s\"]\n",
                tmp, origin_flow->name, dest_flow->name,
                dependency_activates_task ? "00FF00" : "FF0000",
                ((dest_flow->flow_flags == FLOW_ACCESS_NONE) ? "dotted":
                 (dest_flow->flow_flags == FLOW_ACCESS_RW) ? "solid" : "dashed"));
        fflush(grapher_file);
    }
}

void parsec_prof_grapher_fini(void)
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

#endif /* PARSEC_PROF_GRAPHER */
