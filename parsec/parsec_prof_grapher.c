/*
 * Copyright (c) 2010-2019 The University of Tennessee and The University
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
#include "parsec/class/parsec_hash_table.h"
#include "parsec/data_distribution.h"
#include "parsec/data_internal.h"
#include "parsec/utils/mca_param.h"
#include "parsec/execution_stream.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#if defined(PARSEC_PROF_GRAPHER)

FILE *grapher_file = NULL;
static int nbfuncs = -1;
static char **colors = NULL;
static parsec_hash_table_t *data_ht = NULL;
static int parsec_prof_grapher_memmode = 0;

typedef struct {
    parsec_data_collection_t *dc;
    parsec_data_key_t         data_key;
} parsec_grapher_data_identifier_t;

typedef struct {
    parsec_hash_table_item_t         ht_item;
    parsec_grapher_data_identifier_t id;
    char                            *did;
} parsec_grapher_data_identifier_hash_table_item_t;

static int grapher_data_id_key_equal(parsec_key_t a, parsec_key_t b, void *unused)
{
    (void)unused;
    parsec_grapher_data_identifier_t *id_a = (parsec_grapher_data_identifier_t *)a;
    parsec_grapher_data_identifier_t *id_b = (parsec_grapher_data_identifier_t *)b;
    return (id_a->dc == id_b->dc) && (id_a->data_key == id_b->data_key);
}

static char *grapher_data_id_key_print(char *buffer, size_t buffer_size, parsec_key_t k, void *unused)
{
    parsec_grapher_data_identifier_t *id = (parsec_grapher_data_identifier_t*)k;
    (void)unused;
    if( NULL == id->dc )
        snprintf(buffer, buffer_size, "NEW key %"PRIuPTR, (uintptr_t)id->data_key);
    else if( NULL != id->dc->key_base )
        snprintf(buffer, buffer_size, "DC(%s) key %"PRIuPTR, id->dc->key_base, (uintptr_t)id->data_key);
    else
        snprintf(buffer, buffer_size, "Uknown DC(%p) key %"PRIuPTR, id->dc, (uintptr_t)id->data_key);
    return buffer;
}

static uint64_t grapher_data_id_key_hash(parsec_key_t key, void *unused)
{
    parsec_grapher_data_identifier_t *id = (parsec_grapher_data_identifier_t*)key;
    uint64_t k = 0;
    (void)unused;
    k = ((uintptr_t)id->dc) | ((uintptr_t)id->data_key);
    return k;
}

static parsec_key_fn_t parsec_grapher_data_key_fns = {
    .key_equal = grapher_data_id_key_equal,
    .key_print = grapher_data_id_key_print,
    .key_hash  = grapher_data_id_key_hash
};

void parsec_prof_grapher_init(const parsec_context_t *parsec_context, const char *base_filename, int nbthreads)
{
    char *filename;
    char *format;
    int t, l10 = 0, cs;

    cs = parsec_context->nb_nodes;
    while(cs > 0) {
      l10++;
      cs = cs/10;
    }
    asprintf(&format, "%%s-%%0%dd.dot", l10);
    asprintf(&filename, format, base_filename, parsec_context->my_rank);
    free(format);

    parsec_mca_param_reg_int_name("parsec_prof_grapher", "memmode", "How memory references are traced in the DAG of tasks "
                                 "(default is 0, possible values are 0: no tracing of memory references, 1: trace only the "
                                  "direct memory references, 2: trace memory references even when data is passed from task "
                                  "to task)",
                                  false, false, parsec_prof_grapher_memmode, &parsec_prof_grapher_memmode);
    
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

    srandom(parsec_context->nb_nodes*(parsec_context->my_rank+1));  /* for consistent color generation */
    (void)nbthreads;
    nbfuncs = 128;
    colors = (char**)malloc(nbfuncs * sizeof(char*));
    for(t = 0; t < nbfuncs; t++)
        colors[t] = parsec_unique_color(parsec_context->my_rank * nbfuncs + t, parsec_context->nb_nodes * nbfuncs);

    data_ht = PARSEC_OBJ_NEW(parsec_hash_table_t);
    parsec_hash_table_init(data_ht, offsetof(parsec_grapher_data_identifier_hash_table_item_t, ht_item), 16, parsec_grapher_data_key_fns, NULL);
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
                "tooltip=\"tpid=%u:did=%d:tname=%s:tid=%lu\"];\n",
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
                "tooltip=\"tpid=%u:tname=%s:tid=%lu\"];\n",
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
                "tooltip=\"tpid=%u:did=%d:tname=%s:tid=%lu\"];\n",
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
                "tooltip=\"tpid=%u:tname=%s:tid=%lu\"];\n",
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
                ((dest_flow->flow_flags == PARSEC_FLOW_ACCESS_NONE) ? "dotted":
                 (dest_flow->flow_flags == PARSEC_FLOW_ACCESS_RW) ? "solid" : "dashed"));
        fflush(grapher_file);
    }
}

static void parsec_prof_grapher_dataid(const parsec_data_t *dta, char *did, int size)
{
    parsec_grapher_data_identifier_t id;
    parsec_key_t key;
    parsec_grapher_data_identifier_hash_table_item_t *it;

    assert(NULL != dta);
    assert(NULL != grapher_file);
    assert(NULL != data_ht);
    
    id.dc = dta->dc;
    id.data_key = dta->key;
    key = (parsec_key_t)(uintptr_t)&id;
    parsec_hash_table_lock_bucket(data_ht, key);
    if( NULL == (it = parsec_hash_table_nolock_find(data_ht, key)) ) {
        char data_name[MAX_TASK_STRLEN];
        it = (parsec_grapher_data_identifier_hash_table_item_t*)malloc(sizeof(parsec_grapher_data_identifier_hash_table_item_t));
        it->id = id;
        it->ht_item.key = (parsec_key_t)(uintptr_t)&it->id;
        if(NULL != it->id.dc)
            asprintf(&it->did, "dc%p_%"PRIuPTR, it->id.dc, (uintptr_t)it->id.data_key);
        else
            asprintf(&it->did, "dta%p_%"PRIuPTR, dta, (uintptr_t)it->id.data_key);
        parsec_hash_table_nolock_insert(data_ht, &it->ht_item);
        parsec_hash_table_unlock_bucket(data_ht, key);

        if(NULL != dta->dc && NULL != dta->dc->key_to_string) {
            dta->dc->key_to_string(dta->dc, dta->key, data_name, MAX_TASK_STRLEN);
        } else {
            snprintf(data_name, MAX_TASK_STRLEN, "NEW");
        }
        fprintf(grapher_file, "%s [label=\"%s%s\",shape=\"circle\"]\n", it->did, NULL != dta->dc->key_base ? dta->dc->key_base : "", data_name);
    } else
        parsec_hash_table_unlock_bucket(data_ht, key);
    strncpy(did, it->did, size);
}

void  parsec_prof_grapher_data_input(const parsec_data_t *data, const parsec_task_t *task, const parsec_flow_t *flow, int direct_reference)
{
    if( NULL != grapher_file &&
        (( direct_reference == 1 && parsec_prof_grapher_memmode == 1 ) ||
         ( parsec_prof_grapher_memmode == 2 )) ) {
        char tid[128];
        char did[128];
        parsec_prof_grapher_taskid( task, tid, 128 );
        parsec_prof_grapher_dataid( data, did, 128 );
        fprintf(grapher_file, "%s -> %s [label=\"%s\"]\n", did, tid, flow->name);
        fflush(grapher_file);
    }
}

void  parsec_prof_grapher_data_output(const struct parsec_task_s *task, const struct parsec_data_s *data, const struct parsec_flow_s *flow)
{
    /* All output are direct references to a data */
    if( NULL != grapher_file &&
        (parsec_prof_grapher_memmode >= 1 ) ) {
        char tid[128];
        char did[128];
        parsec_prof_grapher_taskid( task, tid, 128 );
        parsec_prof_grapher_dataid( data, did, 128 );
        fprintf(grapher_file, "%s -> %s [label=\"%s\"]\n", tid, did, flow->name);
        fflush(grapher_file);
    }    
}

static void parsec_grapher_data_ht_free_elt(void *_item, void *table)
{
    parsec_grapher_data_identifier_hash_table_item_t *item = (parsec_grapher_data_identifier_hash_table_item_t*)_item;
    parsec_key_t key = (parsec_key_t)(uintptr_t)&item->id;
    parsec_hash_table_nolock_remove(table, key);
    free(item->did);
    free(item);
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

    /* Free all data records */
    parsec_hash_table_for_all(data_ht, parsec_grapher_data_ht_free_elt, data_ht);
    PARSEC_OBJ_RELEASE(data_ht);
    data_ht = NULL;
}

#endif /* PARSEC_PROF_GRAPHER */
