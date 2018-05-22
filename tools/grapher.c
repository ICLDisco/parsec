/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/parsec_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>

#include "parsec/runtime.h"
#include "parsec/data_distribution.h"

static uint32_t pseudo_rank_of(struct parsec_dc *mat, ...)
{
    va_list ap;
    va_start(ap, mat);
    va_end(ap);
    return 0;
}

static void *pseudo_data_of(struct parsec_dc *mat, ...)
{
    va_list ap;
    va_start(ap, mat);
    va_end(ap);
    return NULL;
}

static parsec_data_collection_t pseudo_desc = {
    .myrank = 0,
    .cores = 1,
    .nodes = 1,
    .rank_of = pseudo_rank_of,
    .data_of = pseudo_data_of,
};

typedef struct {
    const char *command_name;
    parsec_taskpool_t *(*create_function)(int argc, char **argv);
} create_function_t;

#define TEST_SET(fname, vname) do {                                     \
        if( 0 == vname##_set ) {                                        \
            fprintf(stderr, fname ": unable to create this function, "  \
                    #vname" is not set\n");                             \
            allset = 0;                                                \
        }                                                               \
    }while(0)
#define TRY_SET(vname) do {                     \
        if( !strcmp(argv[i], #vname) ) {        \
            vname = atoi(argv[i+1]);            \
            vname##_set = 1;                    \
        }                                       \
    } while(0)
/*
 * The following file is auto generated with the
 * grapher_create_objects.sh script. It defines an array
 * create_functions of NB_CREATE_FUNCTIONS create_function_t, and the
 * value of NB_CREATE_FUNCTIONS. It uses the TEST_SET and TRY_SET 
 * macros.
 */
#include "grapher_create_objects.c"

typedef struct vertex_list_t {
    struct vertex_list_t *next;
    char                 *value;
} vertex_list_t;

typedef struct edge_list_t {
    const vertex_list_t *from;
    const vertex_list_t *to;
    struct edge_list_t *next;
} edge_list_t;

static vertex_list_t *vertices = NULL;
static edge_list_t *edges = NULL;

static vertex_list_t *lookup_create_vertex(const char *name)
{
    vertex_list_t *v, *p;

    p = NULL;
    for(v = vertices; NULL != v; v = v->next) {
        if( !strcmp(v->value, name) )
            return v;
        p = v;
    }

    v = (vertex_list_t*)calloc(1, sizeof(vertex_list_t));
    v->value = strdup(name);
    if( NULL == p )
        vertices = v;
    else
        p->next = v;
    return v;
}

static edge_list_t *lookup_create_edge(const vertex_list_t *from, const vertex_list_t *to)
{
    edge_list_t *e, *p;

    p = NULL;
    for(e = edges; NULL != e; e = e->next) {
        if( e->from == from && e->to == to )
            return e;
        p = e;
    }

    e = (edge_list_t*)calloc(1, sizeof(edge_list_t));
    e->from = from;
    e->to = to;
    if( NULL == p )
        edges = e;
    else
        p->next = e;
    return e;
}

static parsec_ontask_iterate_t ontask_function(struct parsec_execution_stream_s *es,
                                              parsec_task_t *newcontext,
                                              parsec_task_t *oldcontext,
                                              int flow_index, int outdep_index,
                                              int rank_src, int rank_dst,
                                              void *param)
{
    char fromstr[MAX_TASK_STRLEN];
    char tostr[MAX_TASK_STRLEN];
    vertex_list_t *from;
    vertex_list_t *to;

    (void)eu;
    (void)flow_index;
    (void)outdep_index;
    (void)rank_src;
    (void)rank_dst;
    (void)param;

    parsec_task_snprintf(fromstr, MAX_TASK_STRLEN, oldcontext);
    parsec_task_snprintf(tostr, MAX_TASK_STRLEN, newcontext);

    from = lookup_create_vertex(fromstr);
    to = lookup_create_vertex(tostr);
    lookup_create_edge(from, to);

    newcontext->function->iterate_successors(es, newcontext, ontask_function, NULL);

    return PARSEC_ITERATE_CONTINUE;
}

static int dump_graph(const char *filename)
{
    FILE *f;
    vertex_list_t *v;
    edge_list_t *e;

    f = fopen(filename, "w");
    if( NULL == f ) {
        fprintf(stderr, "unable to create %s: %s\n", filename, strerror(errno));
        return -1;
    }

    fprintf(f, "digraph {\n");
    for(v = vertices; NULL != v; v = v->next) {
        fprintf(f, "  V%p [label=\"%s\"];\n",
                v, v->value);
    }
    fprintf(f, "\n");
    for(e = edges; NULL != e; e = e->next) {
        fprintf(f, "  V%p -> V%p;\n", e->from, e->to);
    }
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

int main(int argc, char *argv[])
{
    int i;
    parsec_taskpool_t *o;
    parsec_task_t *startup;
    parsec_list_item_t *s;
    parsec_context_t *parsec;

    o = NULL;
    parsec = parsec_init( 1, &argc, &argv, 1 );
    for(i = 0; i < NB_CREATE_FUNCTIONS; i++) {
        if( !strcmp( create_functions[i].command_name, argv[1]) ) {
            o = create_functions[i].create_function(argc-2, argv+2);
            break;
        }
    }

    if( o == NULL ) {
        if(i == NB_CREATE_FUNCTIONS) {
            fprintf(stderr, "Error: unable to find the function '%s'. You must choose between:\n",
                    argv[1]);
            for(i = 0; i < NB_CREATE_FUNCTIONS; i++) {
                fprintf(stderr, "  %s\n", create_functions[i].command_name);
            }
        }
        return 1;
    }

    o->startup_hook( parsec->execution_streams[0], o, &startup );
    s = (parsec_list_item_t*)startup;
    do {
        char fromstr[MAX_TASK_STRLEN];
        parsec_task_snprintf(fromstr, MAX_TASK_STRLEN, (parsec_task_t*)s);
        lookup_create_vertex(fromstr);
        s = (parsec_list_item_t*)s->list_next;
    } while( s != (parsec_list_item_t*)startup );

    s = (parsec_list_item_t*)startup;
    do {
        ((parsec_task_t*)s)->function->iterate_successors(parsec->execution_streams[0], (parsec_task_t*)s, ontask_function, NULL);
        s = (parsec_list_item_t*)s->list_next;
    } while( s!= (parsec_list_item_t*)startup );

    dump_graph("grapher.dot");

    return 0;
}
