#include "dague_config.h"
#include "dague.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#include "scheduling.h"
#include "fifo.h"
#include "remote_dep.h"


struct node;
struct edge;

typedef struct edge {
    dague_list_item_t list_item;

    double cost;

    struct node *dst;
} edge_t;

typedef struct node {
    /* Only to be chained in the ready list */
    dague_list_item_t list_item;

    dague_execution_context_t  ctx;
    uint32_t                   hash;
    dague_fifo_t               out;
    int                        nbin;

    double       date;
    double       cost;
} node_t;

typedef struct critical_path_arg {
    double   *tasks_costs;
    double    comm_cost;
    node_t  **all_nodes;
    uint32_t  nb_nodes;
    uint32_t  nodes_pos;
    int       display_interval;
    struct timeval next_display;
} critical_path_arg_t;

static int same_contexts(const dague_execution_context_t *a, const dague_execution_context_t *b)
{
    if( a->function != b->function )
        return 0;
    if( memcmp(a->locals, b->locals, a->function->nb_parameters * sizeof(assignment_t)) )
        return 0;
    return 1;
}

static uint32_t context_hash(const dague_execution_context_t *c)
{
    uint32_t h = (uint32_t)( (intptr_t)c->function & ~(uint32_t)0) | ( (intptr_t)c->function >> 8 );
    int i;
    for(i = 0; i < c->function->nb_parameters; i++)
        h ^= c->locals[i].value << i;
    return h;
}

static node_t *new_empty_node(critical_path_arg_t *arg, const dague_execution_context_t *c, double *costs)
{
    node_t *n = (node_t*)calloc(1, sizeof(node_t));
    DAGUE_LIST_ITEM_SINGLETON( (dague_list_item_t*)n );
    memcpy(&n->ctx, c, sizeof(dague_execution_context_t));
    n->hash = context_hash(c);
    dague_fifo_construct( &n->out );
    n->date = 0.0;
    n->cost = costs[ c->function->function_id ];

    assert( arg->nodes_pos < arg->nb_nodes );
    arg->all_nodes[arg->nodes_pos++] = n;

    

    return n;
}

static node_t *find_node(critical_path_arg_t *arg, const dague_execution_context_t *src)
{
    uint32_t n;
    uint32_t h = context_hash(src);
    for(n = 0; n < arg->nodes_pos; n++) {
        if( arg->all_nodes[n]->hash == h &&
            same_contexts(src, &arg->all_nodes[n]->ctx) )
            return arg->all_nodes[n];
    }
    return NULL;
}

static void display_progress(critical_path_arg_t *arg, char *welcome)
{
    struct timeval now;
    if( arg->display_interval >= 0 ) {
        gettimeofday( &now, NULL );
        if( timercmp( &now, &arg->next_display, >= ) ) {
            fprintf(stderr, "\r[K%s: %g%% done", welcome, 100.0 * (double)arg->nodes_pos / (double)arg->nb_nodes );
            fflush(stderr);
            arg->next_display = now;
            arg->next_display.tv_sec += arg->display_interval;
        }
    }
}

static void display_clear()
{
    fprintf(stderr, "\r[K");
    fflush(stderr);
}

static dague_ontask_iterate_t critical_path_new_task(dague_execution_unit_t *eu, 
                                                     dague_execution_context_t *newcontext, 
                                                     dague_execution_context_t *oldcontext, 
                                                     int param_index, int outdep_index, 
                                                     int src_rank, int dst_rank,
                                                     dague_arena_t* arena,
                                                     void *param);

static void new_dep(dague_execution_unit_t *eu, 
                    critical_path_arg_t *arg,
                    dague_execution_context_t *src,
                    dague_execution_context_t *dst,
                    int same_node)
{
    node_t *s, *d;
    edge_t *e;
    
    display_progress(arg, "Unrolling DAG");

    s = find_node( arg, src );
    d = find_node( arg, dst );
    if( NULL == d ) {
        d = new_empty_node(arg, dst, arg->tasks_costs);

        d->ctx.function->iterate_successors(eu, &d->ctx, critical_path_new_task, arg);
    }

    e = (edge_t*)calloc(1, sizeof(edge_t));
    e->cost = same_node ? 0.0 : arg->comm_cost;

    e->dst = d;

    DAGUE_LIST_ITEM_SINGLETON( e );
    dague_fifo_push( &s->out, (dague_list_item_t *)e );
    d->nbin++;
}

static dague_ontask_iterate_t critical_path_new_task(dague_execution_unit_t *eu, 
                                                     dague_execution_context_t *newcontext, 
                                                     dague_execution_context_t *oldcontext, 
                                                     int param_index, int outdep_index, 
                                                     int src_rank, int dst_rank,
                                                     dague_arena_t* arena,
                                                     void *param)
{
    (void)param_index;
    (void)outdep_index;
    (void)arena;
    
    new_dep(eu, (critical_path_arg_t*)param, oldcontext, newcontext, src_rank == dst_rank);    

    return DAGUE_ITERATE_CONTINUE;
}

double dague_compute_critical_path( dague_execution_unit_t *eu_context,
                                    dague_execution_context_t *initial_tasks,
                                    double *tasks_costs,
                                    double  comm_cost,
                                    int     display_interval )
{
    dague_execution_context_t *c;
    node_t *node;
    uint32_t n;
    edge_t *e;
    uint32_t nbready;
    double maxdate;
    dague_fifo_t ready_tasks;
    double curdate;
    critical_path_arg_t arg;
    dague_object_t *obj;
    
    if( (dague_execution_context_t *)initial_tasks->list_item.list_next == initial_tasks )
        return 0.0;
    obj = ((dague_execution_context_t *)initial_tasks->list_item.list_next)->dague_object;

    dague_fifo_construct( &ready_tasks );

    arg.tasks_costs = tasks_costs;
    arg.comm_cost   = comm_cost;
    arg.nb_nodes = obj->nb_local_tasks;
    arg.nodes_pos = 0;
    arg.all_nodes = (node_t**)calloc(sizeof(node_t*), obj->nb_local_tasks);

    c = initial_tasks;
    do {
        (void)new_empty_node(&arg, c, tasks_costs);
        c = (dague_execution_context_t *)c->list_item.list_next;
        assert( obj == c->dague_object );
    } while( c != initial_tasks);
    
    arg.display_interval = display_interval;
    if( display_interval >= 0 ) {
        gettimeofday( &arg.next_display, NULL );
        arg.next_display.tv_sec += display_interval;
    } 

    c = initial_tasks;
    do {
        c->function->iterate_successors(eu_context,
                                        c,
                                        critical_path_new_task,
                                        &arg);
        c = (dague_execution_context_t *)c->list_item.list_next;
    } while( c!= initial_tasks );
    
    display_clear();

    assert( arg.nodes_pos == arg.nb_nodes );

    maxdate = 0.0;
    do {
        nbready = 0;

        display_progress(&arg, "Computing Critical Path");

        for(n = 0; n < arg.nodes_pos; n++) {
            node = arg.all_nodes[n];
            if( NULL == node )
                continue;

            if( 0 == node->nbin ) {
                arg.all_nodes[n] = NULL;
                DAGUE_LIST_ITEM_SINGLETON( node );
                dague_fifo_push( &ready_tasks, (dague_list_item_t*)node );
                nbready++;
            } 
        }

        while( (node = (node_t*)dague_fifo_pop(&ready_tasks)) != NULL ) {
            while( (e = (edge_t*)dague_fifo_pop(&node->out)) != NULL ) {
                curdate = node->date + node->cost;
                if( curdate > maxdate )
                    maxdate = curdate;
                e->dst->nbin--;
                if (curdate + e->cost > e->dst->date) {
                    e->dst->date = curdate + e->cost;
                    if( e->dst->date > maxdate )
                        maxdate = e->dst->date;
                }
            }
            free(e);
        }
        free(node);
    } while( nbready > 0 );

    display_clear();

    free( arg.all_nodes );

    return maxdate;
}
