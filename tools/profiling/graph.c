#include "graph.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <graphviz/gvc.h>

typedef struct {
    node_info_t   info;
    int           status;
    unsigned int  nbsucc;
    unsigned int *succ;
    unsigned int  nbpred;
    unsigned int *pred;

    Agnode_t     *rnode;
} inode_t;

/**
 * src and dst are two indexes in the nodes array,
 */
typedef struct {
    unsigned int src;
    unsigned int dst;
} iedge_t;

static unsigned int nb_nodes           = 0;
static unsigned int nb_allocated_nodes = 0;
static inode_t     **nodes             = NULL;
static unsigned int nb_edges           = 0;
static unsigned int nb_allocated_edges = 0;
static iedge_t     **edges             = NULL;

unsigned int add_node(node_info_t *info)
{
    unsigned int id;
    inode_t *n = (inode_t*)malloc(sizeof(inode_t));
    memcpy(&n->info, info, sizeof(node_info_t));

    n->status          = 0;
    n->nbsucc          = 0;
    n->succ            = NULL;
    n->nbpred          = 0;
    n->pred            = NULL;
    
    n->rnode           = NULL;

    if( nb_nodes + 1 > nb_allocated_nodes ) {
        nb_allocated_nodes = 2*nb_allocated_nodes + 1;
        nodes = (inode_t**)realloc(nodes, sizeof(inode_t*) * nb_allocated_nodes);
    }
    nodes[nb_nodes] = n;
    id = nb_nodes;
    nb_nodes++;
    return id;
}

unsigned int find_node_by_task_id(const char *task_id)
{
    unsigned int i;
    for(i = 0; i < nb_nodes; i++) {
        if( !strcmp(task_id, nodes[i]->info.task_id) )
            return i;
    }
    return NID;
}

unsigned int find_node_by_task_name_and_parameters(const char *name, const char *parameters)
{
    unsigned int i;
    for(i = 0; i < nb_nodes; i++) {
        if( !strcmp(name, nodes[i]->info.task_name) &&
            !strcmp(parameters, nodes[i]->info.task_parameters) )
            return i;
    }
    return NID;
}

unsigned int find_node_by_task_name_and_object_id(const char *name, unsigned long long oid)
{
    unsigned int i;
    for(i = 0; i < nb_nodes; i++) {
        if( !strcmp(name, nodes[i]->info.task_name) &&
            (oid == nodes[i]->info.object_id) )
            return i;
    }
    return NID;
}

void get_node_info(unsigned int node, node_info_t *info)
{
    assert( node < nb_nodes );
    memcpy(info, &nodes[node]->info, sizeof(node_info_t));
}

unsigned int add_edge(unsigned int src, unsigned int dst)
{
    unsigned int id;
    assert( src < nb_nodes );
    assert( dst < nb_nodes );

    if( nb_edges + 1 > nb_allocated_edges ) {
        nb_allocated_edges = 2*nb_allocated_edges + 1;
        edges = (iedge_t**)realloc(edges, sizeof(iedge_t*) * nb_allocated_edges);
    }
    id = nb_edges;
    edges[id] = (iedge_t*)malloc(sizeof(iedge_t));
    edges[id]->src = src;
    edges[id]->dst = dst;
    nb_edges++;

    nodes[src]->nbsucc++;
    nodes[src]->succ = (unsigned int*)realloc(nodes[src]->succ, nodes[src]->nbsucc * sizeof(unsigned int));
    nodes[src]->succ[nodes[src]->nbsucc-1] = dst;

    nodes[dst]->nbpred++;
    nodes[dst]->pred = (unsigned int*)realloc(nodes[dst]->pred, nodes[dst]->nbpred * sizeof(unsigned int));
    nodes[dst]->pred[nodes[dst]->nbpred-1] = src;

    return id;
}

static int find_string(char **types, int nbtypes, char *test)
{
    int i;
    for(i = 0; i < nbtypes; i++)
        if( !strcmp(types[i], test) )
            return 1;
    return 0;
}

void add_key_nodes(void)
{
    node_info_t ni;
    unsigned int nid;

    ni.task_id = strdup("KEY_NOT_DISCOVERED");
    ni.task_name = strdup("Not discovered");
    ni.task_parameters = strdup("");
    ni.node = strdup("");
    ni.vp = strdup("");
    ni.thread = strdup("");
    ni.object_id = 0;
    ni.priority = 0;
    nid = add_node(&ni);

    ni.task_id = strdup("KEY_RUNNING");
    ni.task_name = strdup("Running");
    ni.task_parameters = strdup("");
    ni.node = strdup("");
    ni.vp = strdup("");
    ni.thread = strdup("");
    ni.object_id = 0;
    ni.priority = 0;
    nid = add_node(&ni);
    set_node_status(nid, STATUS_RUNNING);

    ni.task_id = strdup("KEY_READY");
    ni.task_name = strdup("Ready");
    ni.task_parameters = strdup("");
    ni.node = strdup("");
    ni.vp = strdup("");
    ni.thread = strdup("");
    ni.object_id = 0;
    ni.priority = 0;
    nid = add_node(&ni);
    set_node_status(nid, STATUS_READY);

    ni.task_id = strdup("KEY_ENABLED");
    ni.task_name = strdup("Enabled");
    ni.task_parameters = strdup("");
    ni.node = strdup("");
    ni.vp = strdup("");
    ni.thread = strdup("");
    ni.object_id = 0;
    ni.priority = 0;
    nid = add_node(&ni);
    set_node_status(nid, STATUS_ENABLED);

    ni.task_id = strdup("KEY_DONE");
    ni.task_name = strdup("Done");
    ni.task_parameters = strdup("");
    ni.node = strdup("");
    ni.vp = strdup("");
    ni.thread = strdup("");
    ni.object_id = 0;
    ni.priority = 0;
    nid = add_node(&ni);
    set_node_status(nid, STATUS_DONE);
}

int add_nodes_from_dotfile(const char *filename, int fileidx,
                           char **types, int nbtypes)
{
    FILE *f;
    char line[4096];
    char id[4096], check[4096], taskname[4096], parameters[4096];
    int vp, thread, priority, object;
    char *l;
    node_info_t ni;
    unsigned int nid;
    int n = 0, s;
    unsigned long long oid;

    f = fopen(filename, "r");
    if( f == NULL ) {
        perror(filename);
        return 0;
    }

    while( !feof(f) ) {
        fgets(line, 4096, f);
        for(s = 0, l = line; *l != '\n' && *l != '\0' && s < 4095; l++, s++) /*nothing*/;
        *l = '\0';
        assert( s < 4095 );

        if( sscanf(line, "%s [%[^=]", id, check) != 2 ||
            strcmp(check, "shape") )
            continue;
        l = strstr(line, "label=");
        if( NULL == l ) {
            fprintf(stderr, "Malformed node line in dot file %s: '%s' (can't find label=)\n",
                    filename, line);
            continue;
        }
        if( sscanf(l + 7, "<%d/%d> %[^(](%[^)])<%d>{%d}",
                   &thread, &vp, taskname, parameters, &priority, &object) != 6 ) {
            fprintf(stderr, "Malformed node line in dot file %s: label is '%s', expected <thread/vp> TASK(PARAMS)<priority>{object})\n",
                    filename, l);
            continue;
        }
        l = strstr(line, "tooltip=");
        if( NULL == l ) {
            fprintf(stderr, "Malformed node line in dot file %s: '%s' (can't find tooltip=)\n",
                    filename, line);
            continue;
        }
        if( sscanf(l+9, "%[^0-9]%llu", check, &oid) != 2 ) {
            fprintf(stderr, "Malformed node line in dot file %s: tooltip is '%s', expected <TASKNAME><OID>\n",
                    filename, l);
            continue;
        }

        if( types == NULL ||
            find_string(types, nbtypes, taskname) ) {
            ni.task_id = strdup(id);
            ni.task_name = strdup(taskname);
            ni.task_parameters = strdup(parameters);
            asprintf(&ni.node, "%d", fileidx);
            asprintf(&ni.vp, "%d", vp);
            asprintf(&ni.thread, "%d", thread);
            ni.object_id = oid;
            ni.priority = priority;
            (void)object;

            n++;
            nid = add_node(&ni); (void)nid;
        }
    }

    fclose(f);
    return n;
}

int add_edges_from_dotfile(const char *filename)
{
    FILE *f;
    char line[4096];
    char id1[4096], id2[4096], *l;
    unsigned int n1, n2, e = 0, s;

    f = fopen(filename, "r");
    if( f == NULL ) {
        perror(filename);
        return 0;
    }
    
    while( !feof(f) ) {
        fgets(line, 4096, f);
        for(s = 0, l = line; *l != '\n' && *l != '\0' && s < 4095; l++, s++) /*nothing*/;
        *l = '\0';
        assert( s < 4095 );
        if( sscanf(line, "%s -> %s", id1, id2) != 2 )
            continue;

        n1 = find_node_by_task_id(id1);
        if( n1 == NID ) {
            continue;
        }
        n2 = find_node_by_task_id(id2);
        if( n2 == NID ) {
            continue;
        }
        
        add_edge(n1, n2);
        e++;
    }

    fclose(f);
    return e;
}

static void update_node_display(unsigned int node);

void set_node_status(unsigned int node, int status_bits)
{
    assert(node < nb_nodes);
    nodes[node]->status |= status_bits;
    
    update_node_display(node);
}

void clear_node_status(unsigned int node, int status_bits)
{
    assert(node < nb_nodes);
    nodes[node]->status &= ~status_bits;

    update_node_display(node);
}

void update_neighbors_status(unsigned int node)
{
    unsigned int i, j;
    inode_t *n;

    assert( node < nb_nodes );
    assert( nodes[node]->status & STATUS_DONE );
    
    for(i = 0; i < nodes[node]->nbsucc; i++) {
        n = nodes[ nodes[node]->succ[i] ];
        if( n->status & (STATUS_DONE|STATUS_RUNNING) )
            continue;
        set_node_status(nodes[node]->succ[i], STATUS_ENABLED);
        for(j = 0; j < n->nbpred; j++) {
            if( nodes[ n->pred[j] ]->status & STATUS_DONE )
                continue;
            break;
        }
        if( j == n->nbpred )
            set_node_status(nodes[node]->succ[i], STATUS_READY);
    }
}

static GVC_t *gvc;

static void update_node_display(unsigned int node)
{
    int status;
    Agnode_t *nn;

    assert(node < nb_nodes);
    if( (nn = nodes[node]->rnode) != NULL ) {
        status = nodes[node]->status;
        agsafeset(nn, "style", "filled", "");
        if( status & STATUS_RUNNING ) {
            agsafeset(nn, "color", "red", "");
            agsafeset(nn, "fillcolor", "red", "");
            agsafeset(nn, "fontcolor", "white", "");
        } else if( status & STATUS_READY ) {
            agsafeset(nn, "color", "green", "");
            agsafeset(nn, "fillcolor", "green", "");
            agsafeset(nn, "fontcolor", "black", "");
        } else if( status & STATUS_ENABLED ) {
            agsafeset(nn, "color", "blue", "");
            agsafeset(nn, "fillcolor", "blue", "");
            agsafeset(nn, "fontcolor", "white", "");
        } else if( status & STATUS_DONE ) {
            agsafeset(nn, "color", "grey", "");
            agsafeset(nn, "fillcolor", "grey", "");
            agsafeset(nn, "fontcolor", "black", "");
        }  else {
            agsafeset(nn, "color", "black", "");
            agsafeset(nn, "fillcolor", "white", "");
            agsafeset(nn, "fontcolor", "black", "");            
        }
    }
}

void graphInit(void)
{
#ifdef WITH_CGRAPH
    gvc = gvContextPlugins(NULL, 1);
#else
    gvc = gvContext();
#endif
}

int graphFini(void)
{
    return gvFreeContext(gvc);
}

static Agnode_t *node(Agraph_t *g, char *name)
{
#ifdef WITH_CGRAPH
    return agnode(g, name, 1);
#else
    // creating a protonode is not permitted
    if (name[0] == '\001' && strcmp (name, "\001proto") == 0)
        return NULL;
    return agnode(g, name);
#endif
}

static Agedge_t *edge(Agnode_t *t, Agnode_t *h)
{
#ifdef WITH_CGRAPH
    // edges from/to the protonode are not permitted
    if (AGTYPE(t) == AGRAPH || AGTYPE(h) == AGRAPH)
        return NULL;
    return agedge(agraphof(t), t, h, NULL, 1);
#else
    // edges from/to the protonode are not permitted
    if ((agnameof(t)[0] == '\001' && strcmp (agnameof(t), "\001proto") == 0)
      || (agnameof(h)[0] == '\001' && strcmp (agnameof(h), "\001proto") == 0))
        return NULL;
    return agedge(t->graph, t, h);
#endif
}

static char *nodename(unsigned int n)
{
    char *name;

    assert( n < nb_nodes );
    asprintf(&name, "%s(%s)", nodes[n]->info.task_name, nodes[n]->info.task_parameters);
    return name;
}

static void createNodesAtDistanceOf(Agraph_t *g, unsigned int n, int distance)
{
    unsigned int i;
    Agnode_t *nn;

    if( distance < 0 )
        return;

    if( nodes[n]->rnode == NULL ) {
        nn = node(g, nodes[n]->info.task_id);
        agsafeset(nn, "label", nodename(n), "");
        nodes[n]->rnode = nn;
        if( nodes[n]->status & STATUS_RUNNING ) {
            agsafeset(nn, "color", "red", "");
        } else if( nodes[n]->status & STATUS_READY ) {
            agsafeset(nn, "color", "green", "");
        } else if( nodes[n]->status & STATUS_ENABLED ) {
            agsafeset(nn, "color", "blue", "");
        } else if( nodes[n]->status & STATUS_DONE ) {
            agsafeset(nn, "color", "grey", "");
        }

        if( nodes[n]->rnode == NULL ) {
            fprintf(stderr, "Fatal error while creating node %s: %s\n",
                    nodename(n), aglasterr());
            assert(0);
            exit(1);
        }
    }

    for(i = 0; i < nodes[n]->nbpred; i++)
        createNodesAtDistanceOf(g, nodes[n]->pred[i], distance-1);
    for(i = 0; i < nodes[n]->nbsucc; i++)
        createNodesAtDistanceOf(g, nodes[n]->succ[i], distance-1);
}

static Agraph_t *persistentGraph = NULL;

void graphRenderStatusAtDistance(int status_bits, int distance,
                                 char **result, unsigned int *length)
{
    unsigned int i, j;
    Agraph_t *g;

#ifdef WITH_CGRAPH
    g = agopen("g", Agdirected, 0);
#else
    g = agopen("g", AGDIGRAPH);
#endif
    agsafeset(g, "size", "30", "");

    for(i = 0; i < nb_nodes; i++) {
        if( nodes[i]->status & status_bits ) {
            createNodesAtDistanceOf(g, i, distance);
        }
    }
    for(i = 0; i < nb_nodes; i++) {
        if( nodes[i]->rnode != NULL ) {
            for(j = 0; j < nodes[i]->nbsucc; j++) {
                if( nodes[ nodes[i]->succ[j] ]->rnode != NULL ) {
                    (void)edge(nodes[i]->rnode, nodes[ nodes[i]->succ[j] ]->rnode);
                }
            }
        }
    }

    gvLayout(gvc, g, "dot");
    gvRenderData(gvc, g, "png", result, length);
    gvFreeLayout(gvc, g);

    for(i = 0; i < nb_nodes; i++) {
        nodes[i]->rnode = NULL;
    }

    agclose(g);
}

void persistentGraphLayoutEntireGraph(void)
{
    Agnode_t *nn;
    unsigned int i;

    if( NULL != persistentGraph ) {
        persistentGraphClose();
    }

#ifdef WITH_CGRAPH
    persistentGraph = agopen("persistentGraph", Agdirected, 0);
#else
    persistentGraph = agopen("persistentGraph", AGDIGRAPH);
#endif
    agsafeset(persistentGraph, "size", "30", "");

    for(i = 0; i < nb_nodes; i++) {
        nn = node(persistentGraph, nodes[i]->info.task_id);
        agsafeset(nn, "label", nodename(i), "");
        nodes[i]->rnode = nn;
        update_node_display( i );
    }
    for(i = 0; i < nb_edges; i++) {
        (void)edge( nodes[edges[i]->src]->rnode, nodes[edges[i]->dst]->rnode );
    }
    gvLayout(gvc, persistentGraph, "fdp");
}

void persistentGraphRender(char **result, unsigned int *length)
{
    gvLayout(gvc, persistentGraph, "nop");
    gvRenderData(gvc, persistentGraph, "png", result, length);
}

void persistentGraphClose(void)
{
    unsigned int i;

    gvFreeLayout(gvc, persistentGraph);
    
    for(i = 0; i < nb_nodes; i++) {
        nodes[i]->rnode = NULL;
    }
    
    agclose(persistentGraph);
    persistentGraph = NULL;
}
