#define _GNU_SOURCE
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct {
    off_t         tname;
    off_t         accesses;
    int           nbsucc;
    int           succ[1];
} filenode_t;

typedef struct {
    int           nbnodes;
    off_t         nodes[1];
} filenode_header_t;

typedef struct node {
    char         *tname;
    char         *accesses;
    int           done;
    int           nbsucc;
    struct node **succ;
    int           nbpred;
    struct node **pred;
} node_t;

typedef struct {
    node_t  **node;
    int       size;
    int       allocated;
} nl_t;

static void nl_add(nl_t *s, node_t *n);

static void dump_dot(node_t *nodes, int nbnodes)
{
    int i, k;
    printf("digraph G {\n");
    for(i = 0; i < nbnodes; i++) {
        fprintf(stderr, "%s has %d successors\n",  nodes[i].tname, nodes[i].nbsucc);
        for(k = 0; k < nodes[i].nbsucc; k++)
            printf("  %s -> %s;\n",
                   nodes[i].tname, nodes[i].succ[k]->tname);
    }
    printf("}\n");
}

static void filenode_add_pred(node_t *n, node_t *p)
{
    n->nbpred++;
    n->pred = (node_t**)realloc( n->pred, n->nbpred * sizeof(node_t*) );
    n->pred[n->nbpred-1] = p;
}

static off_t load_single_node(char *m, off_t offset, int i, node_t *allnodes)
{
    node_t *s, *e;
    int j;
    char *acc, *name;

    name = (char*)(m + offset);
    offset += strlen( name ) + 1;

    acc = strdup( (char*)(m + offset) );
    offset += strlen( acc ) + 1;

    s = &allnodes[2*i];
    e = &allnodes[2*i+1];

    asprintf(&s->tname, "S#%s", name);
    s->accesses = acc;
    
    s->nbsucc = 1;
    s->succ = (node_t**)malloc( sizeof(node_t*) );
    s->succ[0] = e;

    e->accesses = acc;
    asprintf(&e->tname, "E#%s", name);
    e->nbpred = 1;
    e->pred = (node_t**)malloc( sizeof(node_t*) );
    e->pred[0] = s;

    e->nbsucc = ((int*)(m + offset))[0];
    offset += sizeof(int);

    if( e->nbsucc > 0 ) {
        e->succ = (node_t**)malloc( e->nbsucc * sizeof(node_t*) );
        for(j = 0; j < e->nbsucc; j++) {
            int succ = ((int*)(m + offset))[0];
            offset += sizeof(int);
            e->succ[j] = &(allnodes[ 2*succ ]);
            filenode_add_pred( &(allnodes[ 2*succ ]), e );
        }
    } else {
        e->succ = NULL;
    }
    return offset;
}

static nl_t *load_filenode(const char *filename, int *nbnodes)
{
    int fd, i;
    struct stat s;
    char *m;
    nl_t *r = NULL;
    node_t *allnodes;
    off_t offset;

    if( (fd = open(filename, O_RDONLY)) == -1 ) {
        perror(filename);
        return r;
    }

    if( fstat(fd, &s) == -1 ) {
        perror(filename);
        close(fd);
        return r;
    }

    assert( s.st_size % getpagesize() == 0 );

    if( (m = mmap(NULL, s.st_size, PROT_READ, MAP_PRIVATE, fd, 0)) == MAP_FAILED ) {
        perror(filename);
        close(fd);
        return r;
    }
    close(fd);

    *nbnodes = ((int*)m)[0];

    r = (nl_t*)malloc(sizeof(nl_t));
    r->size = 0;
    r->allocated = 1;
    r->node = (node_t**)malloc(1 * sizeof(node_t*));

    allnodes = (node_t*)calloc(2 * (*nbnodes), sizeof(node_t));

    offset = sizeof(int);
    for(i = 0; i < *nbnodes; i++) {
        offset = load_single_node(m, offset, i, allnodes);
    }

    *nbnodes *= 2;

    for(i = 0; i < *nbnodes; i++) {
        if( i % 2 == 0 ) {
            assert( allnodes[i].nbsucc == 1 );
            assert( allnodes[i].succ[0] == &allnodes[i+1] );
        } else {
            assert( allnodes[i].nbpred == 1 );
            assert( allnodes[i].pred[0] == &allnodes[i-1] );
        }
        if( allnodes[i].nbpred == 0 ) {
            nl_add(r, &allnodes[i]);
        }
    }

    munmap( m, s.st_size );
    
    return r;
}

static nl_t *nl_dup(const nl_t *s)
{
    nl_t *n = (nl_t*)malloc(sizeof(nl_t));
    n->node = (node_t**)malloc(s->allocated * sizeof(node_t*));
    memcpy(n->node, s->node, s->size * sizeof(node_t*));
    n->size = s->size;
    n->allocated = s->allocated;
    return n;
}

static void nl_free(nl_t *s)
{
    free(s->node);
    s->node = NULL;
    s->size = -1;
    s->allocated = -1;
    free(s);
}

static void nl_remove(nl_t *s, int p)
{
    assert( p>=0 && p < s->size );
    s->node[p] = s->node[s->size-1];
    s->size--;
}

static void nl_add(nl_t *s, node_t *n)
{
    if( s->size == s->allocated ) {
        s->allocated = s->allocated*2;
        s->node = (node_t**)realloc(s->node, s->allocated * sizeof(node_t*));
    }
    s->node[s->size] = n;
    s->size++;
}

static void add_ready_task(nl_t *s, node_t *n)
{
    int i;
    for(i = 0; i < s->size; i++)
        assert( s->node[i] != n );

    nl_add(s, n);
}

static void display_node_list(nl_t *s)
{
    int i;
    for(i = 0; i < s->size; i++) {
        printf("%s[%s] ", s->node[i]->tname, s->node[i]->accesses);
    }
    printf("\n");
}

static unsigned long long int nbdone = 0;

static void display_node_array(node_t **word, int len)
{
    int i;
    nbdone++;
    if(nbdone == 1000000) exit(0);
    for(i = 0; i < len; i++) {
        printf("%s#%s# ", word[i]->tname, word[i]->accesses);
    }
    printf("\n");
}

static void walk(node_t **word, int nbnodes, int pos, nl_t *ready) {
    int i, j, k;
    nl_t *myready;
    node_t *s, *e;

    //    printf("entering level %d with a list of size %d\n", pos, ready->size);
    assert(pos <= nbnodes);

    if( ready->size == 0 ) {
        word[pos] = NULL;
        display_node_array(word, pos);
        return;
    }

    for(i = 0; i < ready->size; i++) {
        e = ready->node[i];

        assert( e->done == 0 );
        e->done = 1;
        word[pos] = e;

        myready = nl_dup(ready);
        //        printf("Ready at this level: "); display_node_list(myready);
        nl_remove(myready, i);
        for(j = 0; j < e->nbsucc; j++) {
            s = e->succ[j];
            if( s->done == 1 ) {
                continue;
            }
            for(k = 0; k < s->nbpred; k++) {
                if( s->pred[k]->done == 0 )
                    break;
            }
            if( k == s->nbpred ) {
                //                printf("%s is now ready\n", s->tname);
                add_ready_task(myready, s);
            }
        }
        walk(word, nbnodes, pos+1, myready);
        nl_free(myready);

        e->done = 0;
    }
}

int main(int argc, char *argv[])
{
    nl_t *graph;
    node_t **word;
    int nb_nodes;
    graph = load_filenode("dummy.grp", &nb_nodes);
    if( NULL == graph )
        return 1;
    word = (node_t**)calloc(nb_nodes+1, sizeof(node_t*));
    walk(word, nb_nodes, 0, graph);
    return 0;
}
