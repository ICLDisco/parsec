#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

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

static void display_node_list(nl_t *s)
{
    int i;
    for(i = 0; i < s->size; i++) {
        printf("%s[%s] ", s->node[i]->tname, s->node[i]->accesses);
    }
    printf("\n");
}

static void display_node_array(node_t **word, int len)
{
    int i;
    for(i = 0; i < len; i++) {
        printf("%s#%s# ", word[i]->tname, word[i]->accesses);
    }
    printf("\n");
}

static void walk(node_t **word, int pos, nl_t *ready) {
    int i, j, k;
    nl_t *myready;
    node_t *s, *e;

    //    printf("entering level %d with a list of size %d\n", pos, ready->size);

    if( ready->size == 0 ) {
        word[pos] = NULL;
        display_node_array(word, pos);
        return;
    }

    for(i = 0; i < ready->size; i++) {
        e = ready->node[i];
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
                nl_add(myready, s);
            }
        }
        walk(word, pos+1, myready);
        nl_free(myready);

        e->done = 0;
    }
}

static node_t ta_start = {
    .tname = "S#A",
    .accesses = "M0x1",
    .done = 0};
static node_t ta_end = {
    .tname = "E#A",
    .accesses = "M0x1",
    .done = 0};
static node_t tb_start = {
    .tname = "S#B",
    .accesses = "R0x1,W0x2",
    .done = 0};
static node_t tb_end = {
    .tname = "E#B",
    .accesses = "R0x1,W0x2",
    .done = 0};
static node_t tc_start = {
    .tname = "S#C",
    .accesses = "R0x1,W0x3",
    .done = 0};
static node_t tc_end = {
    .tname = "E#C",
    .accesses = "R0x1,W0x3",
    .done = 0};
static node_t td_start = {
    .tname = "S#D",
    .accesses = "R0x3,W0x1",
    .done = 0};
static node_t td_end = {
    .tname = "E#D",
    .accesses = "R0x3,W0x1",
    .done = 0};
static node_t te_start = {
    .tname = "S#E",
    .accesses = "R0x1,W0x3",
    .done = 0};
static node_t te_end = {
    .tname = "E#E",
    .accesses = "R0x1,W0x3",
    .done = 0};

nl_t *load_dummy_graph(int *nbnodes)
{
    nl_t *init = (nl_t*)malloc(sizeof(nl_t));
    init->node = (node_t**)malloc(sizeof(node_t*));
    init->size = 1;
    init->allocated = 1;
    
    ta_start.nbsucc = 1;
    ta_start.succ = (node_t**)malloc(1*sizeof(node_t));
    ta_start.succ[0] = &ta_end;
    ta_end.nbpred = 1;
    ta_end.pred = (node_t**)malloc(1*sizeof(node_t));
    ta_end.pred[0] = &ta_start;

    tb_start.nbsucc = 1;
    tb_start.succ = (node_t**)malloc(1*sizeof(node_t));
    tb_start.succ[0] = &tb_end;
    tb_end.nbpred = 1;
    tb_end.pred = (node_t**)malloc(1*sizeof(node_t));
    tb_end.pred[0] = &tb_start;

    tc_start.nbsucc = 1;
    tc_start.succ = (node_t**)malloc(1*sizeof(node_t));
    tc_start.succ[0] = &tc_end;
    tc_end.nbpred = 1;
    tc_end.pred = (node_t**)malloc(1*sizeof(node_t));
    tc_end.pred[0] = &tc_start;

    td_start.nbsucc = 1;
    td_start.succ = (node_t**)malloc(1*sizeof(node_t));
    td_start.succ[0] = &td_end;
    td_end.nbpred = 1;
    td_end.pred = (node_t**)malloc(1*sizeof(node_t));
    td_end.pred[0] = &td_start;

    te_start.nbsucc = 1;
    te_start.succ = (node_t**)malloc(1*sizeof(node_t));
    te_start.succ[0] = &te_end;
    te_end.nbpred = 1;
    te_end.pred = (node_t**)malloc(1*sizeof(node_t));
    te_end.pred[0] = &te_start;

    ta_end.nbsucc = 2;
    ta_end.succ = (node_t**)malloc(2*sizeof(node_t));
    ta_end.succ[0] = &tb_start;
    ta_end.succ[1] = &tc_start;
    tb_start.nbpred = 1;
    tb_start.pred = (node_t**)malloc(1*sizeof(node_t));
    tb_start.pred[0] = &ta_end;
    tc_start.nbpred = 1;
    tc_start.pred = (node_t**)malloc(1*sizeof(node_t));
    tc_start.pred[0] = &ta_end;
    
    tb_end.nbsucc = 1;
    tb_end.succ = (node_t**)malloc(1*sizeof(node_t));
    tb_end.succ[0] = &te_start;
    te_start.nbpred = 2;
    te_start.pred = (node_t**)malloc(1*sizeof(node_t));
    te_start.pred[0] = &tb_end;
    
    tc_end.nbsucc = 1;
    tc_end.succ = (node_t**)malloc(1*sizeof(node_t));
    tc_end.succ[0] = &td_start;
    td_start.nbpred = 1;
    td_start.pred = (node_t**)malloc(1*sizeof(node_t));
    td_start.pred[0] = &tc_end;
    
    td_end.nbsucc = 1;
    td_end.succ = (node_t**)malloc(1*sizeof(node_t));
    td_end.succ[0] = &te_start;
    te_start.pred[1] = &td_end;

    init->node[0] = &ta_start;
    *nbnodes = 10;
    return init;
}

int main(int argc, char *argv[])
{
    nl_t *graph;
    node_t **word;
    int nb_nodes;
    graph = load_dummy_graph(&nb_nodes);
    word = (node_t**)calloc(nb_nodes+1, sizeof(node_t*));
    walk(word, 0, graph);
    return 0;
}
